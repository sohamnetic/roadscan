package com.roadscan.capture

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.MediaMetadataRetriever
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import android.util.Range
import android.view.Surface
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleService
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Records one ride: video.mp4 (CameraX), gps.csv, imu.csv and meta.json in a session folder.
 *
 * Runs as a camera + location foreground service with a partial wake lock, so recording carries
 * on with the screen off. Started and stopped through [start] / [stop]; status goes to [RecordingState].
 */
class RecordingService : LifecycleService() {
    private val main = Handler(Looper.getMainLooper())
    private var active: ActiveSession? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        super.onStartCommand(intent, flags, startId)
        when (intent?.action) {
            ACTION_START -> {
                // startForegroundService() requires startForeground() even if a ride is already running.
                goForeground()
                if (active == null) begin()
            }
            ACTION_STOP -> active?.requestStop() ?: stopSelf()
            else -> stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        active?.finish("Recording service was destroyed before Stop")
        super.onDestroy()
    }

    private fun begin() {
        val prefs = Prefs(this)
        val vehicle = prefs.vehicle
        val startNs = SystemClock.elapsedRealtimeNanos()
        val startWallMs = System.currentTimeMillis()
        val id = SessionStore.newSessionId(vehicle, startWallMs)
        val dir = File(SessionStore.root(this), id).apply { mkdirs() }
        active = ActiveSession(
            id = id,
            dir = dir,
            clock = SessionClock(startNs),
            startWallMs = startWallMs,
            vehicle = vehicle,
            heightM = prefs.heightM(vehicle),
            pitchDeg = prefs.pitchDeg,
        ).also { it.start() }
    }

    private fun goForeground() {
        val manager = getSystemService(NotificationManager::class.java)!!
        manager.createNotificationChannel(NotificationChannel(CHANNEL_ID, "Recording", NotificationManager.IMPORTANCE_LOW))
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, RecordingService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_record)
            .setContentTitle("Recording ride")
            .setContentText("Video, GPS and motion sensors")
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(0, "Stop", stop)
            .build()
        val types = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_CAMERA or ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION
        } else {
            0
        }
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, types)
    }

    private inner class ActiveSession(
        val id: String,
        val dir: File,
        val clock: SessionClock,
        val startWallMs: Long,
        val vehicle: Vehicle,
        val heightM: Float,
        val pitchDeg: Float,
    ) {
        private val gpsLog = CsvLog(File(dir, "gps.csv"), "t,lat,lon,accuracy_m,speed_mps,bearing_deg")
        private val imuLog = CsvLog(File(dir, "imu.csv"), "t,ax,ay,az,gx,gy,gz")
        private val gps = GpsLogger(this@RecordingService, clock, gpsLog) { loc ->
            RecordingState.update { it.copy(lastAccuracyM = if (loc.hasAccuracy()) loc.accuracy else null) }
        }
        private val imu = ImuLogger(this@RecordingService, clock, imuLog)
        private val wakeLock = getSystemService(PowerManager::class.java)!!
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "roadscan:recording")
        private val videoEvents: ExecutorService = Executors.newSingleThreadExecutor()
        private val warnings = mutableListOf<String>()

        private var cameraProvider: ProcessCameraProvider? = null
        private var videoCapture: VideoCapture<Recorder>? = null
        private var cameraInfo: CameraInfo? = null
        private var recording: Recording? = null
        private var stopRequested = false
        private var finished = false

        // elapsedRealtimeNanos of the first video frame; see onVideoEvent. Written on videoEvents, read on main.
        @Volatile private var videoT0Ns = Long.MAX_VALUE
        @Volatile private var videoBytes = 0L
        private var metaWritten = false

        private val ticker = object : Runnable {
            override fun run() {
                publish(if (recording != null) Phase.RECORDING else Phase.STARTING)
                if (!metaWritten && videoT0Ns != Long.MAX_VALUE &&
                    SystemClock.elapsedRealtimeNanos() - videoT0Ns > 3_000_000_000L
                ) {
                    // Write meta.json early, so a ride cut short by a crash still loads.
                    writeMeta()
                    metaWritten = true
                }
                main.postDelayed(this, 1_000L)
            }
        }

        fun start() {
            RecordingState.update {
                RecordingStatus(phase = Phase.STARTING, sessionId = id, vehicle = vehicle, lastSessionId = it.lastSessionId)
            }
            wakeLock.acquire(MAX_RIDE_MS)
            imu.start()?.let(warnings::add)
            gps.start()
            startCamera()
            main.post(ticker)
        }

        private fun startCamera() {
            val future = ProcessCameraProvider.getInstance(this@RecordingService)
            future.addListener({
                if (finished) return@addListener
                try {
                    val provider = future.get()
                    val recorder = Recorder.Builder()
                        .setQualitySelector(
                            QualitySelector.from(Quality.FHD, FallbackStrategy.lowerQualityOrHigherThan(Quality.FHD)),
                        )
                        .build()
                    val capture = VideoCapture.Builder(recorder)
                        .setTargetFrameRate(Range(30, 30))
                        // Stabilisation crops and warps the frame, which breaks the pinhole geometry in meta.json.
                        .setVideoStabilizationEnabled(false)
                        // Landscape, matching the activity's locked orientation.
                        .setTargetRotation(Surface.ROTATION_90)
                        .build()
                    provider.unbindAll()
                    val camera = provider.bindToLifecycle(this@RecordingService, CameraSelector.DEFAULT_BACK_CAMERA, capture)
                    cameraProvider = provider
                    videoCapture = capture
                    cameraInfo = camera.cameraInfo
                    // No audio: not needed by the pipeline, and it would record riders' conversations.
                    recording = recorder
                        .prepareRecording(this@RecordingService, FileOutputOptions.Builder(File(dir, "video.mp4")).build())
                        .start(videoEvents, ::onVideoEvent)
                    if (stopRequested) recording?.stop()
                } catch (e: Exception) {
                    Log.e(TAG, "Camera failed to start", e)
                    finish("Camera failed to start: ${e.message}")
                }
            }, ContextCompat.getMainExecutor(this@RecordingService))
        }

        /**
         * CameraX does not report the clock time of the first frame, so it is estimated from status
         * events: `now - recordedDuration` is the first frame's time plus however long the latest
         * frame took to reach this callback. The minimum over all events has the least of that delay.
         * It stays a little late (the camera-to-encoder latency, typically tens of milliseconds).
         */
        private fun onVideoEvent(event: VideoRecordEvent) {
            val now = SystemClock.elapsedRealtimeNanos()
            val stats = event.recordingStats
            if (stats.recordedDurationNanos > 0) {
                videoT0Ns = minOf(videoT0Ns, now - stats.recordedDurationNanos)
            }
            videoBytes = stats.numBytesRecorded
            if (event is VideoRecordEvent.Finalize) {
                val error = if (event.hasError()) "Video ended with error ${event.error}: ${event.cause?.message}" else null
                main.post { finish(error) }
            }
        }

        fun requestStop() {
            if (finished) return
            stopRequested = true
            publish(Phase.STOPPING)
            val rec = recording
            if (rec == null) {
                // The camera has not started yet (or failed); nothing to wait for.
                finish(null)
            } else {
                rec.stop() // finish() runs on the Finalize event
                main.postDelayed({ finish("Video did not finalise within 10 s; video.mp4 may be unplayable") }, 10_000L)
            }
        }

        fun finish(error: String?) {
            if (finished) return
            finished = true
            error?.let(warnings::add)
            main.removeCallbacks(ticker)
            recording?.close()
            videoCapture?.let { cameraProvider?.unbind(it) }
            gps.stop()
            imu.stop()
            gpsLog.close()
            imuLog.close()
            videoEvents.shutdown()
            runCatching { writeMeta() }.onFailure { warnings.add("Could not write meta.json: ${it.message}") }
            if (gpsLog.rows.get() < 2) warnings.add("Fewer than 2 GPS fixes: the pipeline will reject this session")
            if (wakeLock.isHeld) wakeLock.release()

            RecordingState.update {
                RecordingStatus(phase = Phase.IDLE, warnings = warnings.toList(), lastSessionId = id)
            }
            active = null
            ServiceCompat.stopForeground(this@RecordingService, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
        }

        private fun publish(phase: Phase) {
            val elapsedS = (SystemClock.elapsedRealtimeNanos() - clock.startNs) / 1_000_000_000L
            RecordingState.update {
                it.copy(
                    phase = if (it.phase == Phase.STOPPING) Phase.STOPPING else phase,
                    elapsedS = elapsedS,
                    gpsFixes = gpsLog.rows.get(),
                    imuRows = imuLog.rows.get(),
                    videoBytes = videoBytes,
                    videoT0 = videoT0Ns.takeIf { t -> t != Long.MAX_VALUE }?.let(clock::t),
                    warnings = warnings.toList(),
                )
            }
        }

        private fun writeMeta() {
            // The size actually written to the file once it is finalised; before that, the 1080p asked for.
            val (width, height) = videoSize() ?: (1920 to 1080)
            val fov = cameraInfo?.let { CameraGeometry.fieldOfView(it, width, height) }
                ?: CameraGeometry.fallback(width, height)
            val t0 = videoT0Ns.takeIf { it != Long.MAX_VALUE }?.let(clock::t)
            if (t0 == null && File(dir, "video.mp4").length() > 0) warnings.add("Could not time the first video frame; video_t0 set to 0")
            SessionStore.writeMeta(
                dir,
                SessionMeta(
                    sessionId = id,
                    vehicle = vehicle,
                    device = "${Build.MANUFACTURER} ${Build.MODEL}",
                    startWallMs = startWallMs,
                    width = width,
                    height = height,
                    fov = fov,
                    heightM = heightM,
                    pitchDeg = pitchDeg,
                    videoT0 = t0 ?: 0.0,
                ),
            )
        }

        /** Displayed (rotation-applied) size of video.mp4, or null until it is finalised. */
        private fun videoSize(): Pair<Int, Int>? {
            val file = File(dir, "video.mp4")
            if (!finished || file.length() == 0L) return null
            val retriever = MediaMetadataRetriever()
            return try {
                retriever.setDataSource(file.absolutePath)
                val w = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull()
                val h = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull()
                val rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
                when {
                    w == null || h == null -> null
                    rotation % 180 == 0 -> w to h
                    else -> h to w
                }
            } catch (e: Exception) {
                Log.w(TAG, "Could not read video size", e)
                null
            } finally {
                retriever.release()
            }
        }
    }

    companion object {
        private const val TAG = "RecordingService"
        private const val ACTION_START = "com.roadscan.capture.START"
        private const val ACTION_STOP = "com.roadscan.capture.STOP"
        private const val CHANNEL_ID = "recording"
        private const val NOTIFICATION_ID = 1
        private const val MAX_RIDE_MS = 6 * 60 * 60 * 1000L

        fun start(context: Context) {
            ContextCompat.startForegroundService(context, Intent(context, RecordingService::class.java).setAction(ACTION_START))
        }

        fun stop(context: Context) {
            context.startService(Intent(context, RecordingService::class.java).setAction(ACTION_STOP))
        }
    }
}
