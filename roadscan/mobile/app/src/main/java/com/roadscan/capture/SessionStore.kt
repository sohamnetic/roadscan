package com.roadscan.capture

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.zip.Deflater
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.math.roundToLong

/** Everything that goes into meta.json. */
data class SessionMeta(
    val sessionId: String,
    val vehicle: Vehicle,
    val device: String,
    val startWallMs: Long,
    val width: Int,
    val height: Int,
    val fov: FieldOfView,
    val heightM: Float,
    val pitchDeg: Float,
    val videoT0: Double,
)

data class SessionInfo(
    val id: String,
    val dir: File,
    val sizeBytes: Long,
    val vehicle: String?,
    val startTimeUtc: String?,
    val hasVideo: Boolean,
    val hasMeta: Boolean,
)

/** Session folders under app-specific external storage: Android/data/<package>/files/sessions/<session_id>/. */
object SessionStore {
    private val ID_FORMAT = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss").withZone(ZoneOffset.UTC)
    private val UTC_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").withZone(ZoneOffset.UTC)

    fun root(context: Context): File =
        (context.getExternalFilesDir(null) ?: context.filesDir).resolve("sessions").apply { mkdirs() }

    /** e.g. `20261104-091502_bike`: sortable, and readable in a file listing. */
    fun newSessionId(vehicle: Vehicle, startWallMs: Long): String =
        "${ID_FORMAT.format(Instant.ofEpochMilli(startWallMs))}_${vehicle.id}"

    fun writeMeta(dir: File, meta: SessionMeta) {
        val json = JSONObject()
            .put("session_id", meta.sessionId)
            .put("vehicle_type", meta.vehicle.id)
            .put("device", meta.device)
            .put("start_time_utc", UTC_FORMAT.format(Instant.ofEpochMilli(meta.startWallMs)))
            .put(
                "camera",
                JSONObject()
                    .put("width", meta.width)
                    .put("height", meta.height)
                    .put("hfov_deg", round(meta.fov.hfovDeg, 2))
                    .put("vfov_deg", round(meta.fov.vfovDeg, 2))
                    .put("height_m", round(meta.heightM.toDouble(), 3))
                    .put("pitch_deg", round(meta.pitchDeg.toDouble(), 2)),
            )
            .put("video_t0", round(meta.videoT0, 6))
        // Write then rename, so a crash never leaves a half-written meta.json.
        val tmp = File(dir, "meta.json.tmp")
        tmp.writeText(json.toString(2), Charsets.UTF_8)
        if (!tmp.renameTo(File(dir, "meta.json"))) {
            File(dir, "meta.json").delete()
            tmp.renameTo(File(dir, "meta.json"))
        }
    }

    fun list(context: Context): List<SessionInfo> =
        root(context).listFiles { f -> f.isDirectory }.orEmpty()
            .sortedByDescending { it.name }
            .map { dir ->
                val metaFile = File(dir, "meta.json")
                val meta = runCatching { JSONObject(metaFile.readText()) }.getOrNull()
                SessionInfo(
                    id = dir.name,
                    dir = dir,
                    sizeBytes = dir.walkTopDown().filter { it.isFile }.sumOf { it.length() },
                    vehicle = meta?.optString("vehicle_type"),
                    startTimeUtc = meta?.optString("start_time_utc"),
                    hasVideo = File(dir, "video.mp4").length() > 0,
                    hasMeta = meta != null,
                )
            }

    fun delete(session: SessionInfo): Boolean = session.dir.deleteRecursively()

    /**
     * Zips the session folder as `<session_id>/...` into the cache and returns the zip. The video is
     * stored without compression (MP4 does not compress further and deflating it is slow). Earlier
     * exports are deleted first so they do not pile up next to multi-GB videos.
     */
    fun zip(context: Context, session: SessionInfo): File {
        val exports = File(context.cacheDir, "exports").apply {
            deleteRecursively()
            mkdirs()
        }
        val zip = File(exports, "${session.id}.zip")
        ZipOutputStream(zip.outputStream().buffered(256 * 1024)).use { out ->
            session.dir.listFiles { f -> f.isFile && !f.name.endsWith(".tmp") }.orEmpty().sortedBy { it.name }.forEach { file ->
                out.setLevel(if (file.extension == "mp4") Deflater.NO_COMPRESSION else Deflater.DEFAULT_COMPRESSION)
                out.putNextEntry(ZipEntry("${session.id}/${file.name}").apply { time = file.lastModified() })
                file.inputStream().buffered(256 * 1024).use { it.copyTo(out, 256 * 1024) }
                out.closeEntry()
            }
        }
        return zip
    }

    fun shareIntent(context: Context, zip: File): Intent {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", zip)
        val send = Intent(Intent.ACTION_SEND)
            .setType("application/zip")
            .putExtra(Intent.EXTRA_STREAM, uri)
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        return Intent.createChooser(send, "Export ${zip.name}")
    }

    private fun round(value: Double, decimals: Int): Double {
        val scale = Math.pow(10.0, decimals.toDouble())
        return (value * scale).roundToLong() / scale
    }
}
