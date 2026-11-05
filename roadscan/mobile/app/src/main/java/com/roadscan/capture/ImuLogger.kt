package com.roadscan.capture

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import kotlin.math.abs

/**
 * Writes imu.csv: `t,ax,ay,az,gx,gy,gz`, one row per accelerometer sample (~100 Hz).
 *
 * Accelerometer is raw m/s² with gravity included; gyroscope is rad/s. The two sensors report
 * independently, so each row carries the most recent gyroscope reading (at most ~10 ms old).
 * On a phone with no gyroscope the g columns are left empty.
 */
class ImuLogger(context: Context, private val clock: SessionClock, private val log: CsvLog) : SensorEventListener {
    private val sensors = context.getSystemService(SensorManager::class.java)!!
    private val thread = HandlerThread("imu")
    private val gyro = FloatArray(3)
    private var hasGyroSensor = false
    private var gyroSeen = false

    // Sensor timestamps share the elapsedRealtimeNanos base on every device that follows the CDD.
    // A few old devices used a different base; if the first event is more than a second off,
    // shift every event by that offset instead of writing nonsense times.
    private var timestampOffsetNs: Long? = null

    /** Returns a warning to show the rider, or null if both sensors started. */
    fun start(): String? {
        val accel = sensors.getDefaultSensor(Sensor.TYPE_ACCELEROMETER) ?: return "No accelerometer: imu.csv will be empty"
        val gyroscope = sensors.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        hasGyroSensor = gyroscope != null
        thread.start()
        val handler = Handler(thread.looper)
        sensors.registerListener(this, accel, SAMPLING_PERIOD_US, handler)
        gyroscope?.let { sensors.registerListener(this, it, SAMPLING_PERIOD_US, handler) }
        return if (hasGyroSensor) null else "No gyroscope: gx,gy,gz will be empty"
    }

    fun stop() {
        sensors.unregisterListener(this)
        thread.quitSafely()
    }

    override fun onSensorChanged(event: SensorEvent) {
        val offset = timestampOffsetNs ?: (SystemClock.elapsedRealtimeNanos() - event.timestamp)
            .let { if (abs(it) > 1_000_000_000L) it else 0L }
            .also { timestampOffsetNs = it }

        when (event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                event.values.copyInto(gyro, endIndex = 3)
                gyroSeen = true
            }
            Sensor.TYPE_ACCELEROMETER -> {
                // Wait for the first gyroscope reading so no row has stale zeros.
                if (hasGyroSensor && !gyroSeen) return
                val t = clock.t(event.timestamp + offset)
                if (t < 0) return
                val v = event.values
                val g = if (gyroSeen) "${fmt(gyro[0], 5)},${fmt(gyro[1], 5)},${fmt(gyro[2], 5)}" else ",,"
                log.row("${fmt(t, 6)},${fmt(v[0], 4)},${fmt(v[1], 4)},${fmt(v[2], 4)},$g")
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit

    companion object {
        // 100 Hz. Rates above 200 Hz would need the HIGH_SAMPLING_RATE_SENSORS permission.
        private const val SAMPLING_PERIOD_US = 10_000
    }
}
