package com.roadscan.capture

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.os.HandlerThread
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

/**
 * Writes gps.csv: `t,lat,lon,accuracy_m,speed_mps,bearing_deg` from the fused location provider,
 * high-accuracy priority, at least 1 Hz. `t` comes from [Location.getElapsedRealtimeNanos], the
 * same clock as the IMU. Fields the fix does not have (speed or bearing when stationary) are empty.
 */
class GpsLogger(
    context: Context,
    private val clock: SessionClock,
    private val log: CsvLog,
    private val onFix: (Location) -> Unit,
) {
    private val client = LocationServices.getFusedLocationProviderClient(context)
    private val thread = HandlerThread("gps")
    private var lastFixNs = Long.MIN_VALUE

    private val callback = object : LocationCallback() {
        override fun onLocationResult(result: LocationResult) {
            result.locations.forEach(::write)
        }
    }

    @SuppressLint("MissingPermission") // MainActivity checks ACCESS_FINE_LOCATION before starting the service.
    fun start() {
        thread.start()
        val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1_000L)
            .setMinUpdateIntervalMillis(100L) // take faster fixes if the provider has them
            .setMinUpdateDistanceMeters(0f)
            .setMaxUpdateDelayMillis(0L) // no batching
            .setWaitForAccurateLocation(false)
            .build()
        client.requestLocationUpdates(request, callback, thread.looper)
    }

    fun stop() {
        client.removeLocationUpdates(callback)
        thread.quitSafely()
    }

    private fun write(loc: Location) {
        val ns = loc.elapsedRealtimeNanos
        // Skip repeats and the cached fix from before the session started.
        if (ns <= lastFixNs) return
        val t = clock.t(ns)
        if (t < 0) return
        lastFixNs = ns
        log.row(
            listOf(
                fmt(t, 6),
                fmt(loc.latitude, 7),
                fmt(loc.longitude, 7),
                fmt(if (loc.hasAccuracy()) loc.accuracy else null, 1),
                fmt(if (loc.hasSpeed()) loc.speed else null, 2),
                fmt(if (loc.hasBearing()) loc.bearing else null, 1),
            ).joinToString(","),
        )
        onFix(loc)
    }
}
