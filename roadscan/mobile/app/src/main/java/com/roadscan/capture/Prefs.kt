package com.roadscan.capture

import android.content.Context
import androidx.core.content.edit

enum class Vehicle(val id: String, val label: String, val defaultHeightM: Float) {
    BIKE("bike", "Bike", 1.1f),
    AUTO("auto", "Auto", 1.3f),
}

/** Vehicle choice and camera mount geometry, kept between rides. Height is stored per vehicle. */
class Prefs(context: Context) {
    private val sp = context.getSharedPreferences("roadscan", Context.MODE_PRIVATE)

    var vehicle: Vehicle
        get() = Vehicle.entries.firstOrNull { it.id == sp.getString(KEY_VEHICLE, null) } ?: Vehicle.BIKE
        set(value) = sp.edit { putString(KEY_VEHICLE, value.id) }

    /** Downward tilt of the camera from horizontal, in degrees. */
    var pitchDeg: Float
        get() = sp.getFloat(KEY_PITCH, DEFAULT_PITCH_DEG)
        set(value) = sp.edit { putFloat(KEY_PITCH, value) }

    /** Height of the camera above the road, in metres. */
    fun heightM(vehicle: Vehicle): Float = sp.getFloat(heightKey(vehicle), vehicle.defaultHeightM)

    fun setHeightM(vehicle: Vehicle, value: Float) = sp.edit { putFloat(heightKey(vehicle), value) }

    private fun heightKey(vehicle: Vehicle) = "height_m_${vehicle.id}"

    companion object {
        const val DEFAULT_PITCH_DEG = 15f
        private const val KEY_VEHICLE = "vehicle"
        private const val KEY_PITCH = "pitch_deg"
    }
}
