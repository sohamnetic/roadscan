package com.roadscan.capture

import android.hardware.camera2.CameraCharacteristics
import androidx.annotation.OptIn
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraInfo
import kotlin.math.atan
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tan

data class FieldOfView(val hfovDeg: Double, val vfovDeg: Double)

object CameraGeometry {
    /** Used when the camera does not report its focal length or sensor size: a typical phone main camera. */
    private const val FALLBACK_HFOV_DEG = 66.0

    /**
     * Field of view of a [width] x [height] video from this camera, at zoom 1x with stabilisation off.
     *
     * The video is a centred crop of the sensor's active array to the video's aspect ratio (a 16:9
     * video from a 4:3 sensor loses the top and bottom), so the FOV is computed from that crop and
     * the lens focal length. Accurate to a degree or two; a checkerboard calibration is better.
     */
    @OptIn(ExperimentalCamera2Interop::class)
    fun fieldOfView(cameraInfo: CameraInfo, width: Int, height: Int): FieldOfView {
        val c2 = Camera2CameraInfo.from(cameraInfo)
        val focalMm = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull()
        val physical = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
        val pixelArray = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE)
        val active = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
        if (focalMm == null || focalMm <= 0f || physical == null || pixelArray == null || active == null) {
            return fallback(width, height)
        }

        // Active array size in millimetres, in the sensor's own (landscape) orientation.
        val activeW = physical.width.toDouble() * active.width() / pixelArray.width
        val activeH = physical.height.toDouble() * active.height() / pixelArray.height
        var longMm = max(activeW, activeH)
        var shortMm = min(activeW, activeH)

        val videoAspect = max(width, height).toDouble() / min(width, height)
        if (longMm / shortMm > videoAspect) longMm = shortMm * videoAspect else shortMm = longMm / videoAspect

        val longFov = fov(longMm, focalMm.toDouble())
        val shortFov = fov(shortMm, focalMm.toDouble())
        return if (width >= height) FieldOfView(longFov, shortFov) else FieldOfView(shortFov, longFov)
    }

    fun fallback(width: Int, height: Int): FieldOfView {
        val halfTan = tan(Math.toRadians(FALLBACK_HFOV_DEG / 2))
        val vfov = Math.toDegrees(2 * atan(halfTan * height / width))
        return FieldOfView(FALLBACK_HFOV_DEG, vfov)
    }

    private fun fov(sizeMm: Double, focalMm: Double) = Math.toDegrees(2 * atan(sizeMm / (2 * focalMm)))
}
