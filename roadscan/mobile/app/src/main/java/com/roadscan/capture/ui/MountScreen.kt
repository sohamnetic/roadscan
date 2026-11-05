package com.roadscan.capture.ui

import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.roadscan.capture.CameraGeometry
import com.roadscan.capture.FieldOfView
import com.roadscan.capture.Prefs
import com.roadscan.capture.Vehicle
import java.util.Locale
import kotlin.math.roundToInt
import kotlin.math.tan

/**
 * Mount settings: camera pitch and height, with a live 16:9 preview of what the video will see.
 *
 * The yellow line is where the horizon should appear for the chosen pitch. Slide pitch until it
 * sits on the real horizon (on a flat road) and the pitch is measured, not guessed.
 */
@Composable
fun MountScreen(prefs: Prefs, vehicle: Vehicle, cameraFree: Boolean) {
    var pitch by remember { mutableFloatStateOf(prefs.pitchDeg) }
    var height by remember(vehicle) { mutableFloatStateOf(prefs.heightM(vehicle)) }
    var fov by remember { mutableStateOf<FieldOfView?>(null) }

    Row(Modifier.fillMaxSize().padding(12.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
        Box(
            Modifier.weight(0.6f).aspectRatio(16f / 9f).background(Color.Black),
            contentAlignment = Alignment.Center,
        ) {
            if (cameraFree) {
                CameraPreview(
                    onCameraInfo = { fov = CameraGeometry.fieldOfView(it, 1920, 1080) },
                    modifier = Modifier.fillMaxSize(),
                )
                fov?.let { MountGuides(pitchDeg = pitch, fov = it, modifier = Modifier.fillMaxSize()) }
            } else {
                Text("Preview is off while recording", color = Color.White)
            }
        }

        Column(
            Modifier.weight(0.4f).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Camera pitch: ${"%.1f".format(Locale.US, pitch)}° down", style = MaterialTheme.typography.titleMedium)
            Slider(
                value = pitch,
                onValueChange = { pitch = (it * 2).roundToInt() / 2f },
                onValueChangeFinished = { prefs.pitchDeg = pitch },
                valueRange = 0f..45f,
            )
            TextButton(onClick = {
                pitch = Prefs.DEFAULT_PITCH_DEG
                prefs.pitchDeg = pitch
            }) { Text("Reset to ${Prefs.DEFAULT_PITCH_DEG.roundToInt()}°") }

            Text(
                "Camera height (${vehicle.label}): ${"%.2f".format(Locale.US, height)} m",
                style = MaterialTheme.typography.titleMedium,
            )
            Slider(
                value = height,
                onValueChange = { height = (it * 20).roundToInt() / 20f },
                onValueChangeFinished = { prefs.setHeightM(vehicle, height) },
                valueRange = 0.5f..2.5f,
            )
            TextButton(onClick = {
                height = vehicle.defaultHeightM
                prefs.setHeightM(vehicle, height)
            }) { Text("Reset to ${vehicle.defaultHeightM} m") }

            Text(
                "Mount the phone in landscape, as this screen is shown, on a rigid handlebar or windshield " +
                    "mount. Tilt it down until the road fills the frame below the dashed line. Measure height " +
                    "from the road to the lens.",
                style = MaterialTheme.typography.bodySmall,
            )
            fov?.let {
                Text(
                    String.format(Locale.US, "Field of view (1080p): %.1f° × %.1f°", it.hfovDeg, it.vfovDeg),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
    }
}

/** Expected horizon for the pitch (solid yellow) and the top of the road area, 40% down (dashed). */
@Composable
private fun MountGuides(pitchDeg: Float, fov: FieldOfView, modifier: Modifier) {
    Canvas(modifier) {
        val roadTop = size.height * 0.4f
        drawLine(
            Color.White, Offset(0f, roadTop), Offset(size.width, roadTop), strokeWidth = 2f,
            pathEffect = PathEffect.dashPathEffect(floatArrayOf(20f, 14f)),
        )
        // Pinhole camera pitched down by p: the horizon sits tan(p) * fy pixels above the centre row.
        val halfV = Math.toRadians(fov.vfovDeg / 2)
        val fraction = 0.5 - tan(Math.toRadians(pitchDeg.toDouble())) / (2 * tan(halfV))
        if (fraction in 0.0..1.0) {
            val y = (size.height * fraction).toFloat()
            drawLine(Color.Yellow, Offset(0f, y), Offset(size.width, y), strokeWidth = 4f)
        }
    }
}

@Composable
private fun CameraPreview(onCameraInfo: (CameraInfo) -> Unit, modifier: Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnCameraInfo by rememberUpdatedState(onCameraInfo)
    val previewView = remember {
        PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER }
    }

    DisposableEffect(lifecycleOwner) {
        // Same 16:9 crop of the sensor as the 1080p video, so the guides line up with the recording.
        val preview = Preview.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .build(),
            )
            .build()
        preview.setSurfaceProvider(previewView.surfaceProvider)
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        var disposed = false
        future.addListener({
            if (disposed) return@addListener
            val p = future.get().also { provider = it }
            runCatching { p.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview) }
                .onSuccess { currentOnCameraInfo(it.cameraInfo) }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            disposed = true
            provider?.unbind(preview)
        }
    }

    AndroidView(factory = { previewView }, modifier = modifier)
}
