package com.roadscan.capture.ui

import android.text.format.Formatter
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.roadscan.capture.Phase
import com.roadscan.capture.RecordingStatus
import com.roadscan.capture.Vehicle
import java.util.Locale

/** Vehicle picker, Start/Stop, and live counters so the rider can see each stream is arriving. */
@Composable
fun RecordScreen(
    status: RecordingStatus,
    vehicle: Vehicle,
    pitchDeg: Float,
    heightM: Float,
    message: String?,
    onVehicle: (Vehicle) -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
    val idle = status.phase == Phase.IDLE
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Vehicle", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 10.dp, end = 8.dp))
            Vehicle.entries.forEach { v ->
                if (v == vehicle) {
                    Button(onClick = {}, enabled = idle) { Text(v.label) }
                } else {
                    OutlinedButton(onClick = { onVehicle(v) }, enabled = idle) { Text(v.label) }
                }
            }
        }
        Text(
            String.format(Locale.US, "Mount: pitch %.1f° down, camera %.2f m above road (change in Mount)", pitchDeg, heightM),
            style = MaterialTheme.typography.bodyMedium,
        )

        if (idle) {
            Button(onClick = onStart, modifier = Modifier.width(260.dp).height(64.dp)) {
                Text("Start recording", style = MaterialTheme.typography.titleMedium)
            }
        } else {
            Button(
                onClick = onStop,
                enabled = status.phase != Phase.STOPPING,
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error),
                modifier = Modifier.width(260.dp).height(64.dp),
            ) {
                Text(if (status.phase == Phase.STOPPING) "Saving…" else "Stop", style = MaterialTheme.typography.titleMedium)
            }
        }

        message?.let { Text(it, color = MaterialTheme.colorScheme.error) }

        if (!idle) LiveStatus(status)

        if (idle && status.lastSessionId != null) {
            Text("Saved ${status.lastSessionId}", fontWeight = FontWeight.Bold)
        }
        status.warnings.forEach { Text("⚠ $it", color = MaterialTheme.colorScheme.error) }
    }
}

@Composable
private fun LiveStatus(status: RecordingStatus) {
    val context = LocalContext.current
    val elapsed = status.elapsedS
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            when (status.phase) {
                Phase.STARTING -> "Starting camera…"
                Phase.STOPPING -> "Finishing video…"
                else -> String.format(Locale.US, "Recording %d:%02d:%02d", elapsed / 3600, elapsed / 60 % 60, elapsed % 60)
            },
            style = MaterialTheme.typography.titleLarge,
        )
        Text("Session ${status.sessionId}")

        val accuracy = status.lastAccuracyM
        Text(
            when {
                status.gpsFixes == 0L -> "GPS: waiting for a fix (stay under open sky)"
                accuracy == null -> "GPS: ${status.gpsFixes} fixes"
                else -> String.format(Locale.US, "GPS: %d fixes, last ±%.0f m", status.gpsFixes, accuracy) +
                    if (accuracy > 30f) " (too coarse; the pipeline drops fixes worse than 30 m)" else ""
            },
        )

        val imuHz = if (elapsed > 0) status.imuRows.toDouble() / elapsed else 0.0
        Text(String.format(Locale.US, "IMU: %d rows (~%.0f Hz)", status.imuRows, imuHz))

        Text(
            "Video: ${Formatter.formatShortFileSize(context, status.videoBytes)}" +
                (status.videoT0?.let { String.format(Locale.US, ", first frame at t=%.3f s", it) } ?: ""),
        )
    }
}
