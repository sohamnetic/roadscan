package com.roadscan.capture

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.Build
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.roadscan.capture.ui.MountScreen
import com.roadscan.capture.ui.RecordScreen
import com.roadscan.capture.ui.SessionsScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val prefs = Prefs(this)
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                Surface(Modifier.fillMaxSize()) { App(prefs) }
            }
        }
    }
}

private enum class Tab(val label: String, val icon: ImageVector) {
    RECORD("Record", Icons.Filled.PlayArrow),
    MOUNT("Mount", Icons.Filled.Build),
    SESSIONS("Sessions", Icons.AutoMirrored.Filled.List),
}

@Composable
private fun App(prefs: Prefs) {
    val context = LocalContext.current
    val status by RecordingState.status.collectAsStateWithLifecycle()
    var tab by rememberSaveable { mutableStateOf(Tab.RECORD) }
    var vehicle by remember { mutableStateOf(prefs.vehicle) }
    var message by remember { mutableStateOf<String?>(null) }

    fun startIfReady() {
        val missing = missingPermissions(context)
        message = when {
            missing.isNotEmpty() -> "Needs ${missing.joinToString { it.substringAfterLast('.') }}. " +
                "Allow camera and precise location in the app's settings."
            !locationEnabled(context) -> "Location is switched off. Turn it on, then press Start."
            else -> null
        }
        if (message == null) RecordingService.start(context)
    }

    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        startIfReady()
    }

    Row(Modifier.fillMaxSize().safeDrawingPadding()) {
        NavigationRail {
            Tab.entries.forEach { t ->
                NavigationRailItem(
                    selected = tab == t,
                    onClick = { tab = t },
                    icon = { Icon(t.icon, contentDescription = null) },
                    label = { Text(t.label) },
                )
            }
        }
        when (tab) {
            Tab.RECORD -> RecordScreen(
                status = status,
                vehicle = vehicle,
                pitchDeg = prefs.pitchDeg,
                heightM = prefs.heightM(vehicle),
                message = message,
                onVehicle = {
                    vehicle = it
                    prefs.vehicle = it
                },
                onStart = {
                    message = null
                    if (missingPermissions(context).isEmpty()) {
                        startIfReady()
                    } else {
                        permissionLauncher.launch(requestedPermissions())
                    }
                },
                onStop = { RecordingService.stop(context) },
            )
            Tab.MOUNT -> MountScreen(prefs = prefs, vehicle = vehicle, cameraFree = status.phase == Phase.IDLE)
            Tab.SESSIONS -> SessionsScreen(status = status)
        }
    }
}

/** Everything the app asks for. Notifications are optional: without them the ride still records. */
private fun requestedPermissions(): Array<String> = buildList {
    add(Manifest.permission.CAMERA)
    add(Manifest.permission.ACCESS_FINE_LOCATION)
    add(Manifest.permission.ACCESS_COARSE_LOCATION)
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) add(Manifest.permission.POST_NOTIFICATIONS)
}.toTypedArray()

/** Permissions a ride cannot do without. Approximate location alone is not enough. */
private fun missingPermissions(context: Context): List<String> =
    listOf(Manifest.permission.CAMERA, Manifest.permission.ACCESS_FINE_LOCATION).filter {
        ContextCompat.checkSelfPermission(context, it) != PackageManager.PERMISSION_GRANTED
    }

private fun locationEnabled(context: Context): Boolean =
    LocationManagerCompat.isLocationEnabled(context.getSystemService(LocationManager::class.java)!!)
