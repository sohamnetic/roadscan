package com.roadscan.capture.ui

import android.text.format.Formatter
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.roadscan.capture.Phase
import com.roadscan.capture.RecordingStatus
import com.roadscan.capture.SessionInfo
import com.roadscan.capture.SessionStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Saved rides, newest first, each exportable as `<session_id>.zip` through the share sheet. */
@Composable
fun SessionsScreen(status: RecordingStatus) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var sessions by remember { mutableStateOf(emptyList<SessionInfo>()) }
    var reloads by remember { mutableIntStateOf(0) }
    var exporting by remember { mutableStateOf<String?>(null) }
    var toDelete by remember { mutableStateOf<SessionInfo?>(null) }
    var error by remember { mutableStateOf<String?>(null) }
    val recordingId = status.sessionId.takeIf { status.phase != Phase.IDLE }

    LaunchedEffect(status.phase, reloads) {
        sessions = withContext(Dispatchers.IO) { SessionStore.list(context) }
    }

    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("${sessions.size} sessions in ${SessionStore.root(context).absolutePath}", style = MaterialTheme.typography.bodySmall)
        error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        LazyColumn(Modifier.weight(1f)) {
            items(sessions, key = { it.id }) { session ->
                val inProgress = session.id == recordingId
                SessionRow(
                    session = session,
                    note = when {
                        inProgress -> "recording…"
                        !session.hasMeta -> "no meta.json"
                        !session.hasVideo -> "no video"
                        else -> null
                    },
                    exporting = exporting == session.id,
                    actionsEnabled = !inProgress && exporting == null,
                    onExport = {
                        scope.launch {
                            exporting = session.id
                            error = null
                            try {
                                val zip = withContext(Dispatchers.IO) { SessionStore.zip(context, session) }
                                context.startActivity(SessionStore.shareIntent(context, zip))
                            } catch (e: Exception) {
                                error = "Export failed: ${e.message}"
                            } finally {
                                exporting = null
                            }
                        }
                    },
                    onDelete = { toDelete = session },
                )
                HorizontalDivider()
            }
        }
    }

    toDelete?.let { session ->
        AlertDialog(
            onDismissRequest = { toDelete = null },
            title = { Text("Delete ${session.id}?") },
            text = { Text("The video and sensor logs are removed from this phone. This cannot be undone.") },
            confirmButton = {
                TextButton(onClick = {
                    toDelete = null
                    scope.launch {
                        withContext(Dispatchers.IO) { SessionStore.delete(session) }
                        reloads++
                    }
                }) { Text("Delete") }
            },
            dismissButton = { TextButton(onClick = { toDelete = null }) { Text("Cancel") } },
        )
    }
}

@Composable
private fun SessionRow(
    session: SessionInfo,
    note: String?,
    exporting: Boolean,
    actionsEnabled: Boolean,
    onExport: () -> Unit,
    onDelete: () -> Unit,
) {
    val context = LocalContext.current
    Row(Modifier.fillMaxWidth().padding(vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(session.id, fontWeight = FontWeight.Bold)
            Text(
                listOfNotNull(
                    session.vehicle,
                    Formatter.formatShortFileSize(context, session.sizeBytes),
                    session.startTimeUtc,
                    note,
                ).joinToString(" · "),
                style = MaterialTheme.typography.bodySmall,
            )
        }
        if (exporting) {
            CircularProgressIndicator(Modifier.padding(12.dp))
        } else {
            IconButton(onClick = onExport, enabled = actionsEnabled) { Icon(Icons.Filled.Share, contentDescription = "Export zip") }
        }
        IconButton(onClick = onDelete, enabled = actionsEnabled) { Icon(Icons.Filled.Delete, contentDescription = "Delete") }
    }
}
