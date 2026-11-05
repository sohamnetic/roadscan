package com.roadscan.capture

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

enum class Phase { IDLE, STARTING, RECORDING, STOPPING }

data class RecordingStatus(
    val phase: Phase = Phase.IDLE,
    val sessionId: String? = null,
    val vehicle: Vehicle? = null,
    val elapsedS: Long = 0,
    val gpsFixes: Long = 0,
    val lastAccuracyM: Float? = null,
    val imuRows: Long = 0,
    val videoBytes: Long = 0,
    val videoT0: Double? = null,
    /** Problems from the current or most recent session; kept after it ends so the rider sees them. */
    val warnings: List<String> = emptyList(),
    val lastSessionId: String? = null,
)

/** Process-wide recording status, written by [RecordingService] and observed by the UI. */
object RecordingState {
    private val _status = MutableStateFlow(RecordingStatus())
    val status: StateFlow<RecordingStatus> = _status.asStateFlow()

    fun update(transform: (RecordingStatus) -> RecordingStatus) = _status.update(transform)
}
