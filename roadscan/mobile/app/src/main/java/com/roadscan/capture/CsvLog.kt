package com.roadscan.capture

import android.os.SystemClock
import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.atomic.AtomicLong

/** Every `t` column is seconds since [startNs], on the `SystemClock.elapsedRealtimeNanos` base. */
class SessionClock(val startNs: Long) {
    fun t(elapsedRealtimeNs: Long): Double = (elapsedRealtimeNs - startNs) / 1e9
}

/**
 * Append-only CSV file. Flushed about once a second so a crash or a killed process loses at
 * most a second of rows. Rows written after [close] are dropped (late sensor callbacks).
 */
class CsvLog(file: File, header: String) : Closeable {
    private val out: BufferedWriter = file.bufferedWriter(Charsets.UTF_8, 64 * 1024)
    private var lastFlushNs = SystemClock.elapsedRealtimeNanos()
    private var closed = false
    val rows = AtomicLong(0)

    init {
        out.write(header)
        out.newLine()
    }

    @Synchronized
    fun row(line: String) {
        if (closed) return
        out.write(line)
        out.write("\n")
        rows.incrementAndGet()
        val now = SystemClock.elapsedRealtimeNanos()
        if (now - lastFlushNs > 1_000_000_000L) {
            out.flush()
            lastFlushNs = now
        }
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        out.close()
    }
}

/** Formats numbers with a '.' decimal separator whatever the phone's locale; null becomes an empty field. */
internal fun fmt(value: Double?, decimals: Int): String =
    if (value == null || value.isNaN()) "" else String.format(Locale.US, "%.${decimals}f", value)

internal fun fmt(value: Float?, decimals: Int): String = fmt(value?.toDouble(), decimals)
