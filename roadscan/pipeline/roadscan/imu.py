"""Jolt detection from the phone's accelerometer.

The phone can be mounted at any angle, so the vertical axis is estimated from the
slow-moving gravity vector rather than assumed to be a device axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt

from .geo import Track

# Minimum vertical jolt (m/s^2) worth reporting; raised automatically on rough roads.
BASE_THRESHOLD = {"bike": 4.0, "auto": 3.0}
NOISE_MULTIPLE = 6.0
REF_SPEED_MPS = 8.0  # jolts are normalised to ~30 km/h
MIN_SPEED_MPS = 1.5  # below this, jolts are from walking the vehicle, kerbs, etc.
MIN_PEAK_GAP_S = 0.4  # front and rear wheel hits merge into one jolt

JOLT_COLUMNS = ["t", "peak", "speed_mps", "threshold", "strength"]


def vertical_accel(imu: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Band-passed vertical acceleration on a uniform time grid: (t, accel, sample_rate)."""
    t = imu["t"].to_numpy(float)
    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt
    grid = np.arange(t[0], t[-1], dt)
    acc = np.column_stack([np.interp(grid, t, imu[c].to_numpy(float)) for c in ("ax", "ay", "az")])

    gravity = sosfiltfilt(butter(2, 0.3, "lowpass", fs=fs, output="sos"), acc, axis=0)
    g_norm = np.linalg.norm(gravity, axis=1)
    vertical = np.sum(acc * gravity, axis=1) / g_norm - g_norm

    high = min(20.0, 0.45 * fs)
    vertical = sosfiltfilt(butter(2, [1.5, high], "bandpass", fs=fs, output="sos"), vertical)
    return grid, vertical, fs


def detect_jolts(imu: pd.DataFrame, track: Track, vehicle_type: str) -> pd.DataFrame:
    """Sharp vertical jolts while moving.

    `strength` is the speed-normalised peak divided by the threshold, so 1.0 means
    "just detectable" and is comparable between bikes, autos and speeds.
    """
    if len(imu) < 200:
        return pd.DataFrame(columns=JOLT_COLUMNS)
    t, vertical, fs = vertical_accel(imu)
    magnitude = np.abs(vertical)
    noise_sigma = 1.4826 * float(np.median(magnitude))
    threshold = max(BASE_THRESHOLD[vehicle_type], NOISE_MULTIPLE * noise_sigma)

    peaks, props = find_peaks(magnitude, height=threshold, distance=max(1, int(MIN_PEAK_GAP_S * fs)))
    times = t[peaks]
    p = track.at(times)
    keep = p.valid & (p.speed >= MIN_SPEED_MPS)
    peak = props["peak_heights"][keep]
    speed = p.speed[keep]
    normalised = peak * np.sqrt(REF_SPEED_MPS / np.maximum(speed, 3.0))
    return pd.DataFrame(
        {
            "t": times[keep],
            "peak": peak,
            "speed_mps": speed,
            "threshold": threshold,
            "strength": normalised / threshold,
        }
    )
