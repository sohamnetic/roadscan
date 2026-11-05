# RoadScan Capture (Android)

Phase 1 capture app. A phone on a bike or auto records a ride as a **session folder**: 1080p video, GPS, accelerometer and gyroscope, all on one clock. The offline pipeline in `../pipeline` reads that folder (`roadscan/session.py` is the contract).

Kotlin, CameraX, Jetpack Compose. minSdk 26 (Android 8.0), targetSdk 35.

## Screens

- **Record**: pick the vehicle (bike or auto), then Start/Stop. While recording it shows elapsed time, GPS fix count and accuracy, IMU row count and rate, video size, and the measured `video_t0`. Recording runs in a foreground service with a partial wake lock, so it keeps going with the screen off. The notification has a Stop button.
- **Mount**: a live 16:9 preview with sliders for camera pitch (default 15°) and height (default 1.1 m for a bike, 1.3 m for an auto, stored separately for each). The yellow line shows where the horizon should appear at the chosen pitch. On a flat road, move the slider until the line sits on the real horizon, and you have measured the pitch. The dashed line at 40% height marks where the road should start (the road fills the lower 60% of the frame).
- **Sessions**: saved rides, newest first. **Share** zips the folder as `<session_id>.zip` (containing `<session_id>/…`) and opens the share sheet. **Delete** asks first.

The app is locked to landscape. Mount the phone so the app's screen is upright. The video is always recorded in that orientation.

## Session folder

Stored in app-specific external storage:
`/sdcard/Android/data/com.roadscan.capture/files/sessions/<session_id>/`. The `session_id` looks like `20261104-091502_bike` (UTC start time plus vehicle).

| File | Contents | How it is produced |
|---|---|---|
| `meta.json` | `session_id`, `vehicle_type`, `device`, `start_time_utc`, `camera{width,height,hfov_deg,vfov_deg,height_m,pitch_deg}`, `video_t0` | `width`/`height` are read back from the finished MP4 after rotation. The FOV is computed from Camera2 focal length, sensor size and the 16:9 crop (accurate to a degree or two). `height_m` and `pitch_deg` come from the Mount screen. The file is written about 3 s into the ride and rewritten at Stop. |
| `gps.csv` | `t,lat,lon,accuracy_m,speed_mps,bearing_deg` | FusedLocationProvider, `PRIORITY_HIGH_ACCURACY`, 1 s interval (faster fixes accepted, no batching). Speed and bearing are left empty when the fix has none. |
| `imu.csv` | `t,ax,ay,az,gx,gy,gz` | One row per accelerometer event at a 10 ms sampling period (~100 Hz). Acceleration is raw m/s² with gravity included. Gyroscope is rad/s, the latest reading at most ~10 ms old. Gyro columns are empty on phones without a gyroscope. |
| `video.mp4` | 1920×1080, 30 fps target, H.264, no audio | CameraX `VideoCapture`/`Recorder`, `Quality.FHD`, stabilisation off (it would distort the pinhole geometry), zoom 1×. |

**Time base.** Every `t` is seconds since session start on `SystemClock.elapsedRealtimeNanos`. Sensor event timestamps and `Location.getElapsedRealtimeNanos()` both use that clock, so neither is restamped on arrival. (If a phone's sensor clock is more than 1 s off from it, a fixed offset is applied.) `start_time_utc` is the wall-clock time at `t = 0`.

**`video_t0`.** CameraX does not report when the first frame was captured. The app estimates it as the minimum over all recording events of `event_time − recordedDuration`. That estimate runs late by the capture-to-encoder latency, typically a few tens of milliseconds and well under 0.1 s. At 30 km/h, 50 ms is about 0.4 m along the road.

## Build and install

Not built here (see [Verification status](#verification-status)). To build:

1. Install a recent Android Studio. Open **`roadscan/mobile`** (this folder, not the repo root).
2. Let Gradle sync. It downloads Gradle 8.14.3, AGP 8.10.1, Kotlin 2.1.21 and SDK Platform 35. If Studio offers to upgrade AGP, skip it until the app has been tested once.
   - Gradle JDK: Studio's bundled JDK (Settings → Build Tools → Gradle → Gradle JDK) is safest. JDK 17 to 24 all work.
3. On the phone, enable Developer options → USB debugging, plug it in, and press **Run**. The debug build is enough for pilot riders.

Command line (needs the Android SDK; set `ANDROID_HOME` or add `sdk.dir` to `local.properties`):

```bash
./gradlew assembleDebug
```

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On first Start the app asks for camera, precise location and notifications. Camera and **precise** location are required. Approximate location is refused because the pipeline drops fixes worse than 30 m. Location services must also be switched on.

## Confirm a recorded session matches the format

1. Mount the phone, set pitch and height on the Mount screen, and ride for 2 minutes or more under open sky. Then press Stop.
2. Copy the session to a computer, either way:
   ```bash
   adb pull /sdcard/Android/data/com.roadscan.capture/files/sessions/ ./sessions/
   ```
   or use **Sessions → Share** and send the zip to yourself, then unzip it.
3. Run the checker (standard-library Python; uses `ffprobe` if it is on PATH; `--pipeline` also runs the real `load_session`, which needs pandas):
   ```bash
   python tools/check_session.py sessions/20261104-091502_bike --pipeline ../pipeline
   ```
   It FAILs on format errors: wrong keys, wrong CSV headers, non-numeric fields, `t` going backwards, IMU slower than ~80 Hz, GPS slower than 1 Hz, video unreadable or not ~30 fps, size not matching `meta.json`, or `load_session` raising. It WARNs on things worth a look: poor GPS accuracy, missing gyroscope, dropped video frames, implausible mount values, or video and sensor logs ending more than 3 s apart.
4. Optional sync check for `video_t0`. Early in a ride, with the bike stopped, rap the mount sharply once. In the video, find the frame where the image jumps (frame `n`). Its time is `video_t0 + n / 30`. Compare that with the spike in `imu.csv`. They should agree within about 0.1 s.

## Verification status

What was checked on the machine this was written on (Windows, JDK 24, **no Android SDK, no device**):

- **Build scripts**: Gradle 8.14.3 configured the project with AGP 8.10.1 and Kotlin 2.1.21, and `:app:dependencies` resolved the whole debug runtime classpath. `compileDebugKotlin` stopped at "SDK location not found", as expected without an SDK.
- **Kotlin sources**: all of them type-check with `kotlinc` 2.1.21 plus the Compose compiler plugin. The classpath was the same library versions plus Robolectric's Android 15 framework jar, standing in for `android.jar`, and a stub `R` class. That pass caught and fixed real API errors.
- **Checker**: `tools/check_session.py` passes a synthetic session written in the app's exact output format (including a 1080p/30 fps MP4 from ffmpeg), and that session loads through the pipeline's `load_session`. It fails a session with a decimal-comma row and an extra camera key.
- **Not verified**: resource and manifest processing (aapt2), the APK build, and anything at runtime. That covers camera binding, the foreground service with the screen off, the real sensor and GPS rates on a phone, the FOV numbers, and `video_t0` accuracy. These need a first install and a test ride, using the steps above.

## Known limitations

- `video_t0` is an estimate, late by tens of milliseconds (see above).
- CameraX can drop frames under load, and the MP4 may be variable frame rate. `detect.py` currently times frames as `video_t0 + frame / fps`. Over a long ride, dropped frames make those times drift. Using each frame's own timestamp (`CAP_PROP_POS_MSEC`) avoids that.
- The FOV comes from the lens and sensor size, not a calibration. A checkerboard calibration per phone model will be more accurate for ground projection.
- Storage: 1080p uses roughly 75–130 MB per minute. The ride stops cleanly if storage runs out (the video finalises and the warning is shown). Exporting a zip needs as much free space again, so `adb pull` is better for long rides.
- No audio is recorded, on purpose.
