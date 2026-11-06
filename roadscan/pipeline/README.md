# roadscan pipeline

The offline pipeline turns recorded rides (a phone on a bike or an auto) into a tracked map of road damage.

```
session folder ──► jolts (IMU) ──┐
       │                         ├─► fusion ─► candidates ─► registry (SQLite) ─► map.html / CSV / GeoJSON
       └─► detections (YOLO) ────┘
```

| Module | What it does |
|---|---|
| `session.py` | The session folder format shared with the capture app (`meta.json`, `gps.csv`, `imu.csv`, `detections.csv`) |
| `detect.py` | Runs a YOLO model over `video.mp4` and writes `detections.csv` (RDD2022 classes D00/D10/D20/D40 are mapped) |
| `imu.py` | Estimates gravity so the phone can be mounted at any angle, band-passes the vertical axis and finds speed-normalised jolts |
| `geo.py` | Projects pixels onto the road plane using the mount height and pitch, then onto lat/lon using GPS position and bearing |
| `fusion.py` | Clusters per-frame boxes into objects and checks objects in the wheel path against jolts at the predicted hit time. Confirms each object and scores severity 1–5 |
| `registry.py` | Matches potholes across rides and runs the lifecycle `open → confirmed → likely_fixed / dismissed → verified_fixed` |
| `render.py` | Writes a self-contained Leaflet map with a repair-priority list and ward filter, plus CSV and GeoJSON exports |
| `places.py` | Tags each defect with its nearest drivable road and its ward (OpenStreetMap, or the city's own ward file). Main roads raise repair priority (primary ×1.4, secondary ×1.3, tertiary ×1.15). Also writes the per-ward summary |
| `mapmatch.py` | Map-matches a ride's GPS track to OSM roads (an HMM over connected roads). Fixes that drifted off the road are pulled back to its edge; fixes already on the road keep their lane position |
| `service.py`, `api.py`, `static/dashboard.html` | Web backend and dashboard: map, filters, ward table, defect history, "mark repaired", ride upload, exports and printable work orders |
| `blur.py` | Pixelates every person and vehicle in a ride video, covering faces and number plates, before footage is stored or shared |
| `synth.py`, `evaluate.py` | Synthetic rides with ground truth, and the scoring used by the tests |

## Fusion rules (in `FusionConfig`)
- **Pothole:** confirmed if confidence ≥ 0.6 on 2+ frames, **or** confidence ≥ 0.3 plus a jolt within ±0.6 s of when the wheel should reach it.
  - These defaults suit the synthetic detector.
  - A real model needs its own thresholds: pass `--config models/<model>.fusion.json` to `process`. See `ml/calibrate.py`.
- **"Puddle" plus a jolt:** treated as a waterlogged pothole.
- **Speed breaker or manhole boxes:** absorb their jolts, so those jolts don't become false potholes.
- **Strong jolts nothing on camera explains:** reported as unconfirmed `imu` candidates.

## Run it

```bash
pip install -e ".[dev]"
```

```bash
python -m roadscan synth ../demo/rides --rides 8
```

```bash
python -m roadscan eval ../demo/rides --db ../demo/roadscan.db
```

```bash
python -m roadscan map --db ../demo/roadscan.db --out ../demo/map.html --exports ../demo/exports
```

For real rides, run these instead of `synth`/`eval` (it needs `pip install -e ".[yolo]"` and trained weights):

```bash
python -m roadscan detect path/to/session --weights ../../models/roadscan_rdd_india_v2_y11m_736.pt
```

```bash
python -m roadscan blur path/to/session --weights ../../models/yolo11n.pt
```

```bash
python -m roadscan process path/to/session1 path/to/session2 --db roadscan.db --config ../../models/roadscan_rdd_india_v2_y11m_736.fusion.json
```

- `detect` runs on the original video, and `blur` writes `video_blurred.mp4`. Delete the original `video.mp4` before footage leaves the laptop.
- `blur` hides whole people and vehicles rather than hunting for faces and plates one by one, because the product only needs the road surface. It also drops the audio.

### Roads and wards

```bash
python -m roadscan osm --db roadscan.db --out osm
```

```bash
python -m roadscan map --db roadscan.db --out map.html --exports exports --places osm
```

- `osm` downloads drivable roads and ward boundaries (OSM `admin_level=10` in Bengaluru) for the area around the mapped defects. It uses the Overpass API and falls back to mirrors when the main server is busy.
- `--places` adds road, road type and ward to the map, the ward filter and the exports.
- Passing `--places osm` to `process` also map-matches each ride's GPS before fusion, and writes `gps_matched.csv` into the session.
- `exports/wards.csv` gives each ward's open defects, severe ones, and ones open more than 30 days. This is the view a ward engineer works from.
- To use the city's official ward map instead of OSM's, add `--wards city_wards.geojson`.

After a site check, close a defect with `python -m roadscan fix <id> --db roadscan.db`.

## Recording with an iPhone (Safari capture page)
Native iPhone apps can only be built on a Mac. For testing, the capture app is a web page instead. Open it in Safari: it records video, GPS and motion sensors in the same session format as the Android app, then uploads the ride straight to the laptop.

**On the laptop:**

```bash
python -m roadscan serve --lan --db roadscan.db --weights ../../models/roadscan_rdd_india_v2_y11m_736.pt --config ../../models/roadscan_rdd_india_v2_y11m_736.fusion.json
```

- `--lan` serves HTTPS on the Wi-Fi, because the iPhone camera needs HTTPS. It uses a self-signed certificate made for this laptop, plus a fresh access token.
- It prints a link like `https://192.168.1.8:8443/capture?token=...`.
- The first time, Windows asks whether Python may accept network connections. Allow **Private networks**.

**On the iPhone (same Wi-Fi):**
1. Open the printed link in **Safari**. Safari warns that the connection isn't private, because the certificate is the laptop's own. Tap **Show Details → visit this website**.
2. Tap **Start camera & sensors**, then allow Camera, Location and Motion.
3. Mount the phone **sideways (landscape)**, with the rear camera on the road ahead. Enter the phone's height above the road, then tap **Measure tilt** while standing still.
4. Tap **Record** and ride. **Keep the screen on and the page open**: iPhone pauses the camera and sensors in the background.
5. Tap **Stop**, then **Upload to laptop**. The laptop runs detection and processing, and the ride shows up on the dashboard. **Save to Files** keeps a copy on the phone.

Details:
- Video is 720p by default, about 30 MB per minute.
- Safari gives motion data at about 60 Hz, which is plenty for jolts (the pipeline needs ≥ 40 Hz).
- Anyone without the token gets a 401, but this laptop itself never needs it. Only run `--lan` on a network you trust.

## Dashboard and API

```bash
pip install -e ".[web,places]"
```

```bash
python -m roadscan serve --db roadscan.db --places osm --config ../../models/roadscan_rdd_india_v2_y11m_736.fusion.json
```

Open http://127.0.0.1:8000/.

| Feature | Details |
|---|---|
| Map | Defects colored by status; open counts, plus how many have been open more than 30, 60 or 90 days |
| Filters | Status, ward, severity |
| Priority list | Click a defect for its history: every ride that saw it, with confidence and jolt |
| Mark repaired | For use after a site check |
| Wards tab | The per-ward table, with a work order for each ward |
| Upload | Takes a `.zip` of one session folder. Processing runs in the background. If the zip has video but no `detections.csv`, detection runs first, provided the server was started with `--weights` |
| Exports | CSV, GeoJSON, ward CSV, and a printable work order. Use the browser's Print → Save as PDF |

API endpoints:
- `GET /api/potholes?status=&ward=&min_severity=`
- `GET /api/potholes/{id}`
- `POST /api/potholes/{id}/fix`
- `GET /api/wards`
- `GET /api/summary`
- `POST /api/sessions`
- `GET /api/jobs`
- `GET /api/export/...`
- `GET /workorder?ward=`

**Limits of this MVP:**
- There are no logins, so the server binds to 127.0.0.1. Add authentication before anyone else can reach it.
- Storage is the SQLite registry. Move to PostGIS when several cities or a lot of concurrent uploads arrive.
- The dashboard is plain JavaScript with MapLibre, no build step. Move it to React when it outgrows one page.

## Synthetic results (8 rides, seed 7)

| | precision | recall | F1 |
|---|---|---|---|
| vision only, conf ≥ 0.30 | 0.58 | 0.87 | 0.69 |
| vision only, conf ≥ 0.45 | 0.66 | 0.84 | 0.74 |
| vision only, conf ≥ 0.60 | 0.85 | 0.69 | 0.76 |
| **vision + IMU** | 0.85 | 0.71 | **0.77** |

Registry after 8 rides:
- all 30 potholes found
- 4 duplicate entities
- median location error 1.4 m (single rides: about 3 m)
- 7 of 8 repairs noticed, with no open pothole closed by mistake

At night, recall drops to about 0.3. The camera is unsure, so a pothole is confirmed only if a wheel hits it.

**These numbers only show that the code works.** The simulated detector and road were designed by us, so they say nothing about real roads. The real test is the walked ground-truth route in the plan.

## Not built yet
- Evidence photo crops, to be cut from the blurred video
- Logins and roles in the dashboard; PostGIS
- One-way streets in map-matching (all roads are treated as two-way)
- On-device inference
- `detect.py`, `blur.py` and map-matching have not been run on a real ride yet
- The OSM ward boundaries have not been checked against the city's official ward list
