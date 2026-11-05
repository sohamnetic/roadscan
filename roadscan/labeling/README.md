# Labelling our own ride frames

The model was trained on RDD2022 India photos. Our phones see roads from a different height and angle, and with things RDD never labelled: speed breakers, manholes, patches and puddles. A few hundred labelled frames from our own rides are the biggest accuracy gain available.

Classes ([`classes.py`](classes.py)):
- **Damage:** `pothole`, `alligator_crack`, `crack` (longitudinal and transverse merged)
- **Look-alikes:** `speed_breaker`, `manhole`, `patch`, `puddle`. These are labelled so the model learns they are **not** potholes.

## One-time setup
Label Studio lives in its own virtualenv, so it can't break the pipeline's packages:

```bash
python -m venv .venv-label
```

```bash
.venv-label/Scripts/python -m pip install label-studio
```

## Each batch of rides
**1. Blur, so faces and plates never reach the labelling tool:**

```bash
python -m roadscan blur D:/rides/ride_01 --weights D:/pothole_mapping/models/yolo11n.pt
```

**2. Pick frames worth labelling.** This takes one frame every 2 s, plus every frame where the model is unsure (confidence 0.10–0.50). The model's boxes are pre-filled, so labelling is mostly fixing, not drawing.

```bash
python make_tasks.py D:/rides/ride_01 D:/rides/ride_02 --out D:/pothole_mapping/data/labeling --weights D:/pothole_mapping/models/roadscan_rdd_india_y11m.pt
```

**3. Start Label Studio:** double-click `start_label_studio.bat`, then open http://localhost:8080 and create an account. The account is local to this laptop.

**4. First time only, create the project:**
1. Create a project named "RoadScan".
2. In **Settings → Labeling Interface → Code**, paste [`label_config.xml`](label_config.xml).
3. In **Settings → Cloud Storage → Add Source Storage → Local files**, set the absolute path to `D:\pothole_mapping\data\labeling\images`. Leave "Treat every bucket object as a source file" **off**, then save. Don't sync.

**5. Import tasks:** use **Import** and choose `data/labeling/tasks.json`.

**6. Label.** For each frame:
- Fix the pre-filled boxes, and add any the model missed: potholes and look-alikes.
- If the frame is unusable, mark it `blurry_or_dark`.
- Hotkeys `1`–`7` pick the class.

**7. Export** with **Export → JSON**, then convert:

```bash
python to_yolo.py export.json --images D:/pothole_mapping/data/labeling/images --out D:/pothole_mapping/data/own_yolo
```

The train/val split is **by ride**, so val frames come from rides the model never trained on. Label at least 2 rides so there is a val set.

## Tips
- Box potholes tightly, including the broken edge. Box a whole speed breaker across the lane.
- Label everything in a frame, or skip the frame. Half-labelled frames teach the model that the unlabelled potholes are background.
- A good first target is about 500 frames (around 3–4 hours of labelling).
