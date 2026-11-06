# Road-damage detector training

The model is fine-tuned from COCO-pretrained YOLO11 on RDD2022 (India). It uses the shared class list in `labeling/classes.py`, so our own labelled frames can be added later. RDD only labels three of those classes: `pothole`, `alligator_crack` and `crack` (longitudinal and transverse merged).

## Data
RDD2022 is CC BY 4.0, from Figshare record [21431547](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547).
- The old per-country S3 links now return 403.
- The Figshare copy is one 13.3 GB zip of nested country zips, and Figshare throttles each connection to about 55 KB/s.
- `fetch_zip_subset.py` gets around this. It reads the remote zip's index, then pulls only the bytes for `India.zip` (527 MB) over many parallel range requests.

```bash
python -I fetch_zip_subset.py https://ndownloader.figshare.com/files/38030910 D:/pothole_mapping/data/rdd2022_raw/RDD2022_partial.zip --match India.zip
```

```bash
python -I prepare_rdd.py D:/pothole_mapping/data/rdd2022_raw/RDD2022_partial.zip D:/pothole_mapping/data/rdd_india_yolo --country India
```

In Git Bash, set `MSYS_NO_PATHCONV=1` so arguments like `/India/` aren't rewritten into Windows paths.

The official test split has no labels, so `prepare_rdd.py` splits the labelled train set 85/15 into train and val (seed 0). Images with no damage stay in as background examples.

Map the RDD classes onto the shared list (same split, images hard-linked):

```bash
python remap_classes.py D:/pothole_mapping/data/rdd_india_yolo D:/pothole_mapping/data/rdd_india_v2_yolo
```

## Train
The current model (v2) started from v1's weights and trained 50 more epochs at 736 px:

```bash
python train.py --data D:/pothole_mapping/data/rdd_india_v2_yolo/data.yaml --model D:/pothole_mapping/runs/rdd_india_y11m/weights/best.pt --imgsz 736 --epochs 50 --name rdd_india_v2_y11m_736
```

- To pause, stop the process; at most the current epoch is lost. To continue later, run the same command with `--resume` added.
- Training runs with `YOLO_OFFLINE=1`, so Ultralytics downloads nothing and sends no analytics.
- Results are saved under `D:/pothole_mapping/runs/<name>/`, and the best weights are copied to `D:/pothole_mapping/models/roadscan_<name>.pt`.
- Use the weights with the pipeline, passing the thresholds calibrated for this model:

```bash
python -m roadscan detect <session> --weights D:/pothole_mapping/models/roadscan_rdd_india_v2_y11m_736.pt
```

```bash
python -m roadscan process <session> --config D:/pothole_mapping/models/roadscan_rdd_india_v2_y11m_736.fusion.json
```

`detect` runs at the image size saved in the checkpoint (736 here).

## Results (RDD2022 India val, 1,155 images, AP50)

| Model | Inference | Pothole | Alligator | Crack | ms/image (GPU) |
|---|---|---|---|---|---|
| v1 `roadscan_rdd_india_y11m` (640, 4 classes) | plain | 0.396 | 0.545 | 0.222 long. / 0.123 trans. | – |
| v1 | TTA | 0.414 | 0.559 | – | – |
| **v2 `roadscan_rdd_india_v2_y11m_736`** (736, cracks merged) | **plain** | **0.416** | **0.567** | **0.249** | 14 |
| v2 | TTA | 0.422 | 0.574 | 0.251 | 40 |

v2 without TTA matches v1 with TTA at a third of the cost, so v2 runs without TTA. `crack` isn't directly comparable to v1's two crack classes.

## Calibrate thresholds for a new model
The model's confidence scores run low: pothole F1 peaks near 0.2, so the pipeline's built-in thresholds (0.6 / 0.3, tuned for the synthetic detector) would drop most real potholes. For each new model, look at its precision and recall at each cutoff, then write `models/<model>.fusion.json`. Pass the model's `--imgsz`, and add `--augment` only if `detect` will run with `--tta`.

```bash
python calibrate.py --weights D:/pothole_mapping/models/roadscan_rdd_india_v2_y11m_736.pt --data D:/pothole_mapping/data/rdd_india_v2_yolo/data.yaml --imgsz 736
```

## On-phone model (TFLite)
TensorFlow lives in its own virtualenv. Pin `ultralytics==8.4.80` there: newer versions export TFLite as "LiteRT", which is Linux/macOS only.

```bash
../../.venv-export/Scripts/python export_mobile.py --weights D:/pothole_mapping/models/roadscan_rdd_india_y11m.pt --data D:/pothole_mapping/data/rdd_india_yolo/data.yaml
```

v1 YOLO11m, India val set (v2 not exported yet). Speed is laptop CPU with 4 threads, only a rough guide to phone speed:

| File | Size | CPU ms/frame | mAP50 | Pothole AP50 |
|---|---|---|---|---|
| `.pt` (PyTorch) | 40.5 MB | n/a | 0.322 | 0.396 |
| `_fp16.tflite` | 40.3 MB | 299 | 0.317 | 0.394 |
| `_int8.tflite` | 20.7 MB | 188 | 0.312 | 0.388 |

- int8 costs under 1 point of accuracy at half the size.
- YOLO11m is still too heavy to detect live on a phone at 5–10 fps. For on-phone detection, train YOLO11n or YOLO11s, which are about 3–10× faster.
- For now, detection runs on the laptop after upload.
