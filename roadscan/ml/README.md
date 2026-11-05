# Road-damage detector training

The model is fine-tuned from COCO-pretrained YOLO11 on RDD2022 (India). Its class names match the pipeline: `longitudinal_crack`, `transverse_crack`, `alligator_crack`, `pothole`.

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

## Train

```bash
python train.py --data D:/pothole_mapping/data/rdd_india_yolo/data.yaml --model D:/pothole_mapping/models/yolo11m.pt
```

- To pause, stop the process; at most the current epoch is lost. To continue later, run the same command with `--resume` added.
- Training runs with `YOLO_OFFLINE=1`, so Ultralytics downloads nothing and sends no analytics.
- Results are saved under `D:/pothole_mapping/runs/<name>/`, and the best weights are copied to `D:/pothole_mapping/models/roadscan_<name>.pt`.
- Use the weights with the pipeline, passing the thresholds calibrated for this model:

```bash
python -m roadscan detect <session> --weights D:/pothole_mapping/models/roadscan_rdd_india_y11m.pt
```

```bash
python -m roadscan process <session> --config D:/pothole_mapping/models/roadscan_rdd_india_y11m.fusion.json
```

## Calibrate thresholds for a new model
The model's confidence scores run low: pothole F1 peaks near 0.2, so the pipeline's built-in thresholds (0.6 / 0.3, tuned for the synthetic detector) would drop most real potholes. For each new model, look at its precision and recall at each cutoff, then write `models/<model>.fusion.json`. Use `--augment` if `detect` runs with `--tta`, which is the default.

```bash
python calibrate.py --weights D:/pothole_mapping/models/roadscan_rdd_india_y11m.pt --data D:/pothole_mapping/data/rdd_india_yolo/data.yaml --augment
```
