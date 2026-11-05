# RoadScan

Phones on bikes and autos find potholes and other road damage, tag each one with GPS, and track it on a city map over time: open → confirmed → fixed. The aim is to give road owners in Indian cities a live, prioritised, ward-wise list of defects, with proof of repair.

| Folder | What's in it |
|---|---|
| [`roadscan/mobile`](roadscan/mobile) | Android capture app (Kotlin, CameraX). Records synced video, GPS and accelerometer data |
| [`roadscan/pipeline`](roadscan/pipeline) | Python pipeline that turns rides into a map: detection, jolt sensing, fusion, pothole tracking, blurring, map-matching, road/ward tagging. Also the web dashboard and API (`roadscan serve`) |
| [`roadscan/labeling`](roadscan/labeling) | Labelling our own ride frames in Label Studio: frame picking with pre-filled boxes, and conversion back to YOLO |
| [`roadscan/ml`](roadscan/ml) | Model training on RDD2022 (India), threshold calibration, and on-phone (TFLite) export |
| [`roadscan/demo`](roadscan/demo) | Demo map from synthetic rides, with ward summary |
| `models/` | Model files. Weights (`*.pt`) are not in git; the `*.fusion.json` threshold files are |

## Status
- The pipeline is tested end to end on synthetic rides.
- The model is trained on RDD2022 India (pothole AP50 0.40).
- Nothing has been tested on a real ride yet. That's the next milestone.

## Licences
- RDD2022 data is CC BY 4.0.
- Map data is © OpenStreetMap contributors (ODbL).
- Ultralytics YOLO is AGPL-3.0. Review this before any commercial deployment.
