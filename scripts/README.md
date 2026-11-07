# Scripts

This directory contains short command-line entry points. Data parsing, preprocessing, modelling and evaluation logic should remain in the importable `src/fod_detection/` package so that it can be reused by scripts, notebooks and tests.

Current commands:

- `python scripts/build_leakage_resistant_split.py` rebuilds the frozen manifests.
- `python scripts/verify_split.py` independently checks completeness, uniqueness and sequence isolation.
- `python scripts/run_eda.py` generates the exploratory-analysis tables, figures and example grids.
- `python scripts/convert_voc_to_yolo.py` validates Pascal VOC XML and creates the local single-class YOLO dataset.
- `python scripts/verify_yolo_conversion.py` independently checks image-label coverage, object preservation and label syntax.
- `python scripts/run_preprocessing_analysis.py` runs the sequence-balanced visibility experiment for the selected lecture techniques.
- `python scripts/check_training_environment.py` verifies PyTorch, Ultralytics and Apple MPS availability.
- `python scripts/train_baseline.py --smoke-test` validates the complete YOLO training pipeline on a tiny non-reportable subset.
- `python scripts/train_baseline.py` starts the configured full YOLOv8n baseline run.
