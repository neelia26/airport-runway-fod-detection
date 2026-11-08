# YOLOv8n Baseline Run

## Run status

This export is a genuine Colab T4 baseline run on the original FOD images. It contains 27 recorded epochs although the configuration requested 30. The run was interrupted by the Colab connection, so it must be reported as an interrupted baseline rather than a completed 30-epoch experiment.

## Configuration

| Setting | Value |
|---|---:|
| Model | YOLOv8n pretrained |
| Target | One `FOD` class |
| Input size | 320 × 320 |
| Batch size | 16 |
| Device | NVIDIA T4 CUDA |
| Seed | 42 |
| Requested epochs | 30 |
| Recorded epochs | 27 |
| Recorded training time | 4,975.21 seconds (82.92 minutes) |

## Validation findings

| Selection | Epoch | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|
| Highest mAP@50–95 | 19 | 0.97982 | 0.95928 | 0.97199 | 0.74448 |
| Highest mAP@50 | 20 | 0.98373 | 0.95811 | 0.97271 | 0.74152 |
| Last recorded epoch | 27 | 0.98007 | 0.95274 | 0.97170 | 0.73827 |

These are validation-set results produced during training. They are not final test-set results and must not be used to answer the research question until the saved best checkpoint is evaluated on the untouched test partition, including performance by object size and environment.

## Retained artefacts

- `args.yaml`: exact Ultralytics run arguments.
- `results.csv`: all 27 recorded epoch metrics.
- `reports/figures/training/baseline/`: learning curves, confusion matrices and one label/prediction comparison.
- `models/baseline/yolov8n_original/`: local `best.pt` and `last.pt` checkpoints. Model weights are intentionally excluded from Git.
