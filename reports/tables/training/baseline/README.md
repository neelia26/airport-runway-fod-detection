# YOLOv8n Baseline Run

## Run status

This export is a genuine completed Colab T4 baseline run on the original FOD images. Although the maximum was 30 epochs, training ended normally at epoch 27 through the configured early-stopping rule. The best mAP@50–95 occurred at epoch 19; eight subsequent epochs did not improve it, exhausting `patience: 8`. Therefore, rerunning the 30 epochs is neither required nor methodologically preferable.

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
| Completed epochs | 27 (early stopping) |
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
