# Baseline Training Protocol

## Purpose

The baseline is a lightweight single-class YOLOv8n detector trained on the original FOD-A images. It establishes the reference performance before any Gaussian or median preprocessing comparison.

## Fixed configuration

| Setting | Value | Rationale |
|---|---:|---|
| Model | YOLOv8n pretrained weights | Lightweight transfer-learning baseline with extensive documentation |
| Target class | `0: FOD` | Safety-focused localisation of all 31 source categories |
| Image size | 320 × 320 | Close to the native 300 × 300 resolution and manageable on the 8 GB M1 Mac |
| Epochs | 30 maximum | Moderate coursework run with early stopping |
| Batch size | 16 | Stable use of the available NVIDIA T4 while keeping the experiment reproducible |
| Patience | 8 epochs | Stops training if validation does not improve |
| Seed | 42 | Controls repeatable data-order and augmentation choices |
| Device | NVIDIA T4 (CUDA) | Practical accelerator available through Google Colab |
| Dataset split | Leakage-resistant manifest | Prevents capture sequences crossing partitions |

The seed and deterministic flag are retained, but exact bit-for-bit reproduction across different CUDA, PyTorch and Ultralytics versions is not claimed.

## Commands

Create a Python 3.12 environment and install the training dependencies:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv-train
.venv-train/bin/pip install -e '.[training]'
```

Verify the environment and pipeline:

```bash
.venv-train/bin/python scripts/check_training_environment.py
.venv-train/bin/python scripts/train_baseline.py --smoke-test
```

Start the full baseline:

```bash
.venv-train/bin/python scripts/train_baseline.py
```

The full run is resumable from its generated `last.pt` checkpoint using Ultralytics if interrupted. Training runs and weights remain excluded from Git because of their size; compact metrics and selected figures will be committed after verification.

## Recorded baseline run

The first exported Colab run is documented in `reports/tables/training/baseline/README.md`. It ended normally at epoch 27 through early stopping: the best mAP@50–95 occurred at epoch 19 and the configured patience of eight epochs was then exhausted. Its figures and complete epoch metrics are retained, but its validation results are not treated as final test-set performance. The locally retained `best.pt` checkpoint should be used for evaluation.

## Google Colab workflow

Use `notebooks/04_baseline_training_colab.ipynb` when local training is impractical. Before running it:

1. Upload `FODPascalVOCFormat-V.2.1.zip` to a folder in your Google Drive.
2. Open the notebook in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
4. Update the Drive path in the configuration cell if your ZIP is stored elsewhere.
5. Run the cells from top to bottom.

The notebook clones the `baseline-model-training` branch, extracts the source dataset into Colab's temporary storage, reproduces the YOLO conversion, runs the unchanged baseline configuration with `device=cuda`, and copies the complete run directory back to Google Drive.

## Experimental control

The Gaussian and median experiments must use this same architecture, initialization, split, image size, batch size, epoch limit, patience and seed. Changing multiple factors simultaneously would prevent a defensible preprocessing comparison.
