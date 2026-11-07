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
| Batch size | 8 | Conservative M1 unified-memory setting |
| Patience | 8 epochs | Stops training if validation does not improve |
| Seed | 42 | Controls repeatable data-order and augmentation choices |
| Device | Apple MPS | Uses the M1 GPU through PyTorch |
| Dataset split | Leakage-resistant manifest | Prevents capture sequences crossing partitions |

MPS currently warns that a small number of gradient operations do not have deterministic implementations. The seed and deterministic flag are retained, but exact bit-for-bit reproduction across hardware is not claimed.

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

## Experimental control

The Gaussian and median experiments must use this same architecture, initialization, split, image size, batch size, epoch limit, patience and seed. Changing multiple factors simultaneously would prevent a defensible preprocessing comparison.
