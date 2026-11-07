"""Report whether the local computer is ready for YOLO training."""

import platform

import torch
import ultralytics


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        raise SystemExit("No GPU accelerator is available; use CPU or a cloud GPU")


if __name__ == "__main__":
    main()
