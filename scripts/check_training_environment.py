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
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; use CPU or a cloud GPU")


if __name__ == "__main__":
    main()
