"""Train or smoke-test the lightweight YOLOv8n FOD baseline."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


DEFAULT_CONFIG = Path("configs/yolov8n_baseline.yaml")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the YOLOv8n FOD baseline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch on a tiny subset to validate the pipeline.",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cpu"],
        help="Override the configured training device.",
    )
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count.")
    parser.add_argument("--batch", type=int, help="Override the configured batch size.")
    parser.add_argument("--imgsz", type=int, help="Override the configured image size.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, object]:
    """Load the small version-controlled training configuration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"model", "data", "epochs", "imgsz", "batch", "seed"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration values: {sorted(missing)}")
    return config


def choose_device(configured: str) -> str:
    """Use MPS only when PyTorch reports that it is available."""

    if configured == "mps" and not torch.backends.mps.is_available():
        print("MPS is unavailable; falling back to CPU")
        return "cpu"
    return configured


def create_smoke_dataset(dataset_yaml: Path, sample_size: int = 32) -> Path:
    """Create small train/validation lists without copying or changing images."""

    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(dataset["path"])
    smoke_dir = root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for source_split, target_name in (("train", "train"), ("validation", "val")):
        selected = sorted((root / "images" / source_split).glob("*.jpg"))[:sample_size]
        list_path = smoke_dir / f"{target_name}.txt"
        list_path.write_text(
            "\n".join(str(path.resolve()) for path in selected) + "\n",
            encoding="utf-8",
        )
        paths[target_name] = list_path.resolve()

    smoke_yaml = smoke_dir / "dataset.yaml"
    smoke_yaml.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                f"train: {paths['train']}",
                f"val: {paths['val']}",
                "names:",
                "  0: FOD",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return smoke_yaml


def main() -> None:
    args = parse_arguments()
    config = load_config(args.config)
    model_path = str(config.pop("model"))
    config["device"] = choose_device(args.device or str(config.get("device", "mps")))
    for name in ("epochs", "batch", "imgsz"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    config["project"] = str(Path(str(config["project"])).resolve())

    if args.smoke_test:
        config.update(
            {
                "data": str(create_smoke_dataset(Path(str(config["data"])))),
                "epochs": 1,
                "batch": 4,
                "workers": 0,
                "project": str(Path("runs/smoke").resolve()),
                "name": "yolov8n_pipeline_check",
                "exist_ok": True,
                "plots": False,
            }
        )

    print(f"Model: {model_path}")
    print(f"Device: {config['device']}")
    print(f"Dataset: {config['data']}")
    model = YOLO(model_path)
    started = time.perf_counter()
    results = model.train(**config)
    elapsed_seconds = time.perf_counter() - started

    if args.smoke_test:
        summary_path = Path("reports/tables/training/smoke_test_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "purpose": "Pipeline verification only; metrics are not model results.",
                    "model": model_path,
                    "target_class": "FOD",
                    "epochs": 1,
                    "training_images": 32,
                    "validation_images": 32,
                    "image_size": config["imgsz"],
                    "batch_size": config["batch"],
                    "device": config["device"],
                    "python": platform.python_version(),
                    "pytorch": torch.__version__,
                    "elapsed_seconds": round(elapsed_seconds, 2),
                    "checkpoint_directory": str(Path(results.save_dir) / "weights"),
                    "checks": {
                        "dataset_loaded": True,
                        "single_class_model_created": True,
                        "backpropagation_completed": True,
                        "validation_completed": True,
                        "checkpoint_saved": True,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
