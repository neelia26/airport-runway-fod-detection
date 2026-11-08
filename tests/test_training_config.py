from pathlib import Path

import yaml


def test_baseline_configuration_is_reproducible():
    config = yaml.safe_load(
        Path("configs/yolov8n_baseline.yaml").read_text(encoding="utf-8")
    )

    assert config["model"] == "yolov8n.pt"
    assert config["imgsz"] == 320
    assert config["seed"] == 42
    assert config["deterministic"] is True
    assert config["data"] == "data/yolo/dataset.yaml"
