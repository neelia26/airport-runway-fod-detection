"""Convert FOD-A Pascal VOC annotations to single-class YOLO format."""

from __future__ import annotations

import csv
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class BoundingBox:
    """One validated Pascal VOC object box."""

    source_category: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Annotation:
    """The information required from one Pascal VOC XML file."""

    image_id: str
    width: int
    height: int
    boxes: tuple[BoundingBox, ...]


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read and validate the corrected partition manifest."""

    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"image_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    if len(rows) != len({row["image_id"] for row in rows}):
        raise ValueError("Manifest contains duplicate image IDs")
    if {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation and test partitions")
    return rows


def read_voc_annotation(xml_path: Path) -> Annotation:
    """Read one XML file and reject missing or invalid boxes."""

    root = ET.parse(xml_path).getroot()
    image_id = xml_path.stem
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    boxes = []
    for obj in root.findall("object"):
        category = obj.findtext("name", "").strip()
        box = obj.find("bndbox")
        if not category or box is None:
            raise ValueError(f"Missing category or bounding box in {xml_path}")

        xmin = float(box.findtext("xmin", "nan"))
        ymin = float(box.findtext("ymin", "nan"))
        xmax = float(box.findtext("xmax", "nan"))
        ymax = float(box.findtext("ymax", "nan"))
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            raise ValueError(
                f"Invalid box in {xml_path}: {(xmin, ymin, xmax, ymax)}"
            )
        boxes.append(BoundingBox(category, xmin, ymin, xmax, ymax))

    if not boxes:
        raise ValueError(f"No annotated objects in {xml_path}")
    return Annotation(image_id, width, height, tuple(boxes))


def voc_box_to_yolo(box: BoundingBox, width: int, height: int) -> tuple[float, ...]:
    """Convert corner coordinates to normalized YOLO centre coordinates."""

    x_center = ((box.xmin + box.xmax) / 2) / width
    y_center = ((box.ymin + box.ymax) / 2) / height
    box_width = (box.xmax - box.xmin) / width
    box_height = (box.ymax - box.ymin) / height
    values = (x_center, y_center, box_width, box_height)
    if not all(0 < value <= 1 for value in values):
        raise ValueError(f"YOLO coordinates outside (0, 1]: {values}")
    return values


def yolo_box_to_voc(
    values: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Reconstruct corner coordinates for verification and visualisation."""

    x_center, y_center, box_width, box_height = values
    xmin = (x_center - box_width / 2) * width
    ymin = (y_center - box_height / 2) * height
    xmax = (x_center + box_width / 2) * width
    ymax = (y_center + box_height / 2) * height
    return xmin, ymin, xmax, ymax


def prepare_output(output_root: Path, overwrite: bool) -> None:
    """Create an empty model-ready directory without touching source data."""

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_root} is not empty; pass --overwrite to rebuild it"
            )
        shutil.rmtree(output_root)

    for split in ("train", "validation", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_dataset(
    voc_root: Path,
    manifest_path: Path,
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert all manifest images and annotations into a single FOD class."""

    rows = read_manifest(manifest_path)
    prepare_output(output_root, overwrite)
    source_images = voc_root / "JPEGImages"
    source_annotations = voc_root / "Annotations"

    partition_images = Counter()
    partition_objects = Counter()
    partition_multi_object_images = Counter()
    source_categories = Counter()
    maximum_roundtrip_error = 0.0

    for position, row in enumerate(rows, start=1):
        image_id = row["image_id"]
        split = row["split"]
        image_path = source_images / f"{image_id}.jpg"
        annotation = read_voc_annotation(source_annotations / f"{image_id}.xml")
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        label_lines = []
        if len(annotation.boxes) > 1:
            partition_multi_object_images[split] += 1
        for box in annotation.boxes:
            yolo_values = voc_box_to_yolo(box, annotation.width, annotation.height)
            label_line = "0 " + " ".join(f"{value:.8f}" for value in yolo_values)
            label_lines.append(label_line)
            source_categories[box.source_category] += 1
            partition_objects[split] += 1

            stored_values = tuple(float(value) for value in label_line.split()[1:])
            reconstructed = yolo_box_to_voc(
                stored_values,
                annotation.width,
                annotation.height,
            )
            original = (box.xmin, box.ymin, box.xmax, box.ymax)
            maximum_roundtrip_error = max(
                maximum_roundtrip_error,
                *(abs(first - second) for first, second in zip(original, reconstructed)),
            )

        shutil.copy2(image_path, output_root / "images" / split / image_path.name)
        (output_root / "labels" / split / f"{image_id}.txt").write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8",
        )
        partition_images[split] += 1
        if position % 5000 == 0:
            print(f"Converted {position:,} images")

    (output_root / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {output_root.resolve()}",
                "train: images/train",
                "val: images/validation",
                "test: images/test",
                "names:",
                "  0: FOD",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_root / "class_mapping.json").write_text(
        json.dumps(
            {
                "target_class": {"id": 0, "name": "FOD"},
                "source_categories": dict(sorted(source_categories.items())),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "images": sum(partition_images.values()),
        "objects": sum(partition_objects.values()),
        "source_category_count": len(source_categories),
        "target_classes": {"0": "FOD"},
        "partition_images": dict(partition_images),
        "partition_objects": dict(partition_objects),
        "partition_multi_object_images": {
            split: partition_multi_object_images[split]
            for split in ("train", "validation", "test")
        },
        "maximum_roundtrip_error_pixels": maximum_roundtrip_error,
    }


def verify_converted_dataset(
    manifest_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """Independently verify file coverage and every normalized label line."""

    rows = read_manifest(manifest_path)
    expected = Counter(row["split"] for row in rows)
    expected_objects = Counter()
    for row in rows:
        expected_objects[row["split"]] += int(row["object_count"])
    image_counts = Counter()
    label_counts = Counter()
    object_counts = Counter()
    invalid_lines = []

    for split in ("train", "validation", "test"):
        images = sorted((output_root / "images" / split).glob("*.jpg"))
        labels = sorted((output_root / "labels" / split).glob("*.txt"))
        image_counts[split] = len(images)
        label_counts[split] = len(labels)
        image_ids = {path.stem for path in images}
        label_ids = {path.stem for path in labels}
        if image_ids != label_ids:
            invalid_lines.append(f"Image-label ID mismatch in {split}")

        for label_path in labels:
            for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                    valid = (
                        len(parts) == 5
                        and class_id == 0
                        and all(0 < value <= 1 for value in coordinates)
                    )
                except (ValueError, IndexError):
                    valid = False
                if not valid:
                    invalid_lines.append(f"{label_path}:{line_number}")
                object_counts[split] += 1

    checks = {
        "all_manifest_images_present": image_counts == expected,
        "one_label_file_per_image": label_counts == expected,
        "all_objects_preserved": object_counts == expected_objects,
        "all_labels_are_single_class_fod": not invalid_lines,
    }
    if not all(checks.values()):
        raise ValueError(f"Converted dataset verification failed: {checks}")

    return {
        "checks": checks,
        "image_counts": dict(image_counts),
        "label_counts": dict(label_counts),
        "object_counts": dict(object_counts),
        "invalid_lines": invalid_lines,
    }


def save_visual_verification(
    output_root: Path,
    output_path: Path,
) -> None:
    """Draw converted YOLO boxes for three examples from every partition."""

    figure, axes = plt.subplots(3, 3, figsize=(10, 10))
    for row_index, split in enumerate(("train", "validation", "test")):
        label_paths = sorted((output_root / "labels" / split).glob("*.txt"))
        selected = [
            label_paths[len(label_paths) // 4],
            label_paths[len(label_paths) // 2],
            label_paths[3 * len(label_paths) // 4],
        ]
        for column_index, label_path in enumerate(selected):
            image_path = output_root / "images" / split / f"{label_path.stem}.jpg"
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for line in label_path.read_text().splitlines():
                _, *coordinates = line.split()
                box = yolo_box_to_voc(
                    tuple(float(value) for value in coordinates),
                    image.width,
                    image.height,
                )
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0] + 2, max(0, box[1] - 12)), "FOD", fill="red")

            axis = axes[row_index, column_index]
            axis.imshow(image)
            axis.set_title(f"{split} · {label_path.stem}")
            axis.axis("off")

    figure.suptitle("Visual verification of converted YOLO boxes")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
