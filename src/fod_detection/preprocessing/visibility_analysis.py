"""Evaluate taught preprocessing techniques on small FOD visibility."""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


WEATHER_NAMES = {"0": "Dry", "1": "Wet"}
LIGHT_NAMES = {"0": "Bright", "1": "Dim", "2": "Dark"}
COCO_SMALL_AREA = 32**2


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    """Stretch the 2nd–98th percentile intensity range to 0–255."""

    low, high = np.percentile(image, (2, 98))
    if high <= low:
        return image.copy()
    stretched = (image.astype(np.float32) - low) * 255 / (high - low)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_transformations(image: np.ndarray) -> dict[str, np.ndarray]:
    """Return moderate candidate transformations taught in the module."""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return {
        "Original": image,
        "Contrast stretch": contrast_stretch(image),
        "CLAHE": clahe.apply(image),
        "Gaussian filter": cv2.GaussianBlur(image, (3, 3), 0),
        "Median filter": cv2.medianBlur(image, 3),
        "Sharpening": cv2.filter2D(image, -1, sharpening_kernel),
    }


def diagnostic_images(image: np.ndarray) -> dict[str, np.ndarray]:
    """Create edge, threshold and morphology visualisations."""

    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.normalize(cv2.magnitude(sobel_x, sobel_y), None, 0, 255, cv2.NORM_MINMAX)
    median_intensity = float(np.median(image))
    lower = int(max(0, 0.67 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    canny = cv2.Canny(image, lower, upper)
    _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    return {
        "Sobel": sobel.astype(np.uint8),
        "Canny": canny,
        "Otsu threshold": threshold,
        "Opening": opened,
        "Closing": closed,
    }


def load_small_objects(manifest_path: Path, annotation_dir: Path) -> pd.DataFrame:
    """Return all COCO-small boxes with split and sequence metadata."""

    with manifest_path.open(encoding="utf-8", newline="") as file:
        manifest = {row["image_id"]: row for row in csv.DictReader(file)}

    rows = []
    for image_id, metadata in manifest.items():
        root = ET.parse(annotation_dir / f"{image_id}.xml").getroot()
        for obj in root.findall("object"):
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            area = (xmax - xmin) * (ymax - ymin)
            if area < COCO_SMALL_AREA:
                rows.append(
                    {
                        "image_id": image_id,
                        "sequence_id": metadata["sequence_id"],
                        "split": metadata["split"],
                        "weather": WEATHER_NAMES[metadata["weather"]],
                        "light": LIGHT_NAMES[metadata["light"]],
                        "category": obj.findtext("name", "Unknown"),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "area": area,
                    }
                )
    return pd.DataFrame(rows)


def choose_group_balanced_sample(small_objects: pd.DataFrame) -> pd.DataFrame:
    """Choose the median-sized small box from each capture sequence."""

    selected_rows = []
    for _, group in small_objects.groupby("sequence_id"):
        median_area = group["area"].median()
        index = (group["area"] - median_area).abs().idxmin()
        selected_rows.append(small_objects.loc[index])
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def context_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return object and surrounding-context masks for a bounding box."""

    height, width = shape
    xmin, ymin, xmax, ymax = box
    box_width = xmax - xmin
    box_height = ymax - ymin
    context_xmin = max(0, xmin - box_width)
    context_ymin = max(0, ymin - box_height)
    context_xmax = min(width, xmax + box_width)
    context_ymax = min(height, ymax + box_height)

    object_mask = np.zeros(shape, dtype=bool)
    object_mask[ymin:ymax, xmin:xmax] = True
    surrounding_mask = np.zeros(shape, dtype=bool)
    surrounding_mask[context_ymin:context_ymax, context_xmin:context_xmax] = True
    surrounding_mask[object_mask] = False
    return object_mask, surrounding_mask


def visibility_metrics(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """Measure local contrast, edge energy and surrounding variation."""

    object_mask, surrounding_mask = context_mask(image.shape, box)
    object_pixels = image[object_mask].astype(np.float32)
    surrounding_pixels = image[surrounding_mask].astype(np.float32)
    contrast_to_noise = abs(object_pixels.mean() - surrounding_pixels.mean()) / (
        surrounding_pixels.std() + 1e-6
    )

    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = cv2.magnitude(sobel_x, sobel_y)[object_mask].mean()
    return float(contrast_to_noise), float(edge_energy), float(surrounding_pixels.std())


def evaluate_transformations(sample: pd.DataFrame, image_dir: Path) -> pd.DataFrame:
    """Calculate visibility metrics for every candidate transformation."""

    rows = []
    for position, item in enumerate(sample.itertuples(), start=1):
        image = cv2.imread(str(image_dir / f"{item.image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_dir / f"{item.image_id}.jpg")
        box = (
            math.floor(item.xmin),
            math.floor(item.ymin),
            math.ceil(item.xmax),
            math.ceil(item.ymax),
        )

        for technique, transformed in apply_transformations(image).items():
            local_contrast, edge_energy, background_std = visibility_metrics(transformed, box)
            rows.append(
                {
                    "image_id": item.image_id,
                    "sequence_id": item.sequence_id,
                    "split": item.split,
                    "weather": item.weather,
                    "light": item.light,
                    "category": item.category,
                    "area": item.area,
                    "technique": technique,
                    "local_contrast_to_noise": local_contrast,
                    "object_edge_energy": edge_energy,
                    "background_std": background_std,
                }
            )
        if position % 100 == 0:
            print(f"Analysed {position:,} sequence-balanced examples")
    return pd.DataFrame(rows)


def summarise_results(results: pd.DataFrame) -> pd.DataFrame:
    """Compare each technique with the paired original measurements."""

    original = results[results["technique"] == "Original"].set_index("image_id")
    rows = []
    for technique, group in results.groupby("technique", sort=False):
        indexed = group.set_index("image_id")
        contrast_change = indexed["local_contrast_to_noise"] - original["local_contrast_to_noise"]
        edge_change = indexed["object_edge_energy"] - original["object_edge_energy"]
        background_change = indexed["background_std"] - original["background_std"]
        rows.append(
            {
                "technique": technique,
                "median_contrast_change": float(contrast_change.median()),
                "contrast_improved_percent": float((contrast_change > 0).mean() * 100),
                "median_edge_change": float(edge_change.median()),
                "edge_improved_percent": float((edge_change > 0).mean() * 100),
                "median_background_std_change": float(background_change.median()),
            }
        )
    return pd.DataFrame(rows)


def save_intensity_histograms(
    sample: pd.DataFrame,
    image_dir: Path,
    output_path: Path,
) -> None:
    """Plot group-balanced pixel-intensity distributions by lighting label."""

    totals = {label: np.zeros(256, dtype=np.float64) for label in ("Bright", "Dim", "Dark")}
    for item in sample.drop_duplicates("image_id").itertuples():
        image = cv2.imread(str(image_dir / f"{item.image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        totals[item.light] += np.bincount(image.ravel(), minlength=256)

    figure, axis = plt.subplots(figsize=(9, 5))
    colours = {"Bright": "#d8b84a", "Dim": "#74899b", "Dark": "#25364a"}
    for label, counts in totals.items():
        axis.plot(counts / counts.sum(), label=label, color=colours[label])
    axis.set(
        title="Group-balanced intensity histograms by FOD-A lighting label",
        xlabel="Grayscale intensity",
        ylabel="Proportion of pixels",
    )
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_metric_figure(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot paired median visibility changes and improvement frequency."""

    plotted = summary[summary["technique"] != "Original"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=plotted, y="technique", x="median_contrast_change", ax=axes[0], color="#4c78a8")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set(title="Median change in local contrast-to-noise", xlabel="Change from original", ylabel="")
    sns.barplot(data=plotted, y="technique", x="contrast_improved_percent", ax=axes[1], color="#72b7b2")
    axes[1].axvline(50, color="black", linestyle="--", linewidth=1)
    axes[1].set(title="Small objects with improved local contrast", xlabel="Examples improved (%)", ylabel="")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def crop_around_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    padding: int = 45,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop around an object and translate its box coordinates."""

    xmin, ymin, xmax, ymax = box
    left = max(0, xmin - padding)
    top = max(0, ymin - padding)
    right = min(image.shape[1], xmax + padding)
    bottom = min(image.shape[0], ymax + padding)
    return image[top:bottom, left:right], (xmin - left, ymin - top, xmax - left, ymax - top)


def save_technique_grid(sample: pd.DataFrame, image_dir: Path, output_path: Path) -> None:
    """Show all taught operations on four condition-diverse small objects."""

    examples = []
    for _, group in sample.groupby(["weather", "light"], sort=True):
        examples.append(group.sort_values("area").iloc[len(group) // 2])

    technique_names = [
        "Original",
        "Contrast stretch",
        "CLAHE",
        "Gaussian filter",
        "Median filter",
        "Sharpening",
        "Sobel",
        "Canny",
        "Otsu threshold",
        "Opening",
        "Closing",
    ]
    figure, axes = plt.subplots(len(examples), len(technique_names), figsize=(22, 2.5 * len(examples)))

    for row_index, item in enumerate(examples):
        image = cv2.imread(str(image_dir / f"{item.image_id}.jpg"), cv2.IMREAD_GRAYSCALE)
        box = (
            math.floor(item.xmin),
            math.floor(item.ymin),
            math.ceil(item.xmax),
            math.ceil(item.ymax),
        )
        transformed = apply_transformations(image) | diagnostic_images(image)
        for column_index, name in enumerate(technique_names):
            crop, crop_box = crop_around_box(transformed[name], box)
            display_image = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(
                display_image,
                (crop_box[0], crop_box[1]),
                (crop_box[2], crop_box[3]),
                (255, 0, 0),
                1,
            )
            axis = axes[row_index, column_index]
            axis.imshow(display_image)
            if row_index == 0:
                axis.set_title(name, fontsize=9)
            if column_index == 0:
                axis.set_ylabel(f"{item.light} · {item.weather}\n{item.image_id}")
            axis.set_xticks([])
            axis.set_yticks([])

    figure.suptitle("Taught preprocessing techniques on small FOD (red ground-truth box)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary(
    sample: pd.DataFrame,
    metrics: pd.DataFrame,
    technique_summary: pd.DataFrame,
    output_dir: Path,
    small_object_count: int,
) -> None:
    """Write compact tables and a machine-readable summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    technique_summary.to_csv(output_dir / "technique_comparison.csv", index=False)
    condition_summary = (
        metrics.groupby(["technique", "light", "weather"])[
            ["local_contrast_to_noise", "object_edge_energy", "background_std"]
        ]
        .median()
        .reset_index()
    )
    condition_summary.to_csv(output_dir / "condition_metrics.csv", index=False)

    comparison = technique_summary.set_index("technique")
    best_contrast = comparison.drop(index="Original")["median_contrast_change"].idxmax()
    most_consistent = comparison.drop(index="Original")["contrast_improved_percent"].idxmax()
    summary = {
        "small_objects_in_dataset": int(small_object_count),
        "sequence_balanced_examples": int(len(sample)),
        "available_conditions": sorted(
            f"{light}-{weather}"
            for light, weather in sample[["light", "weather"]].drop_duplicates().itertuples(index=False, name=None)
        ),
        "candidate_input_techniques": [
            "Contrast stretch",
            "CLAHE",
            "Gaussian filter",
            "Median filter",
            "Sharpening",
        ],
        "diagnostic_techniques": ["Sobel", "Canny", "Otsu threshold", "Opening", "Closing"],
        "best_median_contrast_change": best_contrast,
        "most_consistent_contrast_improvement": most_consistent,
        "training_recommendation": {
            "primary_baseline": "Original images",
            "controlled_variants": ["Gaussian filter", "Median filter"],
            "not_recommended_as_default": [
                "Contrast stretch",
                "CLAHE",
                "Sharpening",
                "Sobel",
                "Canny",
                "Otsu threshold",
                "Opening",
                "Closing",
            ],
            "reason": (
                "Mild filtering improved the local contrast-to-noise proxy but reduced edge energy. "
                "Enhancement and sharpening increased edges while usually reducing local contrast "
                "or amplifying pavement texture. Model experiments are required to resolve the trade-off."
            ),
        },
        "technique_results": technique_summary.round(6).to_dict("records"),
        "interpretation_warning": (
            "Visibility proxies do not prove improved detector accuracy. "
            "Any promising transformation must be tested in a controlled model experiment."
        ),
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
