"""Exploratory analysis utilities for the FOD-A dataset."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageStat


WEATHER_NAMES = {"0": "Dry", "1": "Wet"}
LIGHT_NAMES = {"0": "Bright", "1": "Dim", "2": "Dark"}


def object_size(area_pixels: float) -> str:
    """Return the COCO size category at the native image resolution."""

    if area_pixels < 32**2:
        return "Small"
    if area_pixels < 96**2:
        return "Medium"
    return "Large"


def load_image_table(manifest_path: Path, image_dir: Path) -> pd.DataFrame:
    """Load the split manifest and calculate simple image statistics."""

    images = pd.read_csv(manifest_path, dtype={"image_id": str, "weather": str, "light": str})
    images["weather_name"] = images["weather"].map(WEATHER_NAMES)
    images["light_name"] = images["light"].map(LIGHT_NAMES)

    brightness = []
    contrast = []
    for index, image_id in enumerate(images["image_id"], start=1):
        with Image.open(image_dir / f"{image_id}.jpg") as image:
            grayscale = image.convert("L")
            statistics = ImageStat.Stat(grayscale)
            brightness.append(statistics.mean[0])
            contrast.append(statistics.stddev[0])
        if index % 5000 == 0:
            print(f"Measured {index:,} images")

    images["mean_brightness"] = brightness
    images["contrast_std"] = contrast
    return images


def load_object_table(annotation_dir: Path, images: pd.DataFrame) -> pd.DataFrame:
    """Read Pascal VOC boxes and return one row per annotated object."""

    image_metadata = images.set_index("image_id").to_dict("index")
    rows = []
    for image_id, metadata in image_metadata.items():
        root = ET.parse(annotation_dir / f"{image_id}.xml").getroot()
        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))

        for obj in root.findall("object"):
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            box_width = xmax - xmin
            box_height = ymax - ymin
            area_pixels = box_width * box_height

            rows.append(
                {
                    "image_id": image_id,
                    "category": obj.findtext("name", "Unknown"),
                    "split": metadata["split"],
                    "weather": metadata["weather_name"],
                    "light": metadata["light_name"],
                    "image_width": width,
                    "image_height": height,
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "box_width": box_width,
                    "box_height": box_height,
                    "box_area_pixels": area_pixels,
                    "box_area_fraction": area_pixels / (width * height),
                    "size": object_size(area_pixels),
                }
            )

    return pd.DataFrame(rows)


def cramers_v(table: pd.DataFrame) -> float:
    """Measure association between two categorical variables."""

    observed = table.to_numpy()
    total = observed.sum()
    row_totals = observed.sum(axis=1)
    column_totals = observed.sum(axis=0)
    expected = row_totals[:, None] * column_totals[None, :] / total
    chi_squared = ((observed - expected) ** 2 / expected).sum()
    denominator = total * min(observed.shape[0] - 1, observed.shape[1] - 1)
    return math.sqrt(chi_squared / denominator) if denominator else 0.0


def save_tables(
    images: pd.DataFrame,
    objects: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    """Save compact report-ready tables and return the main findings."""

    output_dir.mkdir(parents=True, exist_ok=True)
    category_counts = objects["category"].value_counts().rename_axis("category").reset_index(name="objects")
    size_counts = objects["size"].value_counts().reindex(["Small", "Medium", "Large"], fill_value=0)
    light_counts = images["light_name"].value_counts().reindex(["Bright", "Dim", "Dark"], fill_value=0)
    weather_counts = images["weather_name"].value_counts().reindex(["Dry", "Wet"], fill_value=0)
    split_counts = images["split"].value_counts().reindex(["train", "validation", "test"], fill_value=0)

    category_counts.to_csv(output_dir / "category_distribution.csv", index=False)
    size_counts.rename("objects").to_csv(output_dir / "object_size_distribution.csv")
    pd.crosstab(objects["category"], objects["light"]).to_csv(output_dir / "class_by_lighting.csv")
    pd.crosstab(objects["category"], objects["weather"]).to_csv(output_dir / "class_by_weather.csv")

    class_light = pd.crosstab(objects["category"], objects["light"])
    class_weather = pd.crosstab(objects["category"], objects["weather"])
    findings = {
        "images": int(len(images)),
        "annotations": int(len(images)),
        "objects": int(len(objects)),
        "categories": int(objects["category"].nunique()),
        "partition_images": {key: int(value) for key, value in split_counts.items()},
        "lighting_images": {key: int(value) for key, value in light_counts.items()},
        "weather_images": {key: int(value) for key, value in weather_counts.items()},
        "objects_per_image": {
            str(key): int(value)
            for key, value in images["object_count"].value_counts().sort_index().items()
        },
        "object_sizes": {key: int(value) for key, value in size_counts.items()},
        "box_area_fraction_percentiles": {
            str(percentile): round(float(objects["box_area_fraction"].quantile(percentile)), 6)
            for percentile in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        },
        "class_lighting_cramers_v": round(cramers_v(class_light), 4),
        "class_weather_cramers_v": round(cramers_v(class_weather), 4),
        "lowest_contrast_image_ids": images.nsmallest(8, "contrast_std")["image_id"].tolist(),
        "smallest_object_image_ids": objects.nsmallest(8, "box_area_fraction")["image_id"].tolist(),
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(findings, indent=2) + "\n",
        encoding="utf-8",
    )
    return findings


def save_distribution_figures(
    images: pd.DataFrame,
    objects: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create the main report-ready EDA charts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    category_counts = objects["category"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(9, 10))
    category_counts.plot.barh(ax=axis, color="#2f5d50")
    axis.set(title="FOD category distribution", xlabel="Annotated objects", ylabel="")
    figure.tight_layout()
    figure.savefig(output_dir / "category_distribution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    images["light_name"].value_counts().reindex(["Bright", "Dim", "Dark"]).plot.bar(
        ax=axes[0], color=["#d8b84a", "#74899b", "#25364a"], rot=0
    )
    axes[0].set(title="Lighting distribution", xlabel="", ylabel="Images")
    images["weather_name"].value_counts().reindex(["Dry", "Wet"]).plot.bar(
        ax=axes[1], color=["#c9a66b", "#4d86a8"], rot=0
    )
    axes[1].set(title="Weather distribution", xlabel="", ylabel="Images")
    figure.tight_layout()
    figure.savefig(output_dir / "environment_distribution.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    images["object_count"].value_counts().sort_index().plot.bar(ax=axes[0], color="#4c78a8", rot=0)
    axes[0].set(title="Objects per image", xlabel="Objects", ylabel="Images")
    objects["size"].value_counts().reindex(["Small", "Medium", "Large"]).plot.bar(
        ax=axes[1], color=["#e45756", "#72b7b2", "#54a24b"], rot=0
    )
    axes[1].set(title="Object-size distribution", xlabel="COCO size", ylabel="Objects")
    figure.tight_layout()
    figure.savefig(output_dir / "objects_and_sizes.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.hexbin(objects["box_width"], objects["box_height"], gridsize=45, bins="log", cmap="viridis")
    axis.axvline(32, color="white", linestyle="--", linewidth=1)
    axis.axhline(32, color="white", linestyle="--", linewidth=1)
    axis.set(title="Bounding-box dimensions", xlabel="Width (pixels)", ylabel="Height (pixels)")
    figure.tight_layout()
    figure.savefig(output_dir / "bounding_box_dimensions.png", dpi=180)
    plt.close(figure)

    light_table = pd.crosstab(objects["category"], objects["light"], normalize="index")
    weather_table = pd.crosstab(objects["category"], objects["weather"], normalize="index")
    figure, axes = plt.subplots(1, 2, figsize=(12, 10), gridspec_kw={"width_ratios": [3, 2]})
    sns.heatmap(light_table, ax=axes[0], cmap="Blues", vmin=0, vmax=1)
    sns.heatmap(weather_table, ax=axes[1], cmap="Greens", vmin=0, vmax=1)
    axes[0].set(title="Within-class lighting proportions", xlabel="", ylabel="Category")
    axes[1].set(title="Within-class weather proportions", xlabel="", ylabel="")
    figure.tight_layout()
    figure.savefig(output_dir / "class_environment_associations.png", dpi=180)
    plt.close(figure)


def draw_annotated_image(image_path: Path, rows: pd.DataFrame) -> Image.Image:
    """Return an image with its ground-truth boxes and labels drawn."""

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in rows.itertuples():
        draw.rectangle((row.xmin, row.ymin, row.xmax, row.ymax), outline="red", width=3)
        draw.text((row.xmin + 2, max(0, row.ymin - 12)), row.category, fill="red")
    return image


def save_example_grid(
    image_ids: list[str],
    images: pd.DataFrame,
    objects: pd.DataFrame,
    image_dir: Path,
    output_path: Path,
    title: str,
) -> None:
    """Save an annotated contact sheet for selected image IDs."""

    unique_ids = list(dict.fromkeys(image_ids))[:8]
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis, image_id in zip(axes.flat, unique_ids):
        object_rows = objects[objects["image_id"] == image_id]
        image = draw_annotated_image(image_dir / f"{image_id}.jpg", object_rows)
        metadata = images.loc[images["image_id"] == image_id].iloc[0]
        axis.imshow(image)
        axis.set_title(f"{image_id} · {metadata.light_name} · {metadata.weather_name}", fontsize=9)
        axis.axis("off")
    for axis in axes.flat[len(unique_ids):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def choose_representative_images(images: pd.DataFrame) -> list[str]:
    """Choose two mid-brightness examples from every available condition."""

    selected = []
    for _, group in images.groupby(["weather_name", "light_name"], sort=True):
        ordered = group.sort_values("mean_brightness")
        selected.extend(
            [ordered.iloc[len(ordered) // 3]["image_id"], ordered.iloc[2 * len(ordered) // 3]["image_id"]]
        )
    return selected


def choose_difficult_images(images: pd.DataFrame, objects: pd.DataFrame) -> list[str]:
    """Choose varied examples with tiny objects, low contrast and adverse conditions."""

    chosen: list[str] = []
    used_sequences: set[str] = set()

    def add_from(candidates: list[str], required: int) -> None:
        added = 0
        for image_id in candidates:
            sequence_id = images.loc[images["image_id"] == image_id, "sequence_id"].iloc[0]
            if sequence_id not in used_sequences:
                chosen.append(image_id)
                used_sequences.add(sequence_id)
                added += 1
            if added == required:
                return

    add_from(objects.nsmallest(100, "box_area_fraction")["image_id"].tolist(), 3)
    add_from(images.nsmallest(100, "contrast_std")["image_id"].tolist(), 3)

    adverse = images[
        (images["light_name"].isin(["Dim", "Dark"]))
        | (images["weather_name"] == "Wet")
    ]
    add_from(adverse.sort_values("contrast_std")["image_id"].tolist(), 2)
    return chosen
