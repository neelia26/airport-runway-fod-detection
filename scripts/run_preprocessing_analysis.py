"""Run the group-balanced small-FOD preprocessing analysis."""

from pathlib import Path

from fod_detection.preprocessing.visibility_analysis import (
    choose_group_balanced_sample,
    evaluate_transformations,
    load_small_objects,
    save_intensity_histograms,
    save_metric_figure,
    save_technique_grid,
    summarise_results,
    write_summary,
)


VOC_ROOT = Path("FOD-A Dataset/FODPascalVOCFormat-V.2.1/VOC2007")
MANIFEST = Path("data/splits/final/manifest.csv")
FIGURE_DIR = Path("reports/figures/preprocessing")
TABLE_DIR = Path("reports/tables/preprocessing")


def main() -> None:
    image_dir = VOC_ROOT / "JPEGImages"
    small_objects = load_small_objects(MANIFEST, VOC_ROOT / "Annotations")
    sample = choose_group_balanced_sample(small_objects)
    metrics = evaluate_transformations(sample, image_dir)
    technique_summary = summarise_results(metrics)

    save_intensity_histograms(sample, image_dir, FIGURE_DIR / "intensity_histograms.png")
    save_metric_figure(technique_summary, FIGURE_DIR / "visibility_metrics.png")
    save_technique_grid(sample, image_dir, FIGURE_DIR / "technique_comparison_grid.png")
    write_summary(
        sample,
        metrics,
        technique_summary,
        TABLE_DIR,
        small_object_count=len(small_objects),
    )

    print(f"Small objects found: {len(small_objects):,}")
    print(f"Sequence-balanced examples: {len(sample):,}")
    print(technique_summary.to_string(index=False))


if __name__ == "__main__":
    main()
