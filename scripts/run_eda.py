"""Run the complete exploratory analysis and save report-ready outputs."""

from pathlib import Path

from fod_detection.data.exploratory_analysis import (
    choose_difficult_images,
    choose_representative_images,
    load_image_table,
    load_object_table,
    save_distribution_figures,
    save_example_grid,
    save_tables,
)


VOC_ROOT = Path("FOD-A Dataset/FODPascalVOCFormat-V.2.1/VOC2007")
MANIFEST = Path("data/splits/final/manifest.csv")
FIGURE_DIR = Path("reports/figures/eda")
TABLE_DIR = Path("reports/tables/eda")


def main() -> None:
    image_dir = VOC_ROOT / "JPEGImages"
    annotation_dir = VOC_ROOT / "Annotations"

    images = load_image_table(MANIFEST, image_dir)
    objects = load_object_table(annotation_dir, images)
    findings = save_tables(images, objects, TABLE_DIR)
    save_distribution_figures(images, objects, FIGURE_DIR)

    save_example_grid(
        choose_representative_images(images),
        images,
        objects,
        image_dir,
        FIGURE_DIR / "representative_images.png",
        "Representative FOD-A images",
    )
    save_example_grid(
        choose_difficult_images(images, objects),
        images,
        objects,
        image_dir,
        FIGURE_DIR / "difficult_examples.png",
        "Potentially difficult examples",
    )

    print(f"Analysed {findings['images']:,} images and {findings['objects']:,} objects")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
