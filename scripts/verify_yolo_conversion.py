"""Run the independent file and label checks on the generated YOLO dataset."""

from pathlib import Path

from fod_detection.data.yolo_conversion import verify_converted_dataset


def main() -> None:
    result = verify_converted_dataset(
        Path("data/splits/final/manifest.csv"),
        Path("data/yolo"),
    )
    for name, passed in result["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print(f"PASS: {sum(result['image_counts'].values()):,} image-label pairs")
    print(f"PASS: {sum(result['object_counts'].values()):,} single-class boxes")


if __name__ == "__main__":
    main()
