"""Build and verify the single-class YOLO version of FOD-A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fod_detection.data.yolo_conversion import (
    convert_dataset,
    save_visual_verification,
    verify_converted_dataset,
)


DEFAULT_VOC_ROOT = Path("FOD-A Dataset/FODPascalVOCFormat-V.2.1/VOC2007")
DEFAULT_MANIFEST = Path("data/splits/final/manifest.csv")
DEFAULT_OUTPUT = Path("data/yolo")
SUMMARY_PATH = Path("reports/tables/annotation_conversion/conversion_summary.json")
FIGURE_PATH = Path("reports/figures/annotation_conversion/yolo_box_verification.png")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FOD-A Pascal VOC annotations to one-class YOLO format."
    )
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    conversion = convert_dataset(
        args.voc_root,
        args.manifest,
        args.output,
        overwrite=args.overwrite,
    )
    verification = verify_converted_dataset(args.manifest, args.output)
    save_visual_verification(args.output, FIGURE_PATH)

    summary = {"conversion": conversion, "verification": verification}
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Converted {conversion['images']:,} images")
    print(f"Converted {conversion['objects']:,} objects to class 0: FOD")
    print(f"Maximum round-trip error: {conversion['maximum_roundtrip_error_pixels']:.8f} pixels")
    print(f"Generated dataset: {args.output}")


if __name__ == "__main__":
    main()
