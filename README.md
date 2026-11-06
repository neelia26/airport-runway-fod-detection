# Sri Lankan Tea Leaf Dataset Metadata

This repository provides the metadata, class-count summary, and leakage-safe split manifests for the **Sri Lankan Tea Leaf Dataset**. The image files are intentionally not duplicated here. Download the complete image dataset from the authoritative Mendeley Data record:

- **Official dataset:** [Sri Lankan Tea Leaf Dataset, Version 5](https://doi.org/10.17632/sjmy6k24d6.5)
- **Mendeley Data page:** <https://data.mendeley.com/datasets/sjmy6k24d6/5>
- **Licence:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

## Dataset summary

The dataset contains standardized RGB photographs of six tea-leaf conditions:

- Algal leaf spot
- Black blight
- Blister blight
- Gray blight
- Healthy
- Spider mites

The metadata describes:

- **1,791 standardized original images**
- **4,177 generated augmentation variants**
- **5,968 total entries** in the expanded set, including standardized-original copies
- **1,151 field originals**
- **640 controlled-environment originals**

All stored images in the official release are standardized to 1024 × 1024 pixels. Aspect-ratio-preserving resizing and white padding were used where required.

## Repository contents

The [`metadata`](metadata/) directory contains:

| File | Purpose |
|---|---|
| `README_metadata.txt` | Field definitions, preprocessing notes, and recommended experimental use |
| `SLTeaLeaf_class_count_summary.csv` | Class and acquisition-condition counts |
| `SLTeaLeaf_metadata_raw.csv` | Metadata for the 1,791 standardized originals |
| `SLTeaLeaf_metadata_augmented.csv` | Metadata for the 5,968-entry expanded set |
| `train_original_plus_augmented_baseid_split.csv` | Training manifest with original and augmented entries |
| `validation_original_only_baseid_split.csv` | Original-image validation manifest |
| `test_original_only_baseid_split.csv` | Original-image test manifest |

## Leakage-safe use

Generated variants are linked to their standardized source photograph through `base_image_id`. All entries sharing the same `base_image_id` should remain in the same data partition. Generated variants should normally be restricted to training, while validation and test evaluation should use original images.

The supplied split manifests follow this design. Researchers should also report acquisition conditions when combining field and controlled-environment images because these settings may differ in background, lighting, framing, and device characteristics.

## Images

This repository does **not** contain the 2.7 GB image collection. Download the images from the official Mendeley Data record and follow the repository-relative paths recorded in the metadata.

## Citation

Please cite the original dataset rather than this metadata mirror:

> Ganesalingam, V., Nasser, M., Osman Ibrahim, A., AbdElgader, E. S., A. Gabralla, L., Ravi, V. and Kandiah, P. (2026). *Sri Lankan Tea Leaf Dataset*. Version 5. Mendeley Data. <https://doi.org/10.17632/sjmy6k24d6.5>

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

## Licence and attribution

The original Mendeley Data release is distributed under the Creative Commons Attribution 4.0 International licence. See [`LICENSE.md`](LICENSE.md) for attribution and reuse information. Users remain responsible for reviewing the authoritative dataset record and complying with its current terms.

