# FOD-A Data Dictionary

This dictionary documents the original FOD-A fields and the variables derived by this project. It should be consulted when interpreting the exploratory analysis, preprocessing experiments and model evaluation.

## Important interpretation note

The FOD-A authors label lighting and weather as **qualitative capture conditions**. They do not publish numerical lux boundaries for bright, dim and dark, nor a numerical surface-moisture threshold for dry and wet. The project must therefore treat these values as publisher-supplied categories rather than calibrated physical measurements.

In particular, **dim** means the dataset authors' intermediate light-level category between bright and dark. It should not be rewritten as a specific lux range without additional measurement data.

## Original dataset fields

| Field | Location | Type | Values | Meaning |
|---|---|---|---|---|
| `File` | Environmental CSV | String | Six-digit filename such as `000123.jpg` | Identifies the image associated with the environmental labels. |
| `Weather` | Environmental CSV | Categorical integer | `0`, `1` | Runway-surface weather condition assigned during dataset creation. See the weather codes below. |
| `Light` | Environmental CSV | Categorical integer | `0`, `1`, `2` | Overall light-level category assigned during dataset creation. See the lighting codes below. |
| `filename` | Pascal VOC XML | String | JPEG filename | Name of the image described by the annotation file. |
| `width` | Pascal VOC XML | Integer | `300` in Version 2.1 | Image width in pixels. |
| `height` | Pascal VOC XML | Integer | `300` in Version 2.1 | Image height in pixels. |
| `depth` | Pascal VOC XML | Integer | `3` | Number of RGB colour channels. |
| `object/name` | Pascal VOC XML | Category | One of 31 source debris categories | Source identity assigned to an annotated FOD instance, such as `Bolt`, `Wire` or `Rock`. |
| `xmin`, `ymin` | Pascal VOC XML | Number | Pixel coordinates | Upper-left corner of the ground-truth bounding box. |
| `xmax`, `ymax` | Pascal VOC XML | Number | Pixel coordinates | Lower-right corner of the ground-truth bounding box. |

## Weather codes

| Code | Label | Interpretation used in this project | Dataset limitation |
|---:|---|---|---|
| `0` | Dry | The image was assigned to the dataset's dry runway-surface condition. | No numerical moisture threshold is supplied. |
| `1` | Wet | The image was assigned to the wet runway-surface condition, meaning surface moisture was present. The authors state that moisture remaining after snow clearance belongs here. | Wetness depth, rainfall intensity and reflectance were not measured. |

Snow is not a separate category because the dataset authors assume it is cleared promptly from the airport environment. This dataset therefore describes the runway-surface condition, not a complete meteorological report.

## Lighting codes

| Code | Label | Interpretation used in this project | Dataset limitation |
|---:|---|---|---|
| `0` | Bright | The author-assigned high-illumination capture condition. | No lux or exposure threshold is supplied. |
| `1` | Dim | The author-assigned intermediate illumination condition between bright and dark. | It is a qualitative label, not a calibrated low-light measurement. |
| `2` | Dark | The author-assigned lowest-illumination capture condition. | No lux or exposure threshold is supplied. |

These labels are suitable for condition-stratified evaluation, but conclusions should be phrased as performance on the **FOD-A labelled conditions**, not performance within universal physical illumination ranges.

## Leakage-resistant manifest fields

The project-generated `data/splits/final/manifest.csv` contains one row per image.

| Field | Type | Meaning |
|---|---|---|
| `image_id` | String | Six-digit identifier shared by the JPEG and XML files. Leading zeros are meaningful. |
| `split` | Category | Corrected partition: `train`, `validation` or `test`. |
| `sequence_id` | String | Identifier for adjacent, visually related frames that must remain in one partition. |
| `weather` | Categorical string | Original weather code, retained as `0` or `1`. |
| `light` | Categorical string | Original lighting code, retained as `0`, `1` or `2`. |
| `labels` | String | Pipe-separated source labels present in the image. |
| `object_count` | Integer | Number of annotated FOD instances in the image. |
| `dhash` | Hexadecimal string | 64-bit perceptual difference hash used to compare adjacent frames during sequence construction. It is not an object label or model input. |

## Exploratory-analysis variables

| Variable | Type | Definition |
|---|---|---|
| `weather_name` | Category | Human-readable mapping of the original weather code: Dry or Wet. |
| `light_name` | Category | Human-readable mapping of the original light code: Bright, Dim or Dark. |
| `mean_brightness` | Continuous | Mean grayscale pixel intensity on the 0–255 scale. This is calculated by the project and is not the publisher's labelling rule. |
| `contrast_std` | Continuous | Standard deviation of grayscale pixel intensity, used as a simple global-contrast indicator. |
| `box_width` | Continuous | `xmax - xmin`, in native-image pixels. |
| `box_height` | Continuous | `ymax - ymin`, in native-image pixels. |
| `box_area_pixels` | Continuous | Bounding-box width multiplied by bounding-box height. |
| `box_area_fraction` | Continuous | Bounding-box area divided by total image area. |
| `size` | Category | Project-derived COCO size band at native resolution: Small below `32²` pixels, Medium from `32²` to below `96²`, and Large from `96²` pixels. |

The COCO size bands describe bounding-box area, not the physical dimensions or real-world danger of the debris.

## Sources

- FOD-A local categorisation file: `ImageSets/Main/CategorizationData/category_information.txt`
- FOD-A environmental annotations: `ImageSets/Main/CategorizationData/FOD_categorization_annotations.csv`
- Munyer, T., Huang, P.-C., Huang, C. and Zhong, X. (2021), [*FOD-A: A Dataset for Foreign Object Debris in Airports*](https://arxiv.org/abs/2110.03072)
