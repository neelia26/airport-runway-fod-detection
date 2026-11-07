# Airport Runway Foreign Object Debris Detection

This repository contains the coursework implementation for a machine-vision system that detects Foreign Object Debris (FOD) in airport runway and taxiway images using deep learning.

## Research question

> How reliably can a lightweight deep-learning model detect airport Foreign Object Debris of different sizes across runway lighting and weather conditions when evaluated using a sequence-aware, leakage-resistant data split?

## Intended system

The system will accept a runway image and return the location and confidence of detected debris. The initial implementation will treat all 31 source categories as a single `FOD` class so that the project focuses on reliable hazard localisation rather than fine-grained object naming.

The implementation will combine selected machine-vision techniques with a lightweight deep-learning object detector:

- Intensity and contrast analysis across bright, dim and dark images
- CLAHE as a controlled local-contrast preprocessing experiment
- Edge and small-object visibility analysis
- Thresholding and morphology as a conventional baseline
- Multiresolution analysis for small debris
- Lightweight YOLO-based object detection

## Dataset

The project uses **FOD-A Version 2.1 in Pascal VOC format**, published by the University of Nebraska at Omaha Aviation Institute.

- 33,793 RGB images at 300 x 300 pixels
- 33,793 Pascal VOC XML annotation files
- 34,472 annotated objects from 31 source categories
- Dry and wet pavement labels
- Bright, dim and dark lighting labels

Official sources:

- [FOD-A dataset repository](https://github.com/FOD-UNOmaha/FOD-data)
- [FOD-A publication](https://arxiv.org/abs/2110.03072)

The downloaded dataset is kept locally in `FOD-A Dataset/` and is intentionally excluded from Git.

## Dataset audit

The local audit confirmed that all images and XML annotations are readable and that all bounding boxes are valid. It also identified problems in the supplied partition files:

- Four image IDs occur in both training and testing
- Split files contain repeated entries
- Seventy images are absent from both supplied partitions
- Visually similar sequential frames cross the supplied partitions
- Debris category is associated with weather and lighting
- 9,057 objects occupy less than one percent of the image

The project will therefore create a sequence-aware split before training. Aggregate performance will be supplemented with object-size, lighting and weather analyses, with particular attention to missed detections.

## Repository status

The project is currently at the dataset validation and preprocessing stage. Model results will be added only after the leakage-resistant split and preprocessing pipeline have been completed.
