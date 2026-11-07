# Data directory

Raw FOD-A files are deliberately stored outside this tracked directory:

```text
FOD-A Dataset/
└── FODPascalVOCFormat-V.2.1/
    └── VOC2007/
        ├── Annotations/
        ├── ImageSets/
        └── JPEGImages/
```

The local `FOD-A Dataset/DATASET_SOURCE.txt` records the download source and checksum. The dataset itself is excluded from Git because of its size and its separate distribution terms.

Project-generated data will use the following locations:

- `data/interim/`: temporary metadata and validation outputs
- `data/splits/`: sequence-aware train, validation and test manifests
- `data/yolo/`: converted single-class YOLO images and labels
- `data/processed/`: any other model-ready derived data

Only documentation belongs in Git until a generated artefact is deliberately selected as small, stable evidence.
