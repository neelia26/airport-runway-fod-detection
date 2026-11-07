# Split manifests

This directory contains train, validation and test manifests from the sequence-aware splitting pipeline.

Draft outputs are ignored. The verified final manifests are version controlled so that the experiments reported in the coursework can be reproduced exactly.

Each image appears once in `manifest.csv`, together with its sequence group, conditions and assigned partition. The three text files provide the image IDs expected by later conversion and training scripts. `split_summary.json` records the parameters and verification results.

## Grouping rule

Two neighbouring frames remain in the same sequence when:

1. their numerical image IDs are consecutive;
2. their weather and lighting codes match;
3. their annotated source-label sets match; and
4. their 64-bit difference-hash distance is no greater than 16.

The threshold was calibrated on the complete dataset. It retains 33,222 of the 33,676 adjacent pairs with matching metadata (98.65%). Sequence groups are assigned as indivisible units within each weather-light stratum using a fixed random seed of 42.

## Final split

- Training: 23,602 images
- Validation: 5,111 images
- Test: 5,080 images
- Total: 33,793 images
- Sequence groups crossing partitions: 0
- Duplicate or cross-partition image IDs: 0
