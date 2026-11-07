# Scripts

This directory contains short command-line entry points. Data parsing, preprocessing, modelling and evaluation logic should remain in the importable `src/fod_detection/` package so that it can be reused by scripts, notebooks and tests.

Current commands:

- `python scripts/build_leakage_resistant_split.py` rebuilds the frozen manifests.
- `python scripts/verify_split.py` independently checks completeness, uniqueness and sequence isolation.
