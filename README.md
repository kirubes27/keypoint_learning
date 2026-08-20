# keypoint_learning

This repository contains a minimal TDW dataset generator for producing ordered, fine‑step pose sequences of single rigid objects with aligned per-frame metadata, intended as training data for keypoint/representation-learning experiments based on temporal predictability and simple operators.

## Canonical research baseline

The curated baseline is the assisted Task-80 world-Z roll experiment documented in [docs/baselines/task80-assisted-roll/README.md](docs/baselines/task80-assisted-roll/README.md). Its exact recipe and artifact bindings are stored alongside that document.

This baseline is intentionally earlier than the OCR-ZNCC, descriptor-attachment, material-sliding, and wobble investigations. Those later research lines are not part of the canonical recipe.

Run the repository-local contract checks with:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The default checks do not require large datasets or checkpoints. The baseline document gives the environment variables for the bound external-artifact check.
