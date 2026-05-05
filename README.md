# NeuroFusion AD

Professional research scaffold for multimodal neuroimaging analysis focused on Alzheimer's disease pattern detection from MRI and PET.

This repository is designed as a public-facing baseline for medical AI engineers and researchers who want a clean, reproducible starting point for BIDS-compatible multimodal pipelines.

## Goals

- Provide a generic, BIDS-compatible data pipeline without hardcoded subject paths.
- Support paired MRI T1-weighted and PET workflows linked to clinical labels from `participants.tsv`.
- Offer clean module boundaries for preprocessing, modeling, training, evaluation, representation learning, and interpretability.
- Keep the first version lightweight and extensible rather than shipping a heavy training stack.

## Repository Layout

```text
.
├── configs/
├── data/
├── models/
├── notebooks/
├── reports/
├── scripts/
├── src/
│   └── neurodegenerative_pet_mri_ai/
└── tests/
```

## Expected Dataset Layout

The project assumes a BIDS dataset similar to:

```text
ds007561/
├── participants.tsv
├── dataset_description.json
├── sub-01/
│   └── ses-01/
│       ├── anat/
│       │   └── sub-01_ses-01_T1w.nii.gz
│       └── pet/
│           └── sub-01_ses-01_pet.nii.gz
```

The loader scans the dataset dynamically, validates the presence of both modalities, and joins records to participant-level labels.

## Quick Start

1. Create an environment and install dependencies:

```bash
pip install -e .
```

2. Copy and adjust the base config:

```bash
cp configs/base.yaml configs/local.yaml
```

3. Run a dry startup of the training pipeline:

```bash
python scripts/train.py --config configs/base.yaml
```

## Validation Workflow

Use the lightweight validation phase before any training:

1. Install the package in editable mode:

```bash
pip install -e .
```

2. Verify package imports:

```bash
python -c "import neurodegenerative_pet_mri_ai; print(neurodegenerative_pet_mri_ai.__version__)"
```

3. Place the dataset at either `data/ds007561` or `data/raw/ds007561`.

4. Run the dry-run validator:

```bash
python scripts/validate_dataset.py --config configs/base.yaml
```

The validator:

- loads `configs/base.yaml`
- resolves `data/ds007561` or `data/raw/ds007561`
- scans the BIDS hierarchy and reads `participants.tsv`
- reports total subjects, valid paired samples, missing modalities, label distribution, and example paths
- runs a minimal dataloader sanity check on one batch only

If the dataset is missing, the script exits clearly without downloading data.

## Dataset Reference

This project is designed to work with OpenNeuro `ds007561`.

Citation:

Barbara Bendlin, Sterling Johnson, and Bradley Christian (2026). *UCB-J PET and T1W Images for Synapse Project Cohort*. OpenNeuro. Dataset. DOI: `10.18112/openneuro.ds007561.v1.0.0`

The dataset itself is not included in this repository. Large imaging files and raw BIDS data should remain local and outside version control.

## Current Scope

Implemented in this scaffold:

- BIDS dataset scanning and participant label resolution
- Sample manifest construction for paired MRI/PET subjects
- MONAI-compatible preprocessing builders
- Unimodal and multimodal baseline model skeletons
- Training and evaluation entry points
- Feature extraction hooks for latent-space studies
- Grad-CAM style interpretability interface for 3D backbones

Deferred for future iterations:

- Full experiment tracking
- Robust affine/voxel-space registration pipeline
- Cross-validation orchestration
- Statistical analysis notebooks
- Full Grad-CAM visualization export workflow

## Reproducibility

- Central YAML configuration
- Seed control utilities for Python, NumPy, and PyTorch
- Explicit sample manifest generation
- Deterministic-ready training hooks

## Notes

- `participants.tsv` label normalization currently maps common variants for `Control`, `MCI`, and `Alzheimer`.
- The code is designed to be reusable across BIDS datasets with the same modality semantics, not only `ds007561`.
- Raw dataset files under `data/raw/` are intentionally ignored so the public repository stays lightweight and useful.
- `ds007561` currently contains only one `AD` subject, so this repository should be treated as a technical proof-of-concept and exploratory multimodal pipeline, not as a clinically reliable Alzheimer's classifier.
