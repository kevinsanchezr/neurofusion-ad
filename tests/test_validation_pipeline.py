from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from neurodegenerative_pet_mri_ai.data.bids import BIDSDatasetScanner
from neurodegenerative_pet_mri_ai.data.dataset import build_dataloaders


def test_validation_pipeline_builds_one_batch(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds007561"
    _write_subject(dataset_root, "sub-01", "Control", offset=1.0)
    _write_subject(dataset_root, "sub-02", "MCI", offset=2.0)

    scanner = BIDSDatasetScanner(dataset_root=dataset_root, label_column="diagnosis")
    summary = scanner.inspect()
    assert summary["total_subjects"] == 2
    assert summary["valid_paired_samples"] == 2
    assert summary["label_distribution"] == {"Control": 1, "MCI": 1}

    config = {
        "experiment": {"seed": 42},
        "data": {
            "dataset_root": str(dataset_root),
            "participants_file": None,
            "session": "ses-01",
            "label_column": "diagnosis",
            "image_size": [16, 16, 16],
            "cache_rate": 0.0,
            "num_workers": 0,
        },
        "training": {
            "batch_size": 1,
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
        },
    }
    dataloaders = build_dataloaders(config)
    batch = next(iter(dataloaders["train"]))

    assert tuple(batch["mri"].shape) == (1, 1, 16, 16, 16)
    assert tuple(batch["pet"].shape) == (1, 1, 16, 16, 16)
    assert str(batch["mri"].dtype) == "torch.float32"
    assert str(batch["pet"].dtype) == "torch.float32"
    assert str(batch["label"].dtype) == "torch.int64"
    assert 0.0 <= float(batch["mri"].min()) <= 1.0
    assert 0.0 <= float(batch["mri"].max()) <= 1.0
    assert 0.0 <= float(batch["pet"].min()) <= 1.0
    assert 0.0 <= float(batch["pet"].max()) <= 1.0
    assert int(batch["label"][0]) in {0, 1}


def _write_subject(dataset_root: Path, subject_id: str, diagnosis: str, offset: float) -> None:
    session_root = dataset_root / subject_id / "ses-01"
    anat_dir = session_root / "anat"
    pet_dir = session_root / "pet"
    anat_dir.mkdir(parents=True, exist_ok=True)
    pet_dir.mkdir(parents=True, exist_ok=True)

    affine = np.eye(4)
    mri = np.linspace(0.0, 1.0, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8) + offset
    pet = np.linspace(0.5, 1.5, 8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8) + offset
    nib.save(nib.Nifti1Image(mri, affine), anat_dir / f"{subject_id}_ses-01_T1w.nii.gz")
    nib.save(nib.Nifti1Image(pet, affine), pet_dir / f"{subject_id}_ses-01_pet.nii.gz")

    participants_file = dataset_root / "participants.tsv"
    participants = []
    if participants_file.exists():
        participants = pd.read_csv(participants_file, sep="\t").to_dict(orient="records")
    participants.append({"participant_id": subject_id, "diagnosis": diagnosis})
    pd.DataFrame(participants).to_csv(participants_file, sep="\t", index=False)
