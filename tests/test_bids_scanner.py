from pathlib import Path

import pandas as pd

from neurodegenerative_pet_mri_ai.data.bids import BIDSDatasetScanner


def test_bids_scanner_discovers_paired_subjects(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ds007561"
    subject_dir = dataset_root / "sub-01" / "ses-01"
    (subject_dir / "anat").mkdir(parents=True)
    (subject_dir / "pet").mkdir(parents=True)
    (subject_dir / "anat" / "sub-01_ses-01_T1w.nii.gz").touch()
    (subject_dir / "pet" / "sub-01_ses-01_pet.nii.gz").touch()

    pd.DataFrame(
        [{"participant_id": "sub-01", "diagnosis": "Control"}]
    ).to_csv(dataset_root / "participants.tsv", sep="\t", index=False)

    scanner = BIDSDatasetScanner(dataset_root=dataset_root)
    records = scanner.scan()

    assert len(records) == 1
    assert records[0].label_index == 0
