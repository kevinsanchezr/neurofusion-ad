from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LABEL_MAP = {
    "control": 0,
    "cn": 0,
    "healthy": 0,
    "mci": 1,
    "mild cognitive impairment": 1,
    "alzheimer": 2,
    "alzheimers": 2,
    "ad": 2,
}


@dataclass(frozen=True)
class BIDSRecord:
    subject_id: str
    session_id: str
    mri_path: Path
    pet_path: Path
    label_name: str
    label_index: int


class BIDSDatasetScanner:
    """Scan a BIDS dataset and resolve paired MRI/PET samples."""

    def __init__(
        self,
        dataset_root: str | Path,
        session: str = "ses-01",
        label_column: str = "diagnosis",
        participants_file: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.session = session
        self.label_column = label_column
        self.participants_file = (
            Path(participants_file) if participants_file else self.dataset_root / "participants.tsv"
        )

    def inspect(self) -> dict[str, object]:
        participants = self._load_participants()
        subject_details: list[dict[str, object]] = []
        missing_mri = 0
        missing_pet = 0
        missing_both = 0
        missing_labels = 0
        valid_paired = 0
        label_distribution: dict[str, int] = {}
        paired_examples: list[dict[str, str]] = []

        for subject_dir in sorted(self.dataset_root.glob("sub-*")):
            if not subject_dir.is_dir():
                continue

            session_dir = subject_dir / self.session
            anat_dir = session_dir / "anat"
            pet_dir = session_dir / "pet"
            mri_path = self._select_first(anat_dir.glob("*T1w.nii.gz")) if anat_dir.exists() else None
            pet_path = self._select_first(pet_dir.glob("*.nii.gz")) if pet_dir.exists() else None
            label_name, label_index = self._resolve_label(subject_dir.name, participants)

            if mri_path is None and pet_path is None:
                missing_both += 1
            elif mri_path is None:
                missing_mri += 1
            elif pet_path is None:
                missing_pet += 1

            if label_index is None:
                missing_labels += 1

            is_valid = mri_path is not None and pet_path is not None and label_index is not None
            if is_valid:
                valid_paired += 1
                label_distribution[label_name] = label_distribution.get(label_name, 0) + 1
                if len(paired_examples) < 3:
                    paired_examples.append(
                        {
                            "subject_id": subject_dir.name,
                            "mri_path": str(mri_path),
                            "pet_path": str(pet_path),
                            "label": label_name,
                        }
                    )

            subject_details.append(
                {
                    "subject_id": subject_dir.name,
                    "has_mri": mri_path is not None,
                    "has_pet": pet_path is not None,
                    "has_label": label_index is not None,
                    "label_name": label_name,
                    "is_valid_pair": is_valid,
                }
            )

        return {
            "dataset_root": str(self.dataset_root),
            "participants_file": str(self.participants_file),
            "total_subjects": len(subject_details),
            "valid_paired_samples": valid_paired,
            "missing_mri_only": missing_mri,
            "missing_pet_only": missing_pet,
            "missing_both_modalities": missing_both,
            "missing_or_unmapped_labels": missing_labels,
            "label_distribution": dict(sorted(label_distribution.items())),
            "examples": paired_examples,
            "subjects": subject_details,
        }

    def scan(self) -> list[BIDSRecord]:
        participants = self._load_participants()
        records: list[BIDSRecord] = []

        for subject_dir in sorted(self.dataset_root.glob("sub-*")):
            if not subject_dir.is_dir():
                continue

            session_dir = subject_dir / self.session
            anat_dir = session_dir / "anat"
            pet_dir = session_dir / "pet"
            if not anat_dir.exists() or not pet_dir.exists():
                continue

            mri_path = self._select_first(anat_dir.glob("*T1w.nii.gz"))
            pet_path = self._select_first(pet_dir.glob("*.nii.gz"))
            if mri_path is None or pet_path is None:
                continue

            subject_id = subject_dir.name
            label_name, label_index = self._resolve_label(subject_id, participants)
            if label_index is None:
                continue

            records.append(
                BIDSRecord(
                    subject_id=subject_id,
                    session_id=self.session,
                    mri_path=mri_path,
                    pet_path=pet_path,
                    label_name=label_name,
                    label_index=label_index,
                )
            )

        return records

    def validate(self) -> dict[str, int]:
        records = self.scan()
        return {
            "dataset_root_exists": int(self.dataset_root.exists()),
            "participants_file_exists": int(self.participants_file.exists()),
            "paired_subjects": len(records),
        }

    def _load_participants(self) -> pd.DataFrame:
        if not self.participants_file.exists():
            raise FileNotFoundError(f"Missing participants file: {self.participants_file}")

        participants = pd.read_csv(self.participants_file, sep="\t")
        if "participant_id" not in participants.columns:
            raise ValueError("participants.tsv must contain a 'participant_id' column.")
        if self.label_column not in participants.columns:
            raise ValueError(
                f"participants.tsv must contain the configured label column: {self.label_column}"
            )
        return participants

    def _resolve_label(
        self, subject_id: str, participants: pd.DataFrame
    ) -> tuple[str, int | None]:
        row = participants.loc[participants["participant_id"] == subject_id]
        if row.empty:
            return "unknown", None

        raw_label = str(row.iloc[0][self.label_column]).strip()
        normalized = raw_label.lower().replace("_", " ").replace("-", " ")
        normalized = " ".join(normalized.split())
        label_index = LABEL_MAP.get(normalized)
        return raw_label, label_index

    @staticmethod
    def _select_first(paths: Iterable[Path]) -> Path | None:
        return next(iter(sorted(paths)), None)
