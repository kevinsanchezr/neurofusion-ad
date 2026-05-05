from __future__ import annotations

import random
from collections import Counter
from typing import Any

import torch
from monai.data import CacheDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split

from neurodegenerative_pet_mri_ai.data.bids import BIDSRecord, BIDSDatasetScanner
from neurodegenerative_pet_mri_ai.data.preprocessing import (
    build_preprocessing,
    registration_placeholder,
)


class MultimodalNeuroimagingDataset:
    """Factory around BIDS scanning and MONAI dataset generation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = _resolve_data_config(config)
        self.records = self._scan_records()

    def manifest(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for record in self.records:
            mri_path, pet_path = registration_placeholder(record.mri_path, record.pet_path)
            sample = {
                "subject_id": record.subject_id,
                "session_id": record.session_id,
                "mri": str(mri_path),
                "pet": str(pet_path),
                "label": record.label_index,
                "label_name": record.label_name,
            }
            samples.append(sample)
        return samples

    def _scan_records(self) -> list[BIDSRecord]:
        scanner = BIDSDatasetScanner(
            dataset_root=self.config["dataset_root"],
            session=self.config.get("session", "ses-01"),
            label_column=self.config.get("label_column", "diagnosis"),
            participants_file=self.config.get("participants_file"),
        )
        return scanner.scan()


def build_dataloaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_config = _resolve_data_config(config)
    training_config = config["training"]
    splits = create_data_splits(config)
    transforms = build_preprocessing(tuple(data_config["image_size"]))
    dataset_cls = CacheDataset if data_config.get("cache_rate", 0.0) > 0 else Dataset
    dataset_kwargs = {
        "transform": transforms,
    }
    if dataset_cls is CacheDataset:
        dataset_kwargs["cache_rate"] = data_config["cache_rate"]

    train_ds = dataset_cls(data=splits["train"], **dataset_kwargs)
    val_ds = dataset_cls(data=splits["val"], **dataset_kwargs)
    test_ds = dataset_cls(data=splits["test"], **dataset_kwargs)

    common_loader_args = {
        "batch_size": training_config["batch_size"],
        "num_workers": data_config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        "train": DataLoader(train_ds, shuffle=True, **common_loader_args),
        "val": DataLoader(val_ds, shuffle=False, **common_loader_args),
        "test": DataLoader(test_ds, shuffle=False, **common_loader_args),
    }


def create_data_splits(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data_config = _resolve_data_config(config)
    training_config = config["training"]
    experiment_config = config["experiment"]

    dataset_builder = MultimodalNeuroimagingDataset(data_config)
    manifest = dataset_builder.manifest()
    if not manifest:
        raise RuntimeError("No paired MRI/PET samples were discovered in the configured dataset.")

    train_ratio = training_config["train_split"]
    val_ratio = training_config["val_split"]
    test_ratio = training_config["test_split"]
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("Train/val/test splits must sum to 1.0.")

    if len(manifest) < 6:
        return {"train": manifest, "val": manifest, "test": manifest}

    label_counts = Counter(sample["label"] for sample in manifest)
    singleton_samples = [sample for sample in manifest if label_counts[sample["label"]] < 2]
    splittable_samples = [sample for sample in manifest if label_counts[sample["label"]] >= 2]

    if not splittable_samples:
        return {"train": manifest, "val": manifest, "test": manifest}

    splittable_samples = sorted(splittable_samples, key=lambda sample: sample["subject_id"])
    labels = [sample["label"] for sample in splittable_samples]
    train_samples, temp_samples, _train_labels, temp_labels = train_test_split(
        splittable_samples,
        labels,
        test_size=(1.0 - train_ratio),
        stratify=_stratify_labels_or_none(labels),
        random_state=experiment_config["seed"],
    )

    if not temp_samples:
        train_samples = sorted(train_samples + singleton_samples, key=lambda sample: sample["subject_id"])
        return {"train": train_samples, "val": train_samples, "test": train_samples}

    val_fraction = val_ratio / (val_ratio + test_ratio)
    if len(temp_samples) < 2:
        val_samples = temp_samples
        test_samples = temp_samples
    else:
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(1.0 - val_fraction),
            stratify=_stratify_labels_or_none(temp_labels),
            random_state=experiment_config["seed"],
        )

    train_samples = sorted(train_samples + singleton_samples, key=lambda sample: sample["subject_id"])
    val_samples = sorted(val_samples, key=lambda sample: sample["subject_id"])
    test_samples = sorted(test_samples, key=lambda sample: sample["subject_id"])
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def build_split_manifest_payload(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for split_name, split_samples in splits.items():
        payload[split_name] = {
            "num_samples": len(split_samples),
            "subject_ids": [sample["subject_id"] for sample in split_samples],
            "label_distribution": dict(
                sorted(Counter(sample["label_name"] for sample in split_samples).items())
            ),
        }
    return payload


def _resolve_data_config(config: dict[str, Any]) -> dict[str, Any]:
    if "data" in config:
        return config["data"]
    return config


def _stratify_labels_or_none(labels: list[int]) -> list[int] | None:
    if len(set(labels)) <= 1:
        return None
    counts = Counter(labels)
    if min(counts.values()) < 2:
        return None
    return labels
