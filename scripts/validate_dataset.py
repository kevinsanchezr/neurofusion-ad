from __future__ import annotations

import argparse

import torch

from neurodegenerative_pet_mri_ai.data.bids import BIDSDatasetScanner
from neurodegenerative_pet_mri_ai.data.dataset import build_dataloaders
from neurodegenerative_pet_mri_ai.utils import (
    dataset_location_message,
    load_config,
    resolve_dataset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local BIDS dataset wiring and dataloaders.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=1,
        help="Number of dataloader batches to inspect during the sanity check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = resolve_dataset_root(config)
    config["data"]["dataset_root"] = str(dataset_root)

    if not dataset_root.exists():
        print(dataset_location_message(config))
        return 1

    participants_file = dataset_root / "participants.tsv"
    if not participants_file.exists():
        print(
            f"Dataset root exists at '{dataset_root}', but 'participants.tsv' is missing. "
            f"{dataset_location_message(config)}"
        )
        return 1

    scanner = BIDSDatasetScanner(
        dataset_root=dataset_root,
        session=config["data"]["session"],
        label_column=config["data"]["label_column"],
        participants_file=participants_file,
    )
    summary = scanner.inspect()
    print_summary(summary)

    if summary["valid_paired_samples"] == 0:
        print("No valid paired MRI/PET samples with mapped labels were found. Skipping dataloader check.")
        return 1

    run_dataloader_sanity_check(config, max_batches=args.max_batches)
    return 0


def print_summary(summary: dict[str, object]) -> None:
    print("Dataset Summary")
    print(f"- Dataset root: {summary['dataset_root']}")
    print(f"- Participants file: {summary['participants_file']}")
    print(f"- Total subjects discovered: {summary['total_subjects']}")
    print(f"- Valid paired samples: {summary['valid_paired_samples']}")
    print(f"- Missing MRI only: {summary['missing_mri_only']}")
    print(f"- Missing PET only: {summary['missing_pet_only']}")
    print(f"- Missing both modalities: {summary['missing_both_modalities']}")
    print(f"- Missing or unmapped labels: {summary['missing_or_unmapped_labels']}")
    print("- Label distribution:")

    label_distribution = summary["label_distribution"]
    if label_distribution:
        for label_name, count in label_distribution.items():
            print(f"  - {label_name}: {count}")
    else:
        print("  - None")

    print("- Example paired records:")
    examples = summary["examples"]
    if examples:
        for example in examples:
            print(
                "  - "
                f"{example['subject_id']} | label={example['label']} | "
                f"MRI={example['mri_path']} | PET={example['pet_path']}"
            )
    else:
        print("  - None")


def run_dataloader_sanity_check(config: dict[str, object], max_batches: int) -> None:
    config["data"]["num_workers"] = 0
    config["training"]["batch_size"] = 1
    dataloaders = build_dataloaders(config)
    loader = dataloaders["train"]
    print("Dataloader Sanity Check")

    for batch_index, batch in enumerate(loader):
        mri = batch["mri"]
        pet = batch["pet"]
        label = batch["label"]

        print(
            f"- Batch {batch_index + 1}: "
            f"mri_shape={tuple(mri.shape)}, pet_shape={tuple(pet.shape)}, labels={label.tolist()}"
        )
        print(
            f"  dtype: mri={mri.dtype}, pet={pet.dtype}, label={label.dtype}"
        )
        print(
            f"  range: mri=({float(mri.min()):.4f}, {float(mri.max()):.4f}), "
            f"pet=({float(pet.min()):.4f}, {float(pet.max()):.4f})"
        )
        print(
            f"  channels: mri={mri.shape[1]}, pet={pet.shape[1]}, "
            f"tensor_device={'cuda' if torch.cuda.is_available() else 'cpu-ready'}"
        )

        if batch_index + 1 >= max_batches:
            break


if __name__ == "__main__":
    raise SystemExit(main())
