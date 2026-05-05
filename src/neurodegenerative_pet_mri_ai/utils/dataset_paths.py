from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    data_config = config["data"]
    configured_root = Path(data_config["dataset_root"])
    candidate_roots = [configured_root]

    raw_candidate = Path("data/raw/ds007561")
    if configured_root != raw_candidate:
        candidate_roots.append(raw_candidate)

    for candidate in candidate_roots:
        if candidate.exists():
            return candidate

    return configured_root


def dataset_location_message(config: dict[str, Any]) -> str:
    configured_root = Path(config["data"]["dataset_root"])
    fallback_root = Path("data/raw/ds007561")
    return (
        "Dataset not found. Place OpenNeuro ds007561 under "
        f"'{configured_root}' or '{fallback_root}' so that 'participants.tsv' and "
        "subject folders such as 'sub-XX/ses-01/anat' and 'sub-XX/ses-01/pet' are present."
    )
