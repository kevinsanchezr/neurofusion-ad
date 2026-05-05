from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityRangePercentilesd,
)


def validate_nifti(path: str | Path) -> tuple[int, ...]:
    image = nib.load(str(path))
    return image.shape


def build_preprocessing(image_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["mri", "pet"], image_only=False),
            Lambdad(keys=["pet"], func=_collapse_dynamic_pet),
            EnsureChannelFirstd(keys=["mri", "pet"], channel_dim="no_channel"),
            ScaleIntensityRangePercentilesd(
                keys=["mri", "pet"],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resized(keys=["mri", "pet"], spatial_size=image_size, mode=("trilinear", "trilinear")),
            EnsureTyped(keys=["mri", "pet", "label"]),
        ]
    )


def registration_placeholder(mri_path: str | Path, pet_path: str | Path) -> tuple[Path, Path]:
    """Hook for future MRI/PET registration integration."""
    return Path(mri_path), Path(pet_path)


def _collapse_dynamic_pet(image: np.ndarray) -> np.ndarray:
    """Reduce 4D PET data to a single 3D volume for baseline validation."""
    if image.ndim == 4:
        return image.mean(axis=-1)
    return image
