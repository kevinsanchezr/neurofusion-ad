from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from neurodegenerative_pet_mri_ai.data.bids import BIDSDatasetScanner
from neurodegenerative_pet_mri_ai.data.preprocessing import _collapse_dynamic_pet


FIGURE_DPI = 360
OUTPUT_STEM = "cn_mci_ad_sample_comparison"
CLASS_ORDER = ("Control", "MCI", "AD")
CLASS_ALIASES = {
    "Control": {"control", "cn", "healthy", "hc"},
    "MCI": {"mci", "mild cognitive impairment"},
    "AD": {"ad", "alzheimer", "alzheimers", "alzheimer's disease"},
}
AXIAL_SLICE_FRACTION = 0.5


@dataclass(frozen=True)
class SubjectSample:
    subject_id: str
    label_name: str
    mri_slice: np.ndarray
    pet_slice: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a publication-style CN/MCI/AD MRI-PET dataset sample figure from ds007561."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/raw/ds007561",
        help="Path to the ds007561 dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures/dataset_samples",
        help="Directory where the PNG and SVG figure exports will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_matplotlib()

    samples = _load_selected_samples(dataset_root)
    crop_box = _shared_crop_box([sample.mri_slice for sample in samples.values()])
    pet_vmax = _shared_pet_scale([sample.pet_slice for sample in samples.values()])

    fig = _build_figure(samples, crop_box, pet_vmax)
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    svg_path = output_dir / f"{OUTPUT_STEM}.svg"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    for class_name in CLASS_ORDER:
        sample = samples[class_name]
        print(f"{class_name}\t{sample.subject_id}\t{sample.label_name}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def _load_selected_samples(dataset_root: Path) -> dict[str, SubjectSample]:
    scanner = BIDSDatasetScanner(dataset_root=dataset_root)
    records = scanner.scan()
    if not records:
        raise FileNotFoundError(f"No paired MRI/PET records found under {dataset_root}")

    selected = {}
    for class_name in CLASS_ORDER:
        selected_record = _select_record_for_class(records, class_name)
        selected[class_name] = _build_subject_sample(selected_record)
    return selected


def _select_record_for_class(records: list, class_name: str):
    aliases = CLASS_ALIASES[class_name]
    matching = [
        record
        for record in records
        if _normalize_label(record.label_name) in aliases
    ]
    if not matching:
        raise ValueError(f"No paired subject found for clinical group '{class_name}'.")
    return sorted(matching, key=lambda record: record.subject_id)[0]


def _build_subject_sample(record) -> SubjectSample:
    mri_img = nib.as_closest_canonical(nib.load(str(record.mri_path)))
    pet_img = nib.as_closest_canonical(nib.load(str(record.pet_path)))

    mri_volume = np.asarray(mri_img.get_fdata(dtype=np.float32), dtype=np.float32)
    pet_volume = np.asarray(pet_img.get_fdata(dtype=np.float32), dtype=np.float32)
    pet_volume = _collapse_dynamic_pet(pet_volume)
    pet_img_3d = nib.Nifti1Image(pet_volume, pet_img.affine, pet_img.header)
    pet_resampled = resample_from_to(pet_img_3d, mri_img, order=1)
    pet_volume_resampled = np.asarray(pet_resampled.get_fdata(dtype=np.float32), dtype=np.float32)

    slice_index = int(round((mri_volume.shape[2] - 1) * AXIAL_SLICE_FRACTION))
    mri_slice = np.rot90(mri_volume[:, :, slice_index])
    pet_slice = np.rot90(pet_volume_resampled[:, :, slice_index])

    return SubjectSample(
        subject_id=record.subject_id,
        label_name=record.label_name,
        mri_slice=mri_slice,
        pet_slice=pet_slice,
    )


def _build_figure(
    samples: dict[str, SubjectSample],
    crop_box: tuple[int, int, int, int],
    pet_vmax: float,
) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        len(CLASS_ORDER),
        figsize=(11.0, 8.6),
        dpi=FIGURE_DPI,
        constrained_layout=True,
    )
    row_labels = ("MRI T1w", "PET UCB-J", "MRI + PET")
    pet_mappable = None

    for col_index, class_name in enumerate(CLASS_ORDER):
        sample = samples[class_name]
        mri_crop = _crop_slice(_robust_normalize(sample.mri_slice), crop_box)
        pet_crop = _crop_slice(np.clip(sample.pet_slice, 0.0, pet_vmax), crop_box)
        pet_norm = np.clip(pet_crop / pet_vmax, 0.0, 1.0)

        column_title = f"{sample.subject_id}\n{class_name}"
        axes[0, col_index].set_title(column_title, pad=10)

        axes[0, col_index].imshow(mri_crop, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        pet_mappable = axes[1, col_index].imshow(
            pet_crop,
            cmap="magma",
            vmin=0.0,
            vmax=pet_vmax,
            interpolation="bilinear",
        )
        axes[2, col_index].imshow(mri_crop, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
        axes[2, col_index].imshow(
            pet_norm,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            alpha=np.clip((pet_norm - 0.12) / 0.88, 0.0, 0.72),
            interpolation="bilinear",
        )

    for row_index, row_label in enumerate(row_labels):
        axes[row_index, 0].set_ylabel(row_label, rotation=90, labelpad=18)
        for col_index in range(len(CLASS_ORDER)):
            ax = axes[row_index, col_index]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_facecolor("white")

    fig.suptitle("Representative ds007561 MRI and PET Samples", fontsize=16, y=1.01)
    fig.text(
        0.015,
        0.992,
        "Central axial slices with shared crop and intensity conventions across Control, MCI, and AD.",
        ha="left",
        va="top",
        fontsize=10,
        color="#475569",
    )
    cbar = fig.colorbar(pet_mappable, ax=axes[1:, :], fraction=0.018, pad=0.02)
    cbar.set_label("PET uptake (shared clipped scale)")
    return fig


def _normalize_label(label_name: str) -> str:
    normalized = label_name.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def _robust_normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.0])
    if np.isclose(high, low):
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _brain_crop_box(image_slice: np.ndarray) -> tuple[int, int, int, int]:
    normalized = _robust_normalize(image_slice)
    mask = normalized > 0.08
    if not np.any(mask):
        return (0, image_slice.shape[0], 0, image_slice.shape[1])
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    margin = 6
    row_start = max(int(rows[0]) - margin, 0)
    row_end = min(int(rows[-1]) + margin + 1, image_slice.shape[0])
    col_start = max(int(cols[0]) - margin, 0)
    col_end = min(int(cols[-1]) + margin + 1, image_slice.shape[1])
    return (row_start, row_end, col_start, col_end)


def _shared_crop_box(image_slices: list[np.ndarray]) -> tuple[int, int, int, int]:
    crop_boxes = [_brain_crop_box(image_slice) for image_slice in image_slices]
    row_start = min(box[0] for box in crop_boxes)
    row_end = max(box[1] for box in crop_boxes)
    col_start = min(box[2] for box in crop_boxes)
    col_end = max(box[3] for box in crop_boxes)
    return (row_start, row_end, col_start, col_end)


def _crop_slice(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    row_start, row_end, col_start, col_end = crop_box
    return image[row_start:row_end, col_start:col_end]


def _shared_pet_scale(pet_slices: list[np.ndarray]) -> float:
    positive_values = np.concatenate([image[image > 0] for image in pet_slices])
    if positive_values.size == 0:
        return 1.0
    scale = float(np.percentile(positive_values, 99.5))
    return scale if scale > 0 else 1.0


if __name__ == "__main__":
    main()
