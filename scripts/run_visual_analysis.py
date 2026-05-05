from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from neurodegenerative_pet_mri_ai.data.dataset import create_data_splits
from neurodegenerative_pet_mri_ai.data.preprocessing import build_preprocessing
from neurodegenerative_pet_mri_ai.interpretability.gradcam import GradCAM3D
from neurodegenerative_pet_mri_ai.models.factory import build_model
from neurodegenerative_pet_mri_ai.representation.embeddings import extract_embeddings
from neurodegenerative_pet_mri_ai.utils.config import load_config
from neurodegenerative_pet_mri_ai.utils.dataset_paths import resolve_dataset_root
from neurodegenerative_pet_mri_ai.utils.io import ensure_dir, write_json
from neurodegenerative_pet_mri_ai.utils.seed import set_seed


LABEL_NAMES = ["Control", "MCI", "AD"]
LABEL_COLORS = {
    "Control": "#2563eb",
    "MCI": "#d97706",
    "AD": "#dc2626",
}
FIGURE_DPI = 320
LATENT_FIGSIZE = (8.8, 6.8)
GRADCAM_FIGSIZE = (11.5, 7.4)
GRADCAM_SLICE_FRACTIONS = {
    "Sagittal": 0.50,
    "Coronal": 0.52,
    "Axial": 0.58,
}
EXPERIMENTS = {
    "multimodal": {
        "config": "configs/smoke_multimodal.yaml",
        "model_name": "multimodal_fusion",
        "checkpoint": "reports/experiments/ds007561_multimodal_smoke/best_model.pt",
        "embedding_file": "reports/experiments/embeddings/ds007561_multimodal_embeddings.npz",
        "embedding_manifest": "reports/experiments/embeddings/ds007561_multimodal_embeddings.json",
    },
    "mri_only": {
        "config": "configs/smoke_mri_only.yaml",
        "model_name": "mri_only",
        "checkpoint": "reports/experiments/ds007561_mri_only_smoke/best_model.pt",
        "embedding_file": "reports/experiments/embeddings/ds007561_mri_only_embeddings.npz",
        "embedding_manifest": "reports/experiments/embeddings/ds007561_mri_only_embeddings.json",
    },
    "pet_only": {
        "config": "configs/smoke_pet_only.yaml",
        "model_name": "pet_only",
        "checkpoint": "reports/experiments/ds007561_pet_only_smoke/best_model.pt",
        "embedding_file": "reports/experiments/embeddings/ds007561_pet_only_embeddings.npz",
        "embedding_manifest": "reports/experiments/embeddings/ds007561_pet_only_embeddings.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory embedding and explainability outputs.")
    parser.add_argument("--device", default="auto", help="Device override: auto, cuda, or cpu.")
    parser.add_argument("--max-gradcam-subjects", type=int, default=3, help="Max multimodal subjects for Grad-CAM.")
    parser.add_argument("--atlas-image", default=None, help="Optional local atlas image path for future region mapping.")
    parser.add_argument("--atlas-labels", default=None, help="Optional local atlas labels JSON/TSV/TXT path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    _configure_matplotlib()
    embedding_results: dict[str, dict[str, Any]] = {}

    for modality, spec in EXPERIMENTS.items():
        config = load_config(spec["config"])
        set_seed(config["experiment"]["seed"])
        config["data"]["dataset_root"] = str(resolve_dataset_root(config))
        result = _extract_and_save_embeddings(
            modality=modality,
            config=config,
            checkpoint_path=Path(spec["checkpoint"]),
            embedding_output=Path(spec["embedding_file"]),
            manifest_output=Path(spec["embedding_manifest"]),
            device=device,
        )
        embedding_results[modality] = result

    _generate_latent_space_figures(embedding_results)
    _generate_multimodal_gradcam(
        embedding_results["multimodal"]["config"],
        checkpoint_path=Path(EXPERIMENTS["multimodal"]["checkpoint"]),
        device=device,
        max_subjects=args.max_gradcam_subjects,
        atlas_image=args.atlas_image,
        atlas_labels=args.atlas_labels,
    )


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams.update(
        {
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _extract_and_save_embeddings(
    modality: str,
    config: dict[str, Any],
    checkpoint_path: Path,
    embedding_output: Path,
    manifest_output: Path,
    device: torch.device,
) -> dict[str, Any]:
    model = build_model(
        model_name=config["training"]["model_name"],
        embedding_dim=config["representation"]["embedding_dim"],
        num_classes=config["training"]["num_classes"],
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    combined_samples = _combine_split_samples(create_data_splits(config))
    dataloader = _build_inference_dataloader(combined_samples, config)

    embeddings: list[np.ndarray] = []
    subject_ids: list[str] = []
    split_names: list[str] = []
    label_names: list[str] = []
    label_indices: list[int] = []

    for batch in dataloader:
        with torch.no_grad():
            features = extract_embeddings(model, batch, device).detach().cpu().numpy()
        embeddings.append(features[0])
        subject_ids.append(batch["subject_id"][0])
        split_names.append(batch["split"][0])
        label_names.append(batch["label_name"][0])
        label_indices.append(int(batch["label"][0].item()))

    embedding_matrix = np.stack(embeddings, axis=0)
    ensure_dir(embedding_output.parent)
    np.savez_compressed(
        embedding_output,
        embeddings=embedding_matrix.astype(np.float32),
        subject_ids=np.asarray(subject_ids),
        split_names=np.asarray(split_names),
        label_names=np.asarray(label_names),
        label_indices=np.asarray(label_indices, dtype=np.int64),
        modality=np.asarray([modality] * len(subject_ids)),
        model_name=np.asarray([config["training"]["model_name"]] * len(subject_ids)),
    )

    manifest = {
        "dataset": Path(config["data"]["dataset_root"]).name,
        "modality": modality,
        "model_name": config["training"]["model_name"],
        "checkpoint_path": str(checkpoint_path),
        "embedding_file": str(embedding_output),
        "num_subjects": len(subject_ids),
        "embedding_dim": int(embedding_matrix.shape[1]),
        "device": str(device),
        "subjects": [
            {
                "subject_id": subject_id,
                "split": split_name,
                "label_name": label_name,
                "label_index": label_index,
            }
            for subject_id, split_name, label_name, label_index in zip(
                subject_ids, split_names, label_names, label_indices, strict=True
            )
        ],
    }
    write_json(manifest_output, manifest)
    return {
        "config": config,
        "embeddings": embedding_matrix,
        "subject_ids": subject_ids,
        "split_names": split_names,
        "label_names": label_names,
        "label_indices": label_indices,
        "manifest": manifest,
    }


def _combine_split_samples(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    seen_subject_ids: set[str] = set()
    for split_name in ("train", "val", "test"):
        for sample in splits[split_name]:
            subject_id = str(sample["subject_id"])
            if subject_id in seen_subject_ids:
                continue
            combined.append({**sample, "split": split_name})
            seen_subject_ids.add(subject_id)
    return combined


def _build_inference_dataloader(samples: list[dict[str, Any]], config: dict[str, Any]) -> DataLoader:
    transform = build_preprocessing(tuple(config["data"]["image_size"]))
    dataset = Dataset(data=samples, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def _generate_latent_space_figures(embedding_results: dict[str, dict[str, Any]]) -> None:
    output_dir = ensure_dir("reports/figures/latent_space")
    comparison_manifest: dict[str, Any] = {"figures": []}

    for modality, payload in embedding_results.items():
        embeddings = payload["embeddings"]
        labels = payload["label_names"]
        pca = PCA(n_components=2, random_state=42)
        pca_coords = pca.fit_transform(embeddings)
        pca_path = output_dir / f"{modality}_pca.png"
        pca_svg_path = output_dir / f"{modality}_pca.svg"
        _plot_latent_projection(
            coordinates=pca_coords,
            labels=labels,
            title=f"{modality.replace('_', ' ').title()} Latent Space (PCA)",
            subtitle="Exploratory only - tiny imbalanced cohort, not for clinical interpretation",
            output_path=pca_path,
            svg_output_path=pca_svg_path,
            axis_labels=(
                f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)",
                f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)",
            ),
        )
        comparison_manifest["figures"].append(str(pca_path))

        if embeddings.shape[0] >= 6:
            perplexity = max(2, min(5, embeddings.shape[0] // 3))
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            )
            tsne_coords = tsne.fit_transform(embeddings)
            tsne_path = output_dir / f"{modality}_tsne.png"
            tsne_svg_path = output_dir / f"{modality}_tsne.svg"
            _plot_latent_projection(
                coordinates=tsne_coords,
                labels=labels,
                title=f"{modality.replace('_', ' ').title()} Latent Space (t-SNE)",
                subtitle="Exploratory only - local neighborhood view on a very small cohort",
                output_path=tsne_path,
                svg_output_path=tsne_svg_path,
                axis_labels=("t-SNE 1", "t-SNE 2"),
            )
            comparison_manifest["figures"].append(str(tsne_path))

    write_json(output_dir / "latent_space_manifest.json", comparison_manifest)


def _plot_latent_projection(
    coordinates: np.ndarray,
    labels: list[str],
    title: str,
    subtitle: str,
    output_path: Path,
    svg_output_path: Path,
    axis_labels: tuple[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=LATENT_FIGSIZE, dpi=FIGURE_DPI)
    coords = np.asarray(coordinates, dtype=float)
    for label_name in LABEL_NAMES:
        mask = np.asarray([name == label_name for name in labels], dtype=bool)
        if not mask.any():
            continue
        class_coords = coords[mask]
        ax.scatter(
            class_coords[:, 0],
            class_coords[:, 1],
            s=130,
            alpha=0.92,
            color=LABEL_COLORS[label_name],
            edgecolors="#ffffff",
            linewidths=1.0,
            label=label_name,
            zorder=4,
        )
        _draw_class_boundary(ax, class_coords, LABEL_COLORS[label_name])

    ax.set_title(title, fontsize=17, pad=18)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=10.5,
        color="#475569",
    )
    ax.set_xlabel(axis_labels[0], fontsize=12)
    ax.set_ylabel(axis_labels[1], fontsize=12)
    ax.grid(True, alpha=0.12, linewidth=0.8)
    ax.set_facecolor("#fcfcfd")
    ax.legend(title="Clinical Label", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _generate_multimodal_gradcam(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    max_subjects: int,
    atlas_image: str | None,
    atlas_labels: str | None,
) -> None:
    output_dir = ensure_dir("reports/figures/explainability")
    figure_manifest: dict[str, Any] = {"selected_subjects": []}
    splits = create_data_splits(config)
    selected_samples = _select_gradcam_samples(
        config=config,
        splits=splits,
        checkpoint_path=checkpoint_path,
        device=device,
        max_subjects=max_subjects,
    )
    dataloader = _build_inference_dataloader(selected_samples, config)

    model = build_model(
        model_name=config["training"]["model_name"],
        embedding_dim=config["representation"]["embedding_dim"],
        num_classes=config["training"]["num_classes"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    mri_cam = GradCAM3D(model, _resolve_module(model, "mri_encoder.backbone.features.transition3.conv"))
    pet_cam = GradCAM3D(model, _resolve_module(model, "pet_encoder.backbone.features.transition3.conv"))
    atlas_summary: dict[str, Any] = {
        "status": "skipped",
        "reason": "Atlas-guided mapping requires local atlas files and registered subject images in atlas space.",
        "atlas_image": atlas_image,
        "atlas_labels": atlas_labels,
        "subjects": [],
    }

    for batch in dataloader:
        mri = batch["mri"].to(device)
        pet = batch["pet"].to(device)
        logits = model(mri, pet)
        target_class = int(logits.argmax(dim=1).item())
        probabilities = torch.softmax(logits, dim=1)
        confidence = float(probabilities[0, target_class].item())

        mri_volume_np = mri.detach().cpu()[0, 0].numpy()
        pet_volume_np = pet.detach().cpu()[0, 0].numpy()
        brain_soft_mask = _build_brain_soft_mask(mri_volume_np)
        mri_heatmap = _upsample_heatmap(
            _refine_heatmap(
                mri_cam.generate(logits, target_class).heatmap,
                brain_soft_mask=brain_soft_mask,
                spatial_size=mri.shape[2:],
            ),
            mri.shape[2:],
        )
        pet_heatmap = _upsample_heatmap(
            _refine_heatmap(
                pet_cam.generate(logits, target_class).heatmap,
                brain_soft_mask=brain_soft_mask,
                spatial_size=pet.shape[2:],
            ),
            pet.shape[2:],
        )

        subject_id = batch["subject_id"][0]
        true_label = batch["label_name"][0]
        pred_label = LABEL_NAMES[target_class]
        figure_path = output_dir / f"{subject_id}_multimodal_gradcam.png"
        svg_path = output_dir / f"{subject_id}_multimodal_gradcam.svg"
        _plot_gradcam_subject(
            subject_id=subject_id,
            mri_volume=mri_volume_np,
            pet_volume=pet_volume_np,
            mri_heatmap=mri_heatmap,
            pet_heatmap=pet_heatmap,
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence,
            output_path=figure_path,
            svg_output_path=svg_path,
        )
        figure_manifest["selected_subjects"].append(
            {
                "subject_id": subject_id,
                "true_label": true_label,
                "predicted_label": pred_label,
                "confidence": confidence,
                "selection_reason": "correct_prediction" if pred_label == true_label else "fallback_prediction",
                "figure_png": str(figure_path),
                "figure_svg": str(svg_path),
            }
        )
        atlas_summary["subjects"].append(
            {
                "subject_id": subject_id,
                "true_label": true_label,
                "predicted_label": pred_label,
                "status": "atlas_mapping_skipped",
            }
        )

    write_json(output_dir / "figure_selection_manifest.json", figure_manifest)
    write_json(output_dir / "atlas_region_summary.json", atlas_summary)


def _select_gradcam_samples(
    config: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    checkpoint_path: Path,
    device: torch.device,
    max_subjects: int,
) -> list[dict[str, Any]]:
    candidates = _rank_multimodal_samples(config, splits, checkpoint_path, device)
    correct = [sample for sample in candidates if sample["predicted_label"] == sample["label_name"]]
    fallback = [sample for sample in candidates if sample["predicted_label"] != sample["label_name"]]
    chosen = (correct + fallback)[:max_subjects]
    return [{**sample["sample"], "split": sample["split"]} for sample in chosen]


def _rank_multimodal_samples(
    config: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    checkpoint_path: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    model = build_model(
        model_name=config["training"]["model_name"],
        embedding_dim=config["representation"]["embedding_dim"],
        num_classes=config["training"]["num_classes"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    ranked: list[dict[str, Any]] = []
    for split_name in ("test", "val", "train"):
        dataset_samples = [{**sample, "split": split_name} for sample in splits[split_name]]
        dataloader = _build_inference_dataloader(dataset_samples, config)
        for sample, batch in zip(dataset_samples, dataloader, strict=True):
            with torch.no_grad():
                logits = model(batch["mri"].to(device), batch["pet"].to(device))
                probabilities = torch.softmax(logits, dim=1)
            predicted_index = int(probabilities.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_index].item())
            ranked.append(
                {
                    "sample": sample,
                    "split": split_name,
                    "label_name": batch["label_name"][0],
                    "predicted_label": LABEL_NAMES[predicted_index],
                    "confidence": confidence,
                }
            )
    ranked.sort(
        key=lambda item: (
            item["predicted_label"] != item["label_name"],
            -item["confidence"],
            item["split"] != "test",
            item["split"] != "val",
            item["sample"]["subject_id"],
        )
    )
    return ranked


def _upsample_heatmap(heatmap: torch.Tensor, spatial_size: torch.Size) -> np.ndarray:
    upsampled = F.interpolate(heatmap, size=tuple(spatial_size), mode="trilinear", align_corners=False)
    return upsampled.detach().cpu()[0, 0].numpy()


def _refine_heatmap(
    heatmap: torch.Tensor,
    brain_soft_mask: np.ndarray,
    spatial_size: torch.Size,
) -> torch.Tensor:
    pooled = F.avg_pool3d(heatmap, kernel_size=3, stride=1, padding=1)
    upsampled = F.interpolate(pooled, size=tuple(spatial_size), mode="trilinear", align_corners=False)
    heatmap_np = upsampled.detach().cpu()[0, 0].numpy()
    heatmap_np = np.maximum(heatmap_np, 0.0)
    heatmap_np = gaussian_filter(heatmap_np, sigma=1.2)
    heatmap_np *= brain_soft_mask
    in_brain = heatmap_np[brain_soft_mask > 0.05]
    if in_brain.size > 0:
        threshold = float(np.percentile(in_brain, 83.0))
        heatmap_np = np.where(heatmap_np >= threshold, heatmap_np, 0.0)
    heatmap_np = gaussian_filter(heatmap_np, sigma=0.9)
    heatmap_np *= brain_soft_mask
    max_value = float(heatmap_np.max())
    if max_value > 0:
        heatmap_np = heatmap_np / max_value
    refined = torch.from_numpy(heatmap_np).to(device=heatmap.device, dtype=heatmap.dtype)
    return refined.unsqueeze(0).unsqueeze(0)


def _plot_gradcam_subject(
    subject_id: str,
    mri_volume: np.ndarray,
    pet_volume: np.ndarray,
    mri_heatmap: np.ndarray,
    pet_heatmap: np.ndarray,
    true_label: str,
    pred_label: str,
    confidence: float,
    output_path: Path,
    svg_output_path: Path,
) -> None:
    mri_slices = _extract_orthogonal_slices(mri_volume, mri_heatmap)
    pet_slices = _extract_orthogonal_slices(pet_volume, pet_heatmap)
    fig, axes = plt.subplots(2, 3, figsize=GRADCAM_FIGSIZE, dpi=FIGURE_DPI)
    fig.suptitle(
        f"{subject_id}  |  True: {true_label}  |  Pred: {pred_label}  |  p={confidence:.2f}",
        fontsize=15,
        y=0.98,
    )

    rows = [("MRI", mri_slices), ("PET", pet_slices)]
    overlay_mappable = None
    crop_boxes = [
        _brain_crop_box(image_slice)
        for _view_name, image_slice, _heatmap_slice in mri_slices
    ]

    for row_index, (row_name, slices) in enumerate(rows):
        for col_index, (view_name, image_slice, heatmap_slice) in enumerate(slices):
            ax = axes[row_index, col_index]
            image_crop, heatmap_crop = _crop_paired_slice(image_slice, heatmap_slice, crop_boxes[col_index])
            normalized_image = _robust_normalize(image_crop)
            cleaned_heatmap = _threshold_slice_heatmap(heatmap_crop, normalized_image)
            ax.imshow(normalized_image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="bilinear")
            overlay_mappable = ax.imshow(
                cleaned_heatmap,
                cmap="inferno",
                alpha=np.where(cleaned_heatmap > 0, 0.60, 0.0),
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            ax.set_title(f"{row_name} {view_name}", fontsize=11, pad=6)
            ax.axis("off")

    cbar = fig.colorbar(overlay_mappable, ax=axes, fraction=0.022, pad=0.015)
    cbar.set_label("Normalized Grad-CAM activation", fontsize=10)
    fig.subplots_adjust(left=0.03, right=0.93, bottom=0.04, top=0.91, wspace=0.04, hspace=0.10)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _extract_orthogonal_slices(
    volume: np.ndarray, heatmap: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    sagittal_idx = _fixed_slice_index(volume, "Sagittal")
    coronal_idx = _fixed_slice_index(volume, "Coronal")
    axial_idx = _fixed_slice_index(volume, "Axial")
    sagittal = (
        "Sagittal",
        np.rot90(volume[sagittal_idx, :, :]),
        np.rot90(heatmap[sagittal_idx, :, :]),
    )
    coronal = (
        "Coronal",
        np.rot90(volume[:, coronal_idx, :]),
        np.rot90(heatmap[:, coronal_idx, :]),
    )
    axial = (
        "Axial",
        np.rot90(volume[:, :, axial_idx]),
        np.rot90(heatmap[:, :, axial_idx]),
    )
    return [sagittal, coronal, axial]


def _fixed_slice_index(volume: np.ndarray, view_name: str) -> int:
    axis = {"Sagittal": 0, "Coronal": 1, "Axial": 2}[view_name]
    size = volume.shape[axis]
    fraction = GRADCAM_SLICE_FRACTIONS[view_name]
    return max(0, min(size - 1, int(round((size - 1) * fraction))))


def _robust_normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, [1.0, 99.0])
    if math.isclose(high, low):
        return np.zeros_like(image)
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return normalized


def _threshold_slice_heatmap(heatmap: np.ndarray, brain_image: np.ndarray) -> np.ndarray:
    normalized = heatmap.astype(np.float32).copy()
    if float(normalized.max()) > 0:
        normalized = normalized / float(normalized.max())
    normalized = gaussian_filter(normalized, sigma=0.65)
    soft_mask = np.clip((brain_image - 0.05) / 0.30, 0.0, 1.0)
    normalized *= soft_mask
    positive = normalized[soft_mask > 0.05]
    threshold = float(np.percentile(positive, 80.0)) if positive.size > 0 else 1.0
    cleaned = np.where(normalized >= threshold, normalized, 0.0)
    cleaned = gaussian_filter(cleaned, sigma=0.55)
    cleaned *= soft_mask
    max_value = float(cleaned.max())
    if max_value > 0:
        cleaned = cleaned / max_value
    return cleaned


def _crop_paired_slice(
    image_slice: np.ndarray,
    heatmap_slice: np.ndarray,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_box is None:
        crop_box = _brain_crop_box(image_slice)
    row_start, row_end, col_start, col_end = crop_box
    return (
        image_slice[row_start:row_end, col_start:col_end],
        heatmap_slice[row_start:row_end, col_start:col_end],
    )


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


def _build_brain_soft_mask(mri_volume: np.ndarray) -> np.ndarray:
    normalized = _robust_normalize(mri_volume)
    soft_mask = np.clip((normalized - 0.04) / 0.28, 0.0, 1.0)
    soft_mask = gaussian_filter(soft_mask, sigma=1.1)
    return np.clip(soft_mask, 0.0, 1.0)


def _draw_class_boundary(ax: plt.Axes, class_coords: np.ndarray, color: str) -> None:
    if class_coords.shape[0] < 3:
        return
    center = class_coords.mean(axis=0)
    covariance = np.cov(class_coords.T)
    if covariance.shape != (2, 2) or np.linalg.det(covariance) <= 0:
        return
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2.6 * np.sqrt(np.maximum(eigenvalues, 1e-8))
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.08,
        zorder=2,
    )
    ax.add_patch(ellipse)


def _resolve_module(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    module: Any = model
    for part in module_path.split("."):
        module = getattr(module, part)
    return module


if __name__ == "__main__":
    main()
