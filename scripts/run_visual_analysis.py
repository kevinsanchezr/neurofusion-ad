from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
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
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)
    for label_name in LABEL_NAMES:
        mask = np.asarray([name == label_name for name in labels], dtype=bool)
        if not mask.any():
            continue
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=120,
            alpha=0.85,
            color=LABEL_COLORS[label_name],
            edgecolors="white",
            linewidths=0.8,
            label=label_name,
        )

    ax.set_title(title, fontsize=18, weight="bold", pad=16)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=11,
        color="#475569",
    )
    ax.set_xlabel(axis_labels[0], fontsize=12)
    ax.set_ylabel(axis_labels[1], fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Clinical Label", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(svg_output_path, bbox_inches="tight")
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
    splits = create_data_splits(config)
    selected_samples = _select_gradcam_samples(splits, max_subjects=max_subjects)
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

        mri_heatmap = _upsample_heatmap(mri_cam.generate(logits, target_class).heatmap, mri.shape[2:])
        pet_heatmap = _upsample_heatmap(pet_cam.generate(logits, target_class).heatmap, pet.shape[2:])

        subject_id = batch["subject_id"][0]
        true_label = batch["label_name"][0]
        pred_label = LABEL_NAMES[target_class]
        figure_path = output_dir / f"{subject_id}_multimodal_gradcam.png"
        svg_path = output_dir / f"{subject_id}_multimodal_gradcam.svg"
        _plot_gradcam_subject(
            subject_id=subject_id,
            mri_volume=mri.detach().cpu()[0, 0].numpy(),
            pet_volume=pet.detach().cpu()[0, 0].numpy(),
            mri_heatmap=mri_heatmap,
            pet_heatmap=pet_heatmap,
            true_label=true_label,
            pred_label=pred_label,
            output_path=figure_path,
            svg_output_path=svg_path,
        )
        atlas_summary["subjects"].append(
            {
                "subject_id": subject_id,
                "true_label": true_label,
                "predicted_label": pred_label,
                "status": "atlas_mapping_skipped",
            }
        )

    write_json(output_dir / "atlas_region_summary.json", atlas_summary)


def _select_gradcam_samples(
    splits: dict[str, list[dict[str, Any]]], max_subjects: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split_name in ("test", "val", "train"):
        for sample in splits[split_name]:
            selected.append({**sample, "split": split_name})
            if len(selected) >= max_subjects:
                return selected
    return selected


def _upsample_heatmap(heatmap: torch.Tensor, spatial_size: torch.Size) -> np.ndarray:
    upsampled = F.interpolate(heatmap, size=tuple(spatial_size), mode="trilinear", align_corners=False)
    return upsampled.detach().cpu()[0, 0].numpy()


def _plot_gradcam_subject(
    subject_id: str,
    mri_volume: np.ndarray,
    pet_volume: np.ndarray,
    mri_heatmap: np.ndarray,
    pet_heatmap: np.ndarray,
    true_label: str,
    pred_label: str,
    output_path: Path,
    svg_output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=180)
    fig.suptitle(
        f"{subject_id} Multimodal Grad-CAM\nTrue: {true_label} | Predicted: {pred_label} | Exploratory only",
        fontsize=16,
        weight="bold",
    )

    mri_slices = _extract_orthogonal_slices(mri_volume, mri_heatmap)
    pet_slices = _extract_orthogonal_slices(pet_volume, pet_heatmap)
    rows = [("MRI", mri_slices), ("PET", pet_slices)]
    overlay_mappable = None

    for row_index, (row_name, slices) in enumerate(rows):
        for col_index, (view_name, image_slice, heatmap_slice) in enumerate(slices):
            ax = axes[row_index, col_index]
            ax.imshow(image_slice, cmap="gray")
            overlay_mappable = ax.imshow(heatmap_slice, cmap="inferno", alpha=0.45, vmin=0.0, vmax=1.0)
            ax.set_title(f"{row_name} {view_name}", fontsize=12)
            ax.axis("off")

    cbar = fig.colorbar(overlay_mappable, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Grad-CAM activation", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.96, 0.94))
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(svg_output_path, bbox_inches="tight")
    plt.close(fig)


def _extract_orthogonal_slices(
    volume: np.ndarray, heatmap: np.ndarray
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    max_index = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    sagittal = (
        "Sagittal",
        np.rot90(volume[max_index[0], :, :]),
        np.rot90(heatmap[max_index[0], :, :]),
    )
    coronal = (
        "Coronal",
        np.rot90(volume[:, max_index[1], :]),
        np.rot90(heatmap[:, max_index[1], :]),
    )
    axial = (
        "Axial",
        np.rot90(volume[:, :, max_index[2]]),
        np.rot90(heatmap[:, :, max_index[2]]),
    )
    return [sagittal, coronal, axial]


def _resolve_module(model: torch.nn.Module, module_path: str) -> torch.nn.Module:
    module: Any = model
    for part in module_path.split("."):
        module = getattr(module, part)
    return module


if __name__ == "__main__":
    main()
