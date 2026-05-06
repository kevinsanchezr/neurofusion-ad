from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from neurodegenerative_pet_mri_ai.utils.io import ensure_dir, write_json


EXPERIMENTS = {
    "multimodal": {
        "experiment_dir": "reports/experiments/ds007561_multimodal_50ep",
        "embedding_path": "reports/experiments/embeddings_50ep/multimodal_50ep_multimodal_embeddings.npz",
        "latent_dir": "reports/figures/latent_space_50ep",
        "evaluation_dir": "reports/figures/evaluation_50ep",
        "explainability_dir": "reports/figures/explainability_50ep",
    },
    "mri_only": {
        "experiment_dir": "reports/experiments/ds007561_mri_only_50ep",
        "embedding_path": "reports/experiments/embeddings_50ep/mri_only_50ep_mri_only_embeddings.npz",
        "latent_dir": "reports/figures/latent_space_50ep_mri_only",
        "evaluation_dir": "reports/figures/evaluation_50ep_mri_only",
        "explainability_dir": None,
    },
    "pet_only": {
        "experiment_dir": "reports/experiments/ds007561_pet_only_50ep",
        "embedding_path": "reports/experiments/embeddings_50ep/pet_only_50ep_pet_only_embeddings.npz",
        "latent_dir": "reports/figures/latent_space_50ep_pet_only",
        "evaluation_dir": "reports/figures/evaluation_50ep_pet_only",
        "explainability_dir": None,
    },
}

MODEL_LABELS = {
    "multimodal": "Multimodal",
    "mri_only": "MRI-only",
    "pet_only": "PET-only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exploratory 50-epoch runs across modalities.")
    parser.add_argument(
        "--output-json",
        default="reports/experiments/ds007561_50ep_model_comparison.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--output-markdown",
        default="reports/ds007561_50ep_model_comparison.md",
        help="Markdown output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = {
        modality: _load_run_payload(
            modality=modality,
            experiment_dir=Path(spec["experiment_dir"]),
            embedding_path=Path(spec["embedding_path"]),
            latent_dir=Path(spec["latent_dir"]),
            evaluation_dir=Path(spec["evaluation_dir"]),
            explainability_dir=Path(spec["explainability_dir"]) if spec["explainability_dir"] else None,
        )
        for modality, spec in EXPERIMENTS.items()
    }
    comparison = _build_comparison_payload(payloads)
    write_json(Path(args.output_json), comparison)
    _write_markdown_report(Path(args.output_markdown), comparison)


def _load_run_payload(
    modality: str,
    experiment_dir: Path,
    embedding_path: Path,
    latent_dir: Path,
    evaluation_dir: Path,
    explainability_dir: Path | None,
) -> dict[str, Any]:
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((experiment_dir / "history.json").read_text(encoding="utf-8"))["history"]
    splits = json.loads((experiment_dir / "subject_splits.json").read_text(encoding="utf-8"))
    best_epoch = min(history, key=lambda row: row["val_loss"])
    final_epoch = history[-1]

    return {
        "modality": modality,
        "model_label": MODEL_LABELS[modality],
        "experiment_dir": str(experiment_dir),
        "metrics": metrics,
        "history": history,
        "splits": splits,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "embedding_summary": _summarize_embeddings(embedding_path),
        "evaluation_outputs": _collect_files(evaluation_dir),
        "latent_space_outputs": _collect_files(latent_dir),
        "explainability_outputs": _collect_files(explainability_dir) if explainability_dir else None,
        "gradcam_selection": _load_optional_json(explainability_dir / "figure_selection_manifest.json") if explainability_dir else None,
    }


def _summarize_embeddings(embedding_path: Path) -> dict[str, Any]:
    if not embedding_path.exists():
        return {"status": "missing", "embedding_path": str(embedding_path)}

    payload = np.load(embedding_path, allow_pickle=True)
    embeddings = payload["embeddings"].astype(np.float64)
    labels = payload["label_names"].astype(str)

    pca = PCA(n_components=min(2, embeddings.shape[0], embeddings.shape[1]))
    coords = pca.fit_transform(embeddings)
    variance = pca.explained_variance_ratio_.tolist()

    centroids: dict[str, list[float]] = {}
    for label_name in sorted(set(labels.tolist())):
        mask = labels == label_name
        centroids[label_name] = coords[mask].mean(axis=0).tolist()

    centroid_distances: dict[str, float] = {}
    label_order = list(centroids.keys())
    for index, left_label in enumerate(label_order):
        for right_label in label_order[index + 1 :]:
            left = np.asarray(centroids[left_label], dtype=np.float64)
            right = np.asarray(centroids[right_label], dtype=np.float64)
            centroid_distances[f"{left_label}_vs_{right_label}"] = float(np.linalg.norm(left - right))

    return {
        "status": "completed",
        "embedding_path": str(embedding_path),
        "num_subjects": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "explained_variance_ratio": variance,
        "centroids": centroids,
        "centroid_distances": centroid_distances,
    }


def _collect_files(directory: Path | None) -> dict[str, Any] | None:
    if directory is None:
        return None
    if not directory.exists():
        return {"status": "missing", "files": []}
    return {
        "status": "completed",
        "files": sorted(str(path) for path in directory.iterdir() if path.is_file()),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_qualitative_takeaways(payloads: dict[str, dict[str, Any]]) -> list[str]:
    multimodal_embed = payloads["multimodal"]["embedding_summary"]
    mri_embed = payloads["mri_only"]["embedding_summary"]
    pet_embed = payloads["pet_only"]["embedding_summary"]

    def _distance(summary: dict[str, Any], key: str) -> float | None:
        value = summary.get("centroid_distances", {}).get(key)
        return float(value) if value is not None else None

    lines = [
        "All three 50-epoch models memorize the training split while validation and test remain flat, so optimization success does not translate into measurable generalization on this cohort.",
    ]
    control_mci = {
        "Multimodal": _distance(multimodal_embed, "Control_vs_MCI"),
        "MRI-only": _distance(mri_embed, "Control_vs_MCI"),
        "PET-only": _distance(pet_embed, "Control_vs_MCI"),
    }
    available = {name: value for name, value in control_mci.items() if value is not None}
    if available:
        best_name = max(available, key=available.get)
        lines.append(
            f"In PCA space, {best_name} shows the largest Control-vs-MCI centroid separation among the saved embeddings, but this does not convert into higher validation/test balanced accuracy."
        )
    lines.append(
        "Multimodal remains the only model with Grad-CAM overlays in the current project state, so any explainability advantage is qualitative and limited to cross-modality visualization rather than better held-out metrics."
    )
    lines.append(
        "PET-only converges more slowly and ends with the weakest validation loss, which suggests harder optimization under the same split despite eventually memorizing the training set."
    )
    return lines


def _build_comparison_payload(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference_split = payloads["multimodal"]["splits"]
    split_consistency = {
        modality: payload["splits"] == reference_split
        for modality, payload in payloads.items()
    }

    summary_rows = []
    overfitting_notes = []
    for modality in ["multimodal", "mri_only", "pet_only"]:
        payload = payloads[modality]
        metrics = payload["metrics"]
        final_epoch = payload["final_epoch"]
        summary_rows.append(
            {
                "modality": modality,
                "model_label": payload["model_label"],
                "completed_epochs": metrics["completed_epochs"],
                "best_val_epoch": payload["best_epoch"]["epoch"],
                "final_train_loss": final_epoch["train_loss"],
                "final_val_loss": final_epoch["val_loss"],
                "train_balanced_accuracy": final_epoch["train_balanced_accuracy"],
                "val_balanced_accuracy": final_epoch["val_balanced_accuracy"],
                "test_balanced_accuracy": metrics["test_metrics"]["balanced_accuracy"],
                "train_macro_f1": final_epoch["train_macro_f1"],
                "val_macro_f1": final_epoch["val_macro_f1"],
                "test_macro_f1": metrics["test_metrics"]["macro_f1"],
                "train_confusion_matrix": metrics["latest_train_metrics"]["confusion_matrix"],
                "val_confusion_matrix": metrics["latest_val_metrics"]["confusion_matrix"],
                "test_confusion_matrix": metrics["test_metrics"]["confusion_matrix"],
            }
        )

        train_ba = final_epoch["train_balanced_accuracy"]
        val_ba = final_epoch["val_balanced_accuracy"]
        test_ba = metrics["test_metrics"]["balanced_accuracy"]
        note = (
            f"{payload['model_label']}: train BA={train_ba:.4f}, val BA={val_ba:.4f}, "
            f"test BA={test_ba:.4f}"
        )
        if train_ba >= 0.95 and val_ba <= 0.5:
            note += " -> strong memorization with flat validation."
        elif train_ba - val_ba >= 0.25:
            note += " -> notable train/validation gap."
        else:
            note += " -> no large generalization gain."
        overfitting_notes.append(note)

    qualitative_takeaways = _build_qualitative_takeaways(payloads)

    return {
        "dataset": "ds007561",
        "comparison_type": "exploratory_50_epoch_modalities",
        "split_consistency": split_consistency,
        "shared_split_manifest": reference_split,
        "summary_table": summary_rows,
        "embeddings": {
            modality: payloads[modality]["embedding_summary"]
            for modality in ["multimodal", "mri_only", "pet_only"]
        },
        "artifacts": {
            modality: {
                "experiment_dir": payloads[modality]["experiment_dir"],
                "evaluation_outputs": payloads[modality]["evaluation_outputs"],
                "latent_space_outputs": payloads[modality]["latent_space_outputs"],
                "explainability_outputs": payloads[modality]["explainability_outputs"],
                "gradcam_selection": payloads[modality]["gradcam_selection"],
            }
            for modality in ["multimodal", "mri_only", "pet_only"]
        },
        "overfitting_analysis": overfitting_notes,
        "qualitative_takeaways": qualitative_takeaways,
        "notes": [
            "Exploratory only. ds007561 is very small and strongly imbalanced.",
            "There is only one AD subject, and under the fixed split it remains in training.",
            "Validation and test splits contain Control and MCI only, so AD generalization is not estimable there.",
            "Grad-CAM qualitative comparison is only available for the multimodal run in the current project state.",
        ],
    }


def _write_markdown_report(output_path: Path, comparison: dict[str, Any]) -> None:
    rows = comparison["summary_table"]
    artifacts = comparison["artifacts"]
    embeddings = comparison["embeddings"]
    lines = [
        "# ds007561 50-Epoch Modality Comparison",
        "",
        "This report is exploratory only. It compares optimization behavior and generalization limits across multimodal, MRI-only, and PET-only 50-epoch runs on the same fixed split.",
        "",
        "## Metrics Summary",
        "",
        "| Model | Epochs | Best Val Epoch | Train Loss | Val Loss | Train BA | Val BA | Test BA | Train Macro F1 | Val Macro F1 | Test Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_label']} | {row['completed_epochs']} | {row['best_val_epoch']} | "
            f"{row['final_train_loss']:.4f} | {row['final_val_loss']:.4f} | "
            f"{row['train_balanced_accuracy']:.4f} | {row['val_balanced_accuracy']:.4f} | {row['test_balanced_accuracy']:.4f} | "
            f"{row['train_macro_f1']:.4f} | {row['val_macro_f1']:.4f} | {row['test_macro_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Overfitting Analysis",
            "",
            *[f"- {note}" for note in comparison["overfitting_analysis"]],
            "",
            "## Evaluation Figures",
            "",
        ]
    )
    for modality in ["multimodal", "mri_only", "pet_only"]:
        evaluation_files = artifacts[modality]["evaluation_outputs"]["files"] if artifacts[modality]["evaluation_outputs"] else []
        loss_link = _find_file(evaluation_files, "loss_curves.png")
        score_link = _find_file(evaluation_files, "balanced_accuracy_macro_f1.png")
        confusion_link = _find_file(evaluation_files, "confusion_matrices.png")
        links: list[str] = []
        if loss_link:
            links.append(f"[loss_curves.png]({Path(loss_link).resolve()})")
        if score_link:
            links.append(f"[balanced_accuracy_macro_f1.png]({Path(score_link).resolve()})")
        if confusion_link:
            links.append(f"[confusion_matrices.png]({Path(confusion_link).resolve()})")
        if links:
            lines.append(f"- {MODEL_LABELS[modality]}: {'; '.join(links)}")

    lines.extend(
        [
            "",
            "## Confusion Matrices",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row['model_label']} train: `{row['train_confusion_matrix']}`")
        lines.append(f"- {row['model_label']} validation: `{row['val_confusion_matrix']}`")
        lines.append(f"- {row['model_label']} test: `{row['test_confusion_matrix']}`")

    lines.extend(["", "## Embeddings", ""])
    for modality in ["multimodal", "mri_only", "pet_only"]:
        embed = embeddings[modality]
        lines.append(
            f"- {MODEL_LABELS[modality]}: PCA variance `{_fmt_list(embed.get('explained_variance_ratio', []))}`, "
            f"centroid distances `{embed.get('centroid_distances', {})}`"
        )
        latent_files = artifacts[modality]["latent_space_outputs"]["files"] if artifacts[modality]["latent_space_outputs"] else []
        pca_link = _find_file(latent_files, f"{modality}_pca.png")
        tsne_link = _find_file(latent_files, f"{modality}_tsne.png")
        links: list[str] = []
        if pca_link:
            links.append(f"[{Path(pca_link).name}]({Path(pca_link).resolve()})")
        if tsne_link:
            links.append(f"[{Path(tsne_link).name}]({Path(tsne_link).resolve()})")
        if links:
            lines.append(f"  Figures: {' and '.join(links)}")

    lines.extend(["", "## Qualitative Takeaways", ""])
    lines.extend(f"- {note}" for note in comparison["qualitative_takeaways"])

    lines.extend(
        [
            "",
            "## Explainability",
            "",
            "- Grad-CAM qualitative comparison is available only for the multimodal 50-epoch run.",
            "- The current pipeline uses MRI-derived masking, percentile thresholding, Gaussian smoothing, consistent cropping, and aligned MRI/PET overlays.",
            "- These overlays remain exploratory and should not be treated as clinical anatomical evidence.",
        ]
    )
    multimodal_gradcam = artifacts["multimodal"]["explainability_outputs"]
    if multimodal_gradcam:
        for path_str in multimodal_gradcam["files"]:
            path = Path(path_str)
            if path.suffix.lower() in {".png", ".svg"} and "gradcam" in path.name:
                lines.append(f"- [${path.name}]({path.resolve()})".replace("$", ""))

    lines.extend(
        [
            "",
            "## Notes",
            "",
            *[f"- {note}" for note in comparison["notes"]],
        ]
    )

    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_list(values: list[float]) -> str:
    return ", ".join(f"{value:.4f}" for value in values)


def _find_file(files: list[str], filename: str) -> str | None:
    for path_str in files:
        if Path(path_str).name == filename:
            return path_str
    return None


if __name__ == "__main__":
    main()
