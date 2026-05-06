from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from neurodegenerative_pet_mri_ai.utils.io import ensure_dir, write_json


LABEL_NAMES = ["Control", "MCI", "AD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare smoke-test and 50-epoch multimodal experiment outputs.")
    parser.add_argument(
        "--smoke-experiment-dir",
        default="reports/experiments/ds007561_multimodal_smoke",
        help="Smoke-test experiment directory.",
    )
    parser.add_argument(
        "--long-experiment-dir",
        default="reports/experiments/ds007561_multimodal_50ep",
        help="50-epoch experiment directory.",
    )
    parser.add_argument(
        "--smoke-embedding",
        default="reports/experiments/embeddings/ds007561_multimodal_embeddings.npz",
        help="Embedding NPZ generated from the smoke-test checkpoint.",
    )
    parser.add_argument(
        "--long-embedding",
        default="reports/experiments/embeddings_50ep/multimodal_50ep_multimodal_embeddings.npz",
        help="Embedding NPZ generated from the 50-epoch checkpoint.",
    )
    parser.add_argument(
        "--smoke-latent-dir",
        default="reports/figures/latent_space",
        help="Latent-space figure directory for the smoke-test run.",
    )
    parser.add_argument(
        "--long-latent-dir",
        default="reports/figures/latent_space_50ep",
        help="Latent-space figure directory for the 50-epoch run.",
    )
    parser.add_argument(
        "--smoke-explainability-dir",
        default="reports/figures/explainability",
        help="Explainability figure directory for the smoke-test run.",
    )
    parser.add_argument(
        "--long-explainability-dir",
        default="reports/figures/explainability_50ep",
        help="Explainability figure directory for the 50-epoch run.",
    )
    parser.add_argument(
        "--json-output",
        default="reports/experiments/ds007561_multimodal_smoke_vs_50ep.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/ds007561_multimodal_smoke_vs_50ep.md",
        help="Markdown report output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smoke = _load_run(Path(args.smoke_experiment_dir))
    long_run = _load_run(Path(args.long_experiment_dir))

    comparison = {
        "dataset": "ds007561",
        "comparison": "multimodal_smoke_vs_50ep",
        "smoke": _build_run_summary(
            run_name="Smoke test",
            run=smoke,
            embedding_path=Path(args.smoke_embedding),
            latent_dir=Path(args.smoke_latent_dir),
            explainability_dir=Path(args.smoke_explainability_dir),
        ),
        "long_run": _build_run_summary(
            run_name="50-epoch exploratory run",
            run=long_run,
            embedding_path=Path(args.long_embedding),
            latent_dir=Path(args.long_latent_dir),
            explainability_dir=Path(args.long_explainability_dir),
        ),
    }
    comparison["delta"] = _build_delta_summary(comparison["smoke"], comparison["long_run"])

    write_json(Path(args.json_output), comparison)
    _write_markdown_report(Path(args.markdown_output), comparison)


def _load_run(experiment_dir: Path) -> dict[str, Any]:
    history = json.loads((experiment_dir / "history.json").read_text(encoding="utf-8"))["history"]
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    subject_splits = json.loads((experiment_dir / "subject_splits.json").read_text(encoding="utf-8"))
    return {
        "experiment_dir": str(experiment_dir),
        "history": history,
        "metrics": metrics,
        "subject_splits": subject_splits,
    }


def _build_run_summary(
    run_name: str,
    run: dict[str, Any],
    embedding_path: Path,
    latent_dir: Path,
    explainability_dir: Path,
) -> dict[str, Any]:
    history = run["history"]
    metrics = run["metrics"]
    best_epoch = min(history, key=lambda row: row["val_loss"])
    final_epoch = history[-1]

    return {
        "run_name": run_name,
        "experiment_dir": run["experiment_dir"],
        "device": metrics.get("device", "unknown"),
        "completed_epochs": metrics.get("completed_epochs", len(history)),
        "best_val_epoch": best_epoch["epoch"],
        "best_val_loss": best_epoch["val_loss"],
        "final_epoch": final_epoch["epoch"],
        "final_train_loss": final_epoch["train_loss"],
        "final_val_loss": final_epoch["val_loss"],
        "final_train_balanced_accuracy": final_epoch["train_balanced_accuracy"],
        "final_val_balanced_accuracy": final_epoch["val_balanced_accuracy"],
        "final_train_macro_f1": final_epoch["train_macro_f1"],
        "final_val_macro_f1": final_epoch["val_macro_f1"],
        "test_balanced_accuracy": metrics["test_metrics"]["balanced_accuracy"],
        "test_macro_f1": metrics["test_metrics"]["macro_f1"],
        "train_confusion_matrix": metrics["latest_train_metrics"]["confusion_matrix"],
        "val_confusion_matrix": metrics["latest_val_metrics"]["confusion_matrix"],
        "test_confusion_matrix": metrics["test_metrics"]["confusion_matrix"],
        "train_per_class": metrics["latest_train_metrics"]["per_class"],
        "val_per_class": metrics["latest_val_metrics"]["per_class"],
        "test_per_class": metrics["test_metrics"]["per_class"],
        "subject_splits": run["subject_splits"],
        "embedding_summary": _summarize_embeddings(embedding_path),
        "latent_space_outputs": _collect_figure_paths(latent_dir),
        "explainability_outputs": _collect_figure_paths(explainability_dir),
        "gradcam_selection": _load_optional_json(explainability_dir / "figure_selection_manifest.json"),
    }


def _summarize_embeddings(embedding_path: Path) -> dict[str, Any]:
    if not embedding_path.exists():
        return {"status": "missing", "embedding_path": str(embedding_path)}

    payload = np.load(embedding_path, allow_pickle=True)
    embeddings = payload["embeddings"].astype(np.float64)
    label_names = payload["label_names"].astype(str)
    subject_ids = payload["subject_ids"].astype(str)

    pca = PCA(n_components=min(2, embeddings.shape[0], embeddings.shape[1]))
    transformed = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_.tolist()

    centroids: dict[str, list[float]] = {}
    for label_name in sorted(set(label_names.tolist()), key=LABEL_NAMES.index):
        mask = label_names == label_name
        centroids[label_name] = transformed[mask].mean(axis=0).tolist()

    centroid_distances: dict[str, float] = {}
    centroid_labels = list(centroids.keys())
    for left_index, left_label in enumerate(centroid_labels):
        for right_label in centroid_labels[left_index + 1 :]:
            left = np.asarray(centroids[left_label], dtype=np.float64)
            right = np.asarray(centroids[right_label], dtype=np.float64)
            centroid_distances[f"{left_label}_vs_{right_label}"] = float(np.linalg.norm(left - right))

    return {
        "status": "completed",
        "embedding_path": str(embedding_path),
        "num_subjects": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "explained_variance_ratio": explained_variance,
        "centroids": centroids,
        "centroid_distances": centroid_distances,
        "subject_ids": subject_ids.tolist(),
    }


def _collect_figure_paths(directory: Path) -> dict[str, list[str] | str]:
    if not directory.exists():
        return {"status": "missing", "files": []}
    files = sorted(str(path) for path in directory.iterdir() if path.is_file())
    return {"status": "completed", "files": files}


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_delta_summary(smoke: dict[str, Any], long_run: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch_increase": long_run["completed_epochs"] - smoke["completed_epochs"],
        "train_loss_delta": long_run["final_train_loss"] - smoke["final_train_loss"],
        "val_loss_delta": long_run["final_val_loss"] - smoke["final_val_loss"],
        "train_balanced_accuracy_delta": long_run["final_train_balanced_accuracy"] - smoke["final_train_balanced_accuracy"],
        "val_balanced_accuracy_delta": long_run["final_val_balanced_accuracy"] - smoke["final_val_balanced_accuracy"],
        "train_macro_f1_delta": long_run["final_train_macro_f1"] - smoke["final_train_macro_f1"],
        "val_macro_f1_delta": long_run["final_val_macro_f1"] - smoke["final_val_macro_f1"],
        "test_balanced_accuracy_delta": long_run["test_balanced_accuracy"] - smoke["test_balanced_accuracy"],
        "test_macro_f1_delta": long_run["test_macro_f1"] - smoke["test_macro_f1"],
    }


def _write_markdown_report(output_path: Path, comparison: dict[str, Any]) -> None:
    smoke = comparison["smoke"]
    long_run = comparison["long_run"]
    delta = comparison["delta"]
    lines = [
        "# ds007561 Multimodal Smoke vs 50-Epoch Comparison",
        "",
        "This comparison is exploratory only. The cohort is very small and strongly imbalanced, with only one AD subject.",
        "",
        "## Metrics Summary",
        "",
        "| Run | Epochs | Best Val Epoch | Final Train Loss | Final Val Loss | Train Balanced Acc. | Val Balanced Acc. | Test Balanced Acc. | Train Macro F1 | Val Macro F1 | Test Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _summary_row(smoke),
        _summary_row(long_run),
        "",
        "## Interpretation",
        "",
        f"- The 50-epoch run reduced training loss by `{delta['train_loss_delta']:.4f}` relative to smoke, while validation loss changed by `{delta['val_loss_delta']:.4f}`.",
        f"- Training balanced accuracy changed by `{delta['train_balanced_accuracy_delta']:.4f}`, but validation balanced accuracy changed by `{delta['val_balanced_accuracy_delta']:.4f}`.",
        f"- Training macro F1 changed by `{delta['train_macro_f1_delta']:.4f}`, while validation macro F1 changed by `{delta['val_macro_f1_delta']:.4f}`.",
        f"- Test balanced accuracy changed by `{delta['test_balanced_accuracy_delta']:.4f}` and test macro F1 changed by `{delta['test_macro_f1_delta']:.4f}`.",
        "- If training improves while validation and test remain flat, that should be read as overfitting rather than as a meaningful gain.",
        "",
        "## Confusion Matrices",
        "",
        f"- Smoke train: `{smoke['train_confusion_matrix']}`",
        f"- Smoke validation: `{smoke['val_confusion_matrix']}`",
        f"- Smoke test: `{smoke['test_confusion_matrix']}`",
        f"- 50-epoch train: `{long_run['train_confusion_matrix']}`",
        f"- 50-epoch validation: `{long_run['val_confusion_matrix']}`",
        f"- 50-epoch test: `{long_run['test_confusion_matrix']}`",
        "",
        "## Latent Space",
        "",
        f"- Smoke embeddings: `{smoke['embedding_summary'].get('embedding_path', 'missing')}`",
        f"- 50-epoch embeddings: `{long_run['embedding_summary'].get('embedding_path', 'missing')}`",
        f"- Smoke PCA variance ratio: `{_fmt_list(smoke['embedding_summary'].get('explained_variance_ratio', []))}`",
        f"- 50-epoch PCA variance ratio: `{_fmt_list(long_run['embedding_summary'].get('explained_variance_ratio', []))}`",
        f"- Smoke centroid distances: `{smoke['embedding_summary'].get('centroid_distances', {})}`",
        f"- 50-epoch centroid distances: `{long_run['embedding_summary'].get('centroid_distances', {})}`",
        "",
        "Figures:",
        f"- Smoke latent space: {_latent_links(smoke['latent_space_outputs']['files'])}",
        f"- 50-epoch latent space: {_latent_links(long_run['latent_space_outputs']['files'])}",
        "",
        "## Grad-CAM",
        "",
        "- Both runs use the same refined visualization pipeline: MRI-derived brain masking, percentile thresholding, Gaussian smoothing, consistent cropping, and aligned MRI/PET overlays.",
        "- The Grad-CAM outputs remain exploratory and should not be read as clinical anatomical evidence.",
        f"- Smoke selection manifest: `{smoke['gradcam_selection']}`",
        f"- 50-epoch selection manifest: `{long_run['gradcam_selection']}`",
        "",
        "Figures:",
        *_figure_bullets("Smoke", smoke["explainability_outputs"]["files"]),
        *_figure_bullets("50-epoch", long_run["explainability_outputs"]["files"]),
        "",
        "## Limitations",
        "",
        "- Only one AD subject is available in ds007561, so val/test do not include AD under the current fixed split.",
        "- Any visual separation in PCA or t-SNE should be interpreted as exploratory structure, not clinical discrimination.",
        "- The Grad-CAM overlays are useful for qualitative inspection of model attention, not for anatomical claims.",
    ]
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_row(run: dict[str, Any]) -> str:
    return (
        f"| {run['run_name']} | {run['completed_epochs']} | {run['best_val_epoch']} | "
        f"{run['final_train_loss']:.4f} | {run['final_val_loss']:.4f} | "
        f"{run['final_train_balanced_accuracy']:.4f} | {run['final_val_balanced_accuracy']:.4f} | "
        f"{run['test_balanced_accuracy']:.4f} | {run['final_train_macro_f1']:.4f} | "
        f"{run['final_val_macro_f1']:.4f} | {run['test_macro_f1']:.4f} |"
    )


def _fmt_list(values: list[float]) -> str:
    return ", ".join(f"{value:.4f}" for value in values)


def _figure_bullets(prefix: str, files: list[str]) -> list[str]:
    bullets: list[str] = []
    for path_str in files:
        path = Path(path_str).resolve()
        bullets.append(f"- {prefix}: [{path.name}]({path})")
    return bullets


def _latent_links(files: list[str]) -> str:
    pca_file = next((Path(path).resolve() for path in files if Path(path).name == "multimodal_pca.png"), None)
    tsne_file = next((Path(path).resolve() for path in files if Path(path).name == "multimodal_tsne.png"), None)
    parts: list[str] = []
    if pca_file:
        parts.append(f"[multimodal_pca.png]({pca_file})")
    if tsne_file:
        parts.append(f"[multimodal_tsne.png]({tsne_file})")
    return " and ".join(parts) if parts else "No multimodal latent-space figures found"


if __name__ == "__main__":
    main()
