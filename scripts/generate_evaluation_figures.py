from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = {
    "multimodal": "reports/experiments/ds007561_multimodal_smoke",
    "mri_only": "reports/experiments/ds007561_mri_only_smoke",
    "pet_only": "reports/experiments/ds007561_pet_only_smoke",
}
LABEL_NAMES = ["Control", "MCI", "AD"]
MODEL_LABELS = {
    "multimodal": "Multimodal",
    "mri_only": "MRI-only",
    "pet_only": "PET-only",
}
MODEL_COLORS = {
    "multimodal": "#0f766e",
    "mri_only": "#2563eb",
    "pet_only": "#d97706",
}
FIGURE_DPI = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation figures from existing smoke-test outputs.")
    parser.add_argument(
        "--experiment-dir",
        default=None,
        help="Optional single experiment directory. If provided, only this experiment is plotted.",
    )
    parser.add_argument(
        "--modality",
        default="multimodal",
        choices=sorted(MODEL_LABELS),
        help="Modality key used when --experiment-dir is provided.",
    )
    parser.add_argument(
        "--title-prefix",
        default="ds007561 Exploratory Smoke-Test",
        help="Prefix used in figure titles.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/figures/evaluation",
        help="Directory for evaluation figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment_dir:
        payloads = {args.modality: _load_experiment_payload(Path(args.experiment_dir))}
    else:
        payloads = {
            modality: _load_experiment_payload(Path(experiment_dir))
            for modality, experiment_dir in EXPERIMENTS.items()
        }

    _plot_loss_curves(payloads, output_dir, args.title_prefix)
    _plot_metric_curves(payloads, output_dir, args.title_prefix)
    _plot_confusion_matrices(payloads, output_dir, args.title_prefix)
    _write_manifest(payloads, output_dir)


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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _load_experiment_payload(experiment_dir: Path) -> dict[str, Any]:
    history_path = experiment_dir / "history.json"
    metrics_path = experiment_dir / "metrics.json"
    if not history_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Missing history.json or metrics.json in {experiment_dir}")
    return {
        "experiment_dir": str(experiment_dir),
        "history": json.loads(history_path.read_text(encoding="utf-8"))["history"],
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
    }


def _plot_loss_curves(payloads: dict[str, dict[str, Any]], output_dir: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=FIGURE_DPI, sharex=True)
    splits = [("train_loss", "Training Loss"), ("val_loss", "Validation Loss")]

    for ax, (history_key, panel_title) in zip(axes, splits, strict=True):
        for modality, payload in payloads.items():
            epochs = [row["epoch"] for row in payload["history"]]
            values = [row[history_key] for row in payload["history"]]
            ax.plot(
                epochs,
                values,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=MODEL_COLORS[modality],
                label=MODEL_LABELS[modality],
            )
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.12, linewidth=0.8)
        ax.set_facecolor("#fcfcfd")
        if payloads:
            ax.set_xticks(sorted({row["epoch"] for payload in payloads.values() for row in payload["history"]}))

    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Model")
    fig.suptitle(f"{title_prefix} Loss Curves", fontsize=17, y=1.02)
    fig.text(
        0.02,
        0.98,
        "Exploratory only - very small and imbalanced cohort",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    _save_figure(fig, output_dir / "loss_curves")


def _plot_metric_curves(payloads: dict[str, dict[str, Any]], output_dir: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), dpi=FIGURE_DPI, sharex=True)
    metric_panels = [
        ("train_balanced_accuracy", "Train Balanced Accuracy"),
        ("val_balanced_accuracy", "Validation Balanced Accuracy"),
        ("train_macro_f1", "Train Macro F1"),
        ("val_macro_f1", "Validation Macro F1"),
    ]

    for ax, (metric_key, panel_title) in zip(axes.ravel(), metric_panels, strict=True):
        for modality, payload in payloads.items():
            epochs = [row["epoch"] for row in payload["history"]]
            values = [row[metric_key] for row in payload["history"]]
            ax.plot(
                epochs,
                values,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=MODEL_COLORS[modality],
                label=MODEL_LABELS[modality],
            )
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.12, linewidth=0.8)
        ax.set_facecolor("#fcfcfd")
        if payloads:
            ax.set_xticks(sorted({row["epoch"] for payload in payloads.values() for row in payload["history"]}))

    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Model")
    fig.suptitle(f"{title_prefix} Balanced Accuracy and Macro F1", fontsize=17, y=1.01)
    fig.text(
        0.02,
        0.98,
        "Exploratory only - balanced accuracy and macro F1 are preferred over raw accuracy here",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    _save_figure(fig, output_dir / "balanced_accuracy_macro_f1")


def _plot_confusion_matrices(payloads: dict[str, dict[str, Any]], output_dir: Path, title_prefix: str) -> None:
    splits = [
        ("latest_train_metrics", "Train"),
        ("latest_val_metrics", "Validation"),
        ("test_metrics", "Test"),
    ]
    modalities = list(payloads.keys())
    fig, axes = plt.subplots(
        len(splits),
        len(modalities),
        figsize=(4.1 * len(modalities), 3.7 * len(splits)),
        dpi=FIGURE_DPI,
        squeeze=False,
    )

    max_value = 1
    for modality in modalities:
        metrics = payloads[modality]["metrics"]
        for metrics_key, _split_label in splits:
            matrix = np.asarray(metrics[metrics_key]["confusion_matrix"], dtype=float)
            max_value = max(max_value, int(matrix.max()))

    for row_index, (metrics_key, split_label) in enumerate(splits):
        for col_index, modality in enumerate(modalities):
            ax = axes[row_index, col_index]
            matrix = np.asarray(payloads[modality]["metrics"][metrics_key]["confusion_matrix"], dtype=float)
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max_value)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = int(matrix[i, j])
                    text_color = "white" if value > max_value * 0.45 else "#0f172a"
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=10.5, color=text_color)
            ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=30, ha="right")
            ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{MODEL_LABELS[modality]} | {split_label}", fontsize=12)
            ax.set_facecolor("#fcfcfd")

    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    cbar.set_label("Sample count", fontsize=10)
    fig.suptitle(f"{title_prefix} Confusion Matrices", fontsize=17, y=0.995)
    fig.text(
        0.02,
        0.975,
        "Exploratory only - train, validation, and test confusion matrices for all three model variants",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#475569",
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.07, top=0.92, wspace=0.34, hspace=0.34)
    _save_figure(fig, output_dir / "confusion_matrices")


def _write_manifest(payloads: dict[str, dict[str, Any]], output_dir: Path) -> None:
    manifest = {
        "dataset": "ds007561",
        "status": "completed",
        "figures": [
            str(output_dir / "loss_curves.png"),
            str(output_dir / "loss_curves.svg"),
            str(output_dir / "balanced_accuracy_macro_f1.png"),
            str(output_dir / "balanced_accuracy_macro_f1.svg"),
            str(output_dir / "confusion_matrices.png"),
            str(output_dir / "confusion_matrices.svg"),
        ],
        "roc_curves": {
            "status": "skipped",
            "reason": "ROC curves were not generated because the current outputs are multiclass smoke-test summaries without stored class-probability traces, and binary reduction would not be statistically meaningful on this cohort.",
        },
        "notes": [
            "All figures are exploratory because the cohort is extremely small and imbalanced.",
            "Validation and test splits do not contain AD, so per-class conclusions for AD are not possible there.",
        ],
        "models": {
            modality: {
                "experiment_dir": payload["experiment_dir"],
                "completed_epochs": payload["metrics"]["completed_epochs"],
                "device": payload["metrics"]["device"],
            }
            for modality, payload in payloads.items()
        },
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
