from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = {
    "multimodal": "reports/experiments/ds007561_multimodal_smoke",
    "mri_only": "reports/experiments/ds007561_mri_only_smoke",
    "pet_only": "reports/experiments/ds007561_pet_only_smoke",
}

SUMMARY_OUTPUT = Path("reports/experiments/ds007561_model_comparison.json")
FIGURE_OUTPUT = Path("reports/figures/ds007561_model_comparison.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ds007561 smoke-test experiment outputs.")
    parser.add_argument(
        "--output-json",
        default=str(SUMMARY_OUTPUT),
        help="Path to the comparison JSON output.",
    )
    parser.add_argument(
        "--output-figure",
        default=str(FIGURE_OUTPUT),
        help="Path to the comparison SVG figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_payloads = {
        experiment_name: _load_experiment_payload(Path(experiment_dir))
        for experiment_name, experiment_dir in EXPERIMENTS.items()
    }
    comparison_payload = _build_comparison_payload(experiment_payloads)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    output_figure = Path(args.output_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)
    output_figure.write_text(_build_svg_chart(comparison_payload["summary_table"]), encoding="utf-8")

    _print_summary_table(comparison_payload["summary_table"])
    print(f"\nSaved comparison JSON: {output_json}")
    print(f"Saved comparison figure: {output_figure}")


def _load_experiment_payload(experiment_dir: Path) -> dict[str, Any]:
    metrics_path = experiment_dir / "metrics.json"
    splits_path = experiment_dir / "subject_splits.json"
    history_path = experiment_dir / "history.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {experiment_dir}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing subject_splits.json in {experiment_dir}")
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json in {experiment_dir}")
    return {
        "experiment_dir": str(experiment_dir),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "splits": json.loads(splits_path.read_text(encoding="utf-8")),
        "history": json.loads(history_path.read_text(encoding="utf-8")),
    }


def _build_comparison_payload(experiments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reference_split = experiments["multimodal"]["splits"]
    split_consistency = {
        experiment_name: payload["splits"] == reference_split
        for experiment_name, payload in experiments.items()
    }
    summary_table = [
        _build_summary_row(experiment_name, payload)
        for experiment_name, payload in experiments.items()
    ]
    summary_table.sort(key=lambda row: row["modality"])

    return {
        "dataset": "ds007561",
        "comparison_type": "smoke_test_baseline",
        "split_consistency": split_consistency,
        "shared_split_manifest": reference_split,
        "summary_table": summary_table,
        "notes": [
            "This comparison is a technical proof-of-concept on a very small exploratory cohort.",
            "The dataset contains only one AD subject, which remains in the training split.",
            "Validation and test splits contain Control and MCI only, so class-wise AD performance is not estimable there.",
        ],
    }


def _build_summary_row(experiment_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload["metrics"]
    history = payload["history"]["history"]
    train_history = history[-1] if history else {}
    train_metrics = metrics["latest_train_metrics"]
    val_metrics = metrics["latest_val_metrics"]
    test_metrics = metrics["test_metrics"]
    return {
        "modality": experiment_name,
        "experiment_dir": payload["experiment_dir"],
        "status": metrics["status"],
        "device": metrics["device"],
        "completed_epochs": metrics["completed_epochs"],
        "train": {
            "loss": train_history.get("train_loss"),
            "balanced_accuracy": train_metrics["balanced_accuracy"],
            "macro_f1": train_metrics["macro_f1"],
            "confusion_matrix": train_metrics["confusion_matrix"],
            "per_class": train_metrics["per_class"],
        },
        "val": {
            "loss": train_history.get("val_loss"),
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "per_class": val_metrics["per_class"],
        },
        "test": {
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "per_class": test_metrics["per_class"],
        },
    }


def _build_svg_chart(summary_rows: list[dict[str, Any]]) -> str:
    width = 1200
    height = 760
    margin_left = 90
    margin_right = 40
    margin_top = 80
    margin_bottom = 90
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    categories = [
        ("Train Balanced Acc.", "train", "balanced_accuracy"),
        ("Val Balanced Acc.", "val", "balanced_accuracy"),
        ("Test Balanced Acc.", "test", "balanced_accuracy"),
        ("Train Macro F1", "train", "macro_f1"),
        ("Val Macro F1", "val", "macro_f1"),
        ("Test Macro F1", "test", "macro_f1"),
    ]
    colors = {
        "multimodal": "#0f766e",
        "mri_only": "#2563eb",
        "pet_only": "#d97706",
    }
    modality_labels = {
        "multimodal": "Multimodal",
        "mri_only": "MRI-only",
        "pet_only": "PET-only",
    }
    category_slot = chart_width / len(categories)
    group_width = category_slot * 0.72
    bar_width = group_width / max(len(summary_rows), 1)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="90" y="42" font-size="28" font-family="Arial, sans-serif" font-weight="700" fill="#0f172a">ds007561 Smoke-Test Model Comparison</text>',
        '<text x="90" y="66" font-size="14" font-family="Arial, sans-serif" fill="#334155">Balanced accuracy and macro F1 across multimodal, MRI-only, and PET-only baselines</text>',
    ]

    for tick in range(6):
        y_value = tick / 5
        y = margin_top + chart_height - (chart_height * y_value)
        svg_parts.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" stroke="#cbd5e1" stroke-width="1"/>'
        )
        svg_parts.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="12" font-family="Arial, sans-serif" fill="#475569">{y_value:.1f}</text>'
        )

    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#0f172a" stroke-width="2"/>'
    )
    svg_parts.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#0f172a" stroke-width="2"/>'
    )

    for category_index, (label, split_name, metric_name) in enumerate(categories):
        group_x = margin_left + (category_index * category_slot) + ((category_slot - group_width) / 2)
        for row_index, row in enumerate(summary_rows):
            value = float(row[split_name][metric_name])
            bar_height = chart_height * value
            x = group_x + (row_index * bar_width)
            y = margin_top + chart_height - bar_height
            modality = row["modality"]
            svg_parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 6:.2f}" height="{bar_height:.2f}" fill="{colors[modality]}" rx="4" ry="4"/>'
            )
            svg_parts.append(
                f'<text x="{x + (bar_width - 6) / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="11" font-family="Arial, sans-serif" fill="#0f172a">{value:.2f}</text>'
            )
        svg_parts.append(
            f'<text x="{group_x + (group_width / 2):.2f}" y="{height - 42}" text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#334155">{label}</text>'
        )

    legend_x = width - 360
    legend_y = 30
    for index, row in enumerate(summary_rows):
        modality = row["modality"]
        y = legend_y + (index * 24)
        svg_parts.append(
            f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{colors[modality]}" rx="3" ry="3"/>'
        )
        svg_parts.append(
            f'<text x="{legend_x + 22}" y="{y + 12}" font-size="13" font-family="Arial, sans-serif" fill="#0f172a">{modality_labels[modality]}</text>'
        )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def _print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    print("Modality        Train BA  Val BA  Test BA  Train F1  Val F1  Test F1")
    print("---------------------------------------------------------------------")
    for row in summary_rows:
        print(
            f"{row['modality']:<14}"
            f"{row['train']['balanced_accuracy']:<10.3f}"
            f"{row['val']['balanced_accuracy']:<8.3f}"
            f"{row['test']['balanced_accuracy']:<9.3f}"
            f"{row['train']['macro_f1']:<10.3f}"
            f"{row['val']['macro_f1']:<8.3f}"
            f"{row['test']['macro_f1']:<8.3f}"
        )


if __name__ == "__main__":
    main()
