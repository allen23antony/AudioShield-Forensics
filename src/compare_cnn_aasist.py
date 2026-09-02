from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CNN_METRICS_PATH = RESULTS / "cnn_final_eval_results.json"
AASIST_METRICS_PATH = RESULTS / "aasist" / "aasist_eval_results.json"
CNN_ATTACK_PATH = RESULTS / "cnn_final_attack_analysis.json"
AASIST_ATTACK_PATH = RESULTS / "aasist" / "aasist_attack_analysis.json"

METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "fpr",
    "roc_auc",
    "eer",
)
THRESHOLD_DEPENDENT = {"accuracy", "precision", "recall", "f1", "specificity", "fpr"}
LOWER_IS_BETTER = {"fpr", "eer"}
ATTACKS = [f"A{i:02d}" for i in range(7, 20)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required result artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def extract_eval_metrics(payload: dict[str, Any], path: Path) -> dict[str, float]:
    """Extract official-eval metrics without falling back to DEV metrics."""
    if path.name == "cnn_final_eval_results.json" and payload.get("split") != "eval":
        raise ValueError(f"CNN result artifact is not marked as eval: {path}")

    nested = payload.get("metrics")
    nested_metrics = nested if isinstance(nested, dict) else {}
    independent = payload.get("threshold_independent_metrics")
    independent_metrics = independent if isinstance(independent, dict) else {}

    values: dict[str, float] = {}
    for metric in METRICS:
        source = independent_metrics if metric in {"roc_auc", "eer"} else nested_metrics
        value = source.get(metric)
        if metric == "f1" and value is None:
            value = source.get("f1_score")
        if value is None:
            value = payload.get(metric)
        if value is None:
            raise KeyError(f"Missing official-eval metric '{metric}' in {path}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite official-eval metric '{metric}' in {path}")
        values[metric] = number
    return values


def metric_comparison(cnn: float, aasist: float, metric: str) -> dict[str, Any]:
    difference = aasist - cnn
    relative = None if cnn == 0 else (difference / abs(cnn)) * 100.0
    favorable_difference = -difference if metric in LOWER_IS_BETTER else difference
    if math.isclose(favorable_difference, 0.0, abs_tol=1e-12):
        better = "tie"
    else:
        better = "AASIST" if favorable_difference > 0 else "CNN"
    return {
        "cnn": cnn,
        "aasist": aasist,
        "absolute_difference_aasist_minus_cnn": difference,
        "relative_difference_percent": relative,
        "better": better,
    }


def load_attack_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    raw_rows = payload.get("attack_analysis")
    if raw_rows is None:
        raw_rows = payload.get("per_attack")
    if not isinstance(raw_rows, list):
        raise KeyError(f"Missing attack_analysis/per_attack list in {path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or "attack" not in row:
            raise ValueError(f"Malformed attack row in {path}")
        rows[str(row["attack"])] = row
    return rows


def extract_attack_metric(row: dict[str, Any], metric: str, attack: str, path: Path) -> float | None:
    value = row.get(metric)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def build_attack_comparison(cnn_rows: dict[str, dict[str, Any]], aasist_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for attack in ATTACKS:
        row: dict[str, Any] = {"attack": attack}
        for metric in ("recall", "f1", "eer"):
            cnn_value = extract_attack_metric(cnn_rows.get(attack, {}), metric, attack, CNN_ATTACK_PATH)
            aasist_value = extract_attack_metric(aasist_rows.get(attack, {}), metric, attack, AASIST_ATTACK_PATH)
            row[f"cnn_{metric}"] = cnn_value
            row[f"aasist_{metric}"] = aasist_value
            if cnn_value is not None and aasist_value is not None:
                row[f"{metric}_absolute_difference_aasist_minus_cnn"] = aasist_value - cnn_value
                row[f"{metric}_better"] = (
                    "AASIST"
                    if (cnn_value - aasist_value if metric in LOWER_IS_BETTER else aasist_value - cnn_value) > 0
                    else "CNN"
                    if (cnn_value - aasist_value if metric in LOWER_IS_BETTER else aasist_value - cnn_value) < 0
                    else "tie"
                )
            else:
                row[f"{metric}_absolute_difference_aasist_minus_cnn"] = None
                row[f"{metric}_better"] = None
        comparisons.append(row)
    return comparisons


def save_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "attack",
        "cnn_recall",
        "aasist_recall",
        "recall_absolute_difference_aasist_minus_cnn",
        "recall_better",
        "cnn_f1",
        "aasist_f1",
        "f1_absolute_difference_aasist_minus_cnn",
        "f1_better",
        "cnn_eer",
        "aasist_eer",
        "eer_absolute_difference_aasist_minus_cnn",
        "eer_better",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_bar_plot(values_a: list[float], values_b: list[float], labels: list[str], title: str, ylabel: str, output_path: Path) -> None:
    x = range(len(labels))
    plt.figure(figsize=(10, 6))
    plt.bar([i - 0.2 for i in x], values_a, width=0.4, label="CNN")
    plt.bar([i + 0.2 for i in x], values_b, width=0.4, label="AASIST")
    plt.xticks(list(x), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    cnn_payload = read_json(CNN_METRICS_PATH)
    aasist_payload = read_json(AASIST_METRICS_PATH)
    cnn_metrics = extract_eval_metrics(cnn_payload, CNN_METRICS_PATH)
    aasist_metrics = extract_eval_metrics(aasist_payload, AASIST_METRICS_PATH)

    overall = {
        metric: metric_comparison(cnn_metrics[metric], aasist_metrics[metric], metric)
        for metric in METRICS
    }
    cnn_attack = load_attack_rows(CNN_ATTACK_PATH)
    aasist_attack = load_attack_rows(AASIST_ATTACK_PATH)
    attack_comparison = build_attack_comparison(cnn_attack, aasist_attack)

    comparison = {
        "source_artifacts": {
            "cnn": str(CNN_METRICS_PATH),
            "aasist": str(AASIST_METRICS_PATH),
            "cnn_attack": str(CNN_ATTACK_PATH),
            "aasist_attack": str(AASIST_ATTACK_PATH),
        },
        "protocol": "Train -> Dev threshold selection -> frozen threshold -> Official Eval",
        "threshold_tuning_performed": False,
        "metric_categories": {
            "threshold_dependent": sorted(THRESHOLD_DEPENDENT),
            "threshold_independent": ["roc_auc", "eer"],
        },
        "overall": overall,
        "attack_wise": attack_comparison,
    }

    output_json = RESULTS / "cnn_vs_aasist_comparison.json"
    output_csv = RESULTS / "cnn_vs_aasist_comparison.csv"
    output_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    save_comparison_csv(attack_comparison, output_csv)

    labels = [row["attack"] for row in attack_comparison]
    for metric, title, ylabel, filename in (
        ("f1", "CNN vs AASIST F1 by Attack", "F1", "cnn_vs_aasist_f1.png"),
        ("recall", "CNN vs AASIST Recall by Attack", "Recall", "cnn_vs_aasist_recall.png"),
        ("eer", "CNN vs AASIST EER by Attack", "EER", "cnn_vs_aasist_eer.png"),
    ):
        cnn_values = [row[f"cnn_{metric}"] if row[f"cnn_{metric}"] is not None else float("nan") for row in attack_comparison]
        aasist_values = [row[f"aasist_{metric}"] if row[f"aasist_{metric}"] is not None else float("nan") for row in attack_comparison]
        save_bar_plot(cnn_values, aasist_values, labels, title, ylabel, RESULTS / filename)

    print("CNN official result keys:", sorted(cnn_payload.keys()))
    print("CNN official metrics source: metrics + threshold_independent_metrics")
    print("AASIST official result keys:", sorted(aasist_payload.keys()))
    print("AASIST official metrics source: top-level fields")
    print(json.dumps({"overall": overall}, indent=2))
    print("Comparison outputs saved to:", output_json, "and", output_csv)
    print("No training, threshold tuning, or model/result artifact modification performed.")


if __name__ == "__main__":
    main()
