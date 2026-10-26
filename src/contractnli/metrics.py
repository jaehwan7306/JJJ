from __future__ import annotations

from .data import EVALUATION_LABELS


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def classification(rows: list[dict]) -> dict:
    """Macro F1 over all three gold classes; invalid outputs count as errors."""
    if not rows:
        raise ValueError("Cannot evaluate zero predictions")
    per_label = {}
    for label in EVALUATION_LABELS:
        tp = sum(r["gold"] == label and r["prediction"] == label for r in rows)
        fp = sum(r["gold"] != label and r["prediction"] == label for r in rows)
        fn = sum(r["gold"] == label and r["prediction"] != label for r in rows)
        per_label[label] = {"support": sum(r["gold"] == label for r in rows),
                            "precision": tp / (tp + fp) if tp + fp else 0.0,
                            "recall": tp / (tp + fn) if tp + fn else 0.0,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    return {"n": len(rows), "accuracy": mean(r["gold"] == r["prediction"] for r in rows),
            "macro_f1": mean(v["f1"] for v in per_label.values()), "per_label": per_label,
            "invalid_outputs": sum(r["prediction"] is None for r in rows),
            "not_mentioned_outputs": sum(r["prediction"] == "NotMentioned" for r in rows),
            "confusion": {label: {pred: sum(r["gold"] == label and
                             (r["prediction"] or "Invalid") == pred for r in rows)
                             for pred in (*EVALUATION_LABELS, "Invalid")}
                          for label in EVALUATION_LABELS}}


def paired_metrics(original: list[dict], compressed: list[dict]) -> dict:
    baseline = {r["example_id"]: r for r in original}
    other = {r["example_id"]: r for r in compressed}
    if len(baseline) != len(original) or len(other) != len(compressed):
        raise ValueError("Duplicate example ids")
    if baseline.keys() != other.keys():
        raise ValueError("Paired metrics require identical example sets")
    pairs = [(baseline[k], other[k]) for k in baseline]
    if any(a["gold"] != b["gold"] for a, b in pairs):
        raise ValueError("Paired gold labels disagree")
    valid = [(a, b) for a, b in pairs if a["prediction"] is not None and b["prediction"] is not None]
    correct_original = [(a, b) for a, b in pairs if a["prediction"] == a["gold"]]
    return {
        "paired_n": len(pairs), "valid_pair_n": len(valid),
        "decision_flip_rate": mean(a["prediction"] != b["prediction"] for a, b in valid),
        "decision_agreement_rate": mean(a["prediction"] == b["prediction"] for a, b in valid),
        "original_correct_n": len(correct_original),
        "correct_decision_retention": mean(b["prediction"] == b["gold"] for _, b in correct_original),
        "correct_to_wrong_n": sum(a["prediction"] == a["gold"] and b["prediction"] != b["gold"]
                                  for a, b in pairs),
        "wrong_to_correct_n": sum(a["prediction"] != a["gold"] and b["prediction"] == b["gold"]
                                  for a, b in pairs),
    }
