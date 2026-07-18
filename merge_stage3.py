from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.baseline import (
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa,
    write_submission,
)
from src.calibration import (
    blend_raw,
    crossfit_multimodel_blend,
    optimize_multimodel_blend,
)
from src.transformer_data import TokenCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge five Transformer folds and calibrate a stage-3 ensemble."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage3"))
    parser.add_argument(
        "--stage2-predictions",
        type=Path,
        default=Path("artifacts/stage2_predictions.npz"),
    )
    parser.add_argument(
        "--token-cache", type=Path, default=Path("artifacts/token_cache/modernbert")
    )
    parser.add_argument("--folds-file", type=Path, default=Path("artifacts/folds.csv"))
    parser.add_argument(
        "--submission", type=Path, default=Path("stage3_submission.csv")
    )
    return parser.parse_args()


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> list[list[int]]:
    matrix = np.zeros((6, 6), dtype=np.int64)
    for label, prediction in zip(labels, predictions, strict=True):
        matrix[int(label) - 1, int(prediction) - 1] += 1
    return matrix.tolist()


def grouped_qwk(
    labels: np.ndarray, predictions: np.ndarray, groups: np.ndarray
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for group in sorted(np.unique(groups)):
        mask = groups == group
        result[str(group)] = {
            "rows": int(mask.sum()),
            "qwk": quadratic_weighted_kappa(labels[mask], predictions[mask]),
        }
    return result


def load_topics(path: Path, train_ids: np.ndarray) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    mapping = {row["essay_id"]: int(row["topic_cluster"]) for row in rows}
    if set(mapping) != set(map(str, train_ids)):
        raise ValueError("Topic manifest does not align with stage-2 train IDs")
    return np.asarray([mapping[str(essay_id)] for essay_id in train_ids], dtype=np.int8)


def main() -> None:
    args = parse_args()
    stage2 = np.load(args.stage2_predictions)
    labels = stage2["labels"].astype(np.int8)
    fold_ids = stage2["fold_ids"].astype(np.int8)
    transformer_oof = np.full(len(labels), np.nan, dtype=np.float64)
    transformer_test = np.zeros(len(stage2["test_ids"]), dtype=np.float64)
    fold_metrics: list[dict[str, float | int]] = []

    for fold in range(5):
        path = args.root / args.run_name / f"fold_{fold}" / "predictions.npz"
        prediction = np.load(path)
        expected_indices = np.flatnonzero(fold_ids == fold)
        if not np.array_equal(prediction["valid_indices"], expected_indices):
            raise ValueError(f"Validation indices do not align for fold {fold}")
        expected_test_indices = np.arange(len(stage2["test_ids"]))
        if not np.array_equal(prediction["test_indices"], expected_test_indices):
            raise ValueError(f"Test indices do not align for fold {fold}")
        transformer_oof[expected_indices] = prediction["valid_raw"]
        transformer_test += prediction["test_raw"] / 5
        fold_labels = labels[expected_indices]
        fold_raw = prediction["valid_raw"]
        rounded = np.clip(np.rint(fold_raw), 1, 6).astype(np.int8)
        fold_metrics.append(
            {
                "fold": fold,
                "rows": len(expected_indices),
                "rounded_qwk": quadratic_weighted_kappa(fold_labels, rounded),
                "optimized_qwk": optimize_thresholds(fold_labels, fold_raw)[1],
                "rmse": float(np.sqrt(np.mean((fold_raw - fold_labels) ** 2))),
            }
        )
    if not np.isfinite(transformer_oof).all():
        raise RuntimeError("Transformer OOF predictions are incomplete")

    transformer_thresholds, transformer_qwk = optimize_thresholds(
        labels, transformer_oof
    )
    transformer_rounded = np.clip(np.rint(transformer_oof), 1, 6).astype(np.int8)
    streams = np.vstack(
        (stage2["text_oof"], stage2["feature_oof"], transformer_oof)
    )
    crossfit_predictions, crossfit_qwk, crossfit_parameters = (
        crossfit_multimodel_blend(labels, streams, fold_ids)
    )
    final_weights, final_thresholds, full_oof_qwk = optimize_multimodel_blend(
        labels, streams
    )
    test_streams = np.vstack(
        (stage2["text_test"], stage2["feature_test"], transformer_test)
    )
    test_raw = blend_raw(test_streams, final_weights)
    test_scores = apply_thresholds(test_raw, final_thresholds)
    write_submission(args.submission, list(map(str, stage2["test_ids"])), test_scores)

    stage2_crossfit_qwk = quadratic_weighted_kappa(
        labels, stage2["crossfit_predictions"].astype(np.int8)
    )
    train_cache = TokenCache(args.token_cache, "train")
    lengths = np.diff(train_cache.offsets)
    length_edges = np.quantile(lengths, (0.25, 0.5, 0.75))
    length_groups = np.digitize(lengths, length_edges).astype(np.int8)
    topics = load_topics(args.folds_file, stage2["train_ids"])
    residuals = transformer_oof - labels
    per_class = {}
    for score in range(1, 7):
        mask = labels == score
        per_class[str(score)] = {
            "rows": int(mask.sum()),
            "prediction_mean": float(transformer_oof[mask].mean()),
            "mae": float(np.abs(residuals[mask]).mean()),
        }

    metrics = {
        "run_name": args.run_name,
        "folds": fold_metrics,
        "transformer_rounded_qwk": quadratic_weighted_kappa(
            labels, transformer_rounded
        ),
        "transformer_optimized_qwk": transformer_qwk,
        "transformer_thresholds": transformer_thresholds.tolist(),
        "transformer_rmse": float(np.sqrt(np.mean(residuals**2))),
        "transformer_mae": float(np.mean(np.abs(residuals))),
        "transformer_residual_length_correlation": float(
            np.corrcoef(residuals, lengths)[0, 1]
        ),
        "transformer_per_class": per_class,
        "stage2_crossfit_qwk": stage2_crossfit_qwk,
        "stage3_crossfit_qwk": crossfit_qwk,
        "crossfit_improvement": crossfit_qwk - stage2_crossfit_qwk,
        "passes_full_cv_gate": crossfit_qwk - stage2_crossfit_qwk >= 0.005,
        "crossfit_parameters": crossfit_parameters,
        "full_oof_qwk": full_oof_qwk,
        "full_oof_optimism_gap": full_oof_qwk - crossfit_qwk,
        "final_weights": {
            "text": float(final_weights[0]),
            "features": float(final_weights[1]),
            "transformer": float(final_weights[2]),
        },
        "final_thresholds": final_thresholds.tolist(),
        "crossfit_confusion_matrix": confusion_matrix(
            labels, crossfit_predictions
        ),
        "crossfit_qwk_by_length_quartile": grouped_qwk(
            labels, crossfit_predictions, length_groups
        ),
        "crossfit_qwk_by_topic": grouped_qwk(
            labels, crossfit_predictions, topics
        ),
        "length_quartile_edges": length_edges.tolist(),
        "submission_distribution": dict(
            sorted(Counter(map(int, test_scores)).items())
        ),
    }
    metrics_path = args.root / args.run_name / "stage3_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.root / args.run_name / "stage3_predictions.npz",
        train_ids=stage2["train_ids"],
        test_ids=stage2["test_ids"],
        fold_ids=fold_ids,
        labels=labels,
        transformer_oof=transformer_oof,
        transformer_test=transformer_test,
        crossfit_predictions=crossfit_predictions,
        final_test_raw=test_raw,
        final_test_scores=test_scores,
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Wrote {args.submission} and {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
