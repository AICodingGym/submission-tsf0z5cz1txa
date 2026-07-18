from __future__ import annotations

import itertools

import numpy as np

from src.baseline import apply_thresholds, optimize_thresholds, quadratic_weighted_kappa


def blend_raw(prediction_streams: np.ndarray, weights: np.ndarray) -> np.ndarray:
    streams = np.asarray(prediction_streams, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if streams.ndim != 2:
        raise ValueError("prediction_streams must have shape (models, rows)")
    if weights.shape != (streams.shape[0],):
        raise ValueError("One weight is required for every prediction stream")
    if np.any(weights < -1e-12) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Blend weights must be nonnegative and sum to one")
    return weights @ streams


def simplex_grid(model_count: int, denominator: int) -> list[np.ndarray]:
    """Enumerate nonnegative weights summing to one on a finite grid."""
    if model_count < 2 or denominator < 1:
        raise ValueError("At least two models and a positive denominator are required")
    weights: list[np.ndarray] = []
    for cuts in itertools.combinations_with_replacement(
        range(denominator + 1), model_count - 1
    ):
        boundaries = (0, *cuts, denominator)
        parts = np.diff(boundaries)
        weights.append(parts.astype(np.float64) / denominator)
    return weights


def optimize_multimodel_blend(
    labels: np.ndarray,
    prediction_streams: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Tune a small nonnegative linear ensemble and five ordinal thresholds."""
    labels = np.asarray(labels, dtype=np.int8)
    streams = np.asarray(prediction_streams, dtype=np.float64)
    if streams.ndim != 2 or streams.shape[1] != len(labels):
        raise ValueError("Prediction streams must align with labels")
    class_counts = np.bincount(labels, minlength=7)[1:7]
    cumulative = np.cumsum(class_counts)[:-1] / len(labels)

    best_weights = np.full(streams.shape[0], 1.0 / streams.shape[0])
    best_thresholds = np.arange(1.5, 6.0, 1.0)
    best_score = -np.inf
    for weights in simplex_grid(streams.shape[0], denominator=20):
        raw = blend_raw(streams, weights)
        thresholds = np.quantile(raw, cumulative)
        score = quadratic_weighted_kappa(labels, apply_thresholds(raw, thresholds))
        if score > best_score:
            best_weights, best_thresholds, best_score = weights, thresholds, score

    for transfer in (0.05, 0.02, 0.01, 0.005):
        raw = blend_raw(streams, best_weights)
        best_thresholds, best_score = optimize_thresholds(labels, raw)
        improved = True
        while improved:
            improved = False
            for source in range(len(best_weights)):
                for destination in range(len(best_weights)):
                    if source == destination or best_weights[source] < transfer:
                        continue
                    trial_weights = best_weights.copy()
                    trial_weights[source] -= transfer
                    trial_weights[destination] += transfer
                    trial_raw = blend_raw(streams, trial_weights)
                    score = quadratic_weighted_kappa(
                        labels, apply_thresholds(trial_raw, best_thresholds)
                    )
                    if score > best_score + 1e-12:
                        best_weights, best_score = trial_weights, score
                        improved = True

    raw = blend_raw(streams, best_weights)
    best_thresholds, best_score = optimize_thresholds(labels, raw)
    return best_weights, best_thresholds, best_score


def crossfit_multimodel_blend(
    labels: np.ndarray,
    prediction_streams: np.ndarray,
    fold_ids: np.ndarray,
) -> tuple[np.ndarray, float, list[dict[str, object]]]:
    """Fit weights/thresholds off-fold, then evaluate on the untouched fold."""
    labels = np.asarray(labels, dtype=np.int8)
    streams = np.asarray(prediction_streams, dtype=np.float64)
    fold_ids = np.asarray(fold_ids, dtype=np.int8)
    if streams.shape[1] != len(labels) or fold_ids.shape != labels.shape:
        raise ValueError("Labels, streams, and fold IDs must align")

    predictions = np.zeros(len(labels), dtype=np.int8)
    parameters: list[dict[str, object]] = []
    for fold in sorted(np.unique(fold_ids)):
        calibration = fold_ids != fold
        evaluation = fold_ids == fold
        weights, thresholds, calibration_qwk = optimize_multimodel_blend(
            labels[calibration], streams[:, calibration]
        )
        raw = blend_raw(streams[:, evaluation], weights)
        predictions[evaluation] = apply_thresholds(raw, thresholds)
        parameters.append(
            {
                "fold": int(fold),
                "weights": weights.tolist(),
                "thresholds": thresholds.tolist(),
                "calibration_qwk": calibration_qwk,
                "evaluation_qwk": quadratic_weighted_kappa(
                    labels[evaluation], predictions[evaluation]
                ),
            }
        )
    return predictions, quadratic_weighted_kappa(labels, predictions), parameters
