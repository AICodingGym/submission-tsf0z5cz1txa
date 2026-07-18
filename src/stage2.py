from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold

from src.baseline import (
    BaselineConfig,
    apply_thresholds,
    optimize_thresholds,
    quadratic_weighted_kappa,
)


@dataclass(frozen=True)
class Stage2Config:
    learning_rate: float = 0.05
    max_iter: int = 350
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 30
    l2_regularization: float = 4.0


def train_cross_validated_feature_model(
    train_features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    baseline_config: BaselineConfig,
    stage2_config: Stage2Config,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """Train a nonlinear model on compact hand-engineered essay features."""
    splitter = StratifiedKFold(
        n_splits=baseline_config.n_splits,
        shuffle=True,
        random_state=baseline_config.seed,
    )
    oof = np.zeros(len(train_features), dtype=np.float64)
    test_predictions = np.zeros(len(test_features), dtype=np.float64)
    fold_metrics: list[dict[str, float | int]] = []

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(train_features, labels), 1
    ):
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=stage2_config.learning_rate,
            max_iter=stage2_config.max_iter,
            max_leaf_nodes=stage2_config.max_leaf_nodes,
            min_samples_leaf=stage2_config.min_samples_leaf,
            l2_regularization=stage2_config.l2_regularization,
            early_stopping=False,
            random_state=baseline_config.seed + fold,
        )
        model.fit(train_features[train_index], labels[train_index])
        valid_raw = model.predict(train_features[valid_index])
        oof[valid_index] = valid_raw
        test_predictions += model.predict(test_features) / baseline_config.n_splits

        rounded = np.clip(np.rint(valid_raw), 1, 6).astype(np.int8)
        fold_qwk = quadratic_weighted_kappa(labels[valid_index], rounded)
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "features": train_features.shape[1],
                "rounded_qwk": fold_qwk,
            }
        )
        print(f"[feature fold {fold}] rounded QWK={fold_qwk:.6f}", flush=True)

    return oof, test_predictions, fold_metrics


def optimize_blend(
    labels: np.ndarray,
    text_oof: np.ndarray,
    feature_oof: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Jointly tune text weight and ordinal thresholds on OOF predictions."""
    if not (labels.shape == text_oof.shape == feature_oof.shape):
        raise ValueError("Labels and both OOF prediction arrays must align")

    class_counts = np.bincount(labels, minlength=7)[1:7]
    cumulative = np.cumsum(class_counts)[:-1] / len(labels)

    # Use distribution-matching thresholds to cheaply find a robust initial
    # weight, then alternate exact threshold and weight coordinate searches.
    best_weight = 1.0
    best_thresholds = np.arange(1.5, 6.0, 1.0)
    best_score = -np.inf
    for weight in np.linspace(0.0, 1.0, 41):
        raw = weight * text_oof + (1.0 - weight) * feature_oof
        thresholds = np.quantile(raw, cumulative)
        score = quadratic_weighted_kappa(labels, apply_thresholds(raw, thresholds))
        if score > best_score:
            best_weight, best_thresholds, best_score = weight, thresholds, score

    for radius in (0.25, 0.10, 0.04, 0.015, 0.005):
        blended = best_weight * text_oof + (1.0 - best_weight) * feature_oof
        best_thresholds, best_score = optimize_thresholds(labels, blended)
        lower = max(0.0, best_weight - radius)
        upper = min(1.0, best_weight + radius)
        for weight in np.linspace(lower, upper, 31):
            raw = weight * text_oof + (1.0 - weight) * feature_oof
            score = quadratic_weighted_kappa(
                labels, apply_thresholds(raw, best_thresholds)
            )
            if score > best_score + 1e-12:
                best_weight, best_score = float(weight), score

    blended = best_weight * text_oof + (1.0 - best_weight) * feature_oof
    best_thresholds, best_score = optimize_thresholds(labels, blended)
    return best_weight, best_thresholds, best_score
