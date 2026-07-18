from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion


@dataclass(frozen=True)
class BaselineConfig:
    n_splits: int = 5
    seed: int = 42
    ridge_alpha: float = 8.0
    word_max_features: int = 60_000
    char_max_features: int = 80_000
    min_score: int = 1
    max_score: int = 6


def _read_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with archive.open(name) as raw_file:
        with io.TextIOWrapper(raw_file, encoding="utf-8", newline="") as text_file:
            return list(csv.DictReader(text_file))


def load_competition_data(
    archive_path: Path,
) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Load train/test CSV files directly from the competition archive."""
    with zipfile.ZipFile(archive_path) as archive:
        train_rows = _read_csv_from_zip(archive, "train.csv")
        test_rows = _read_csv_from_zip(archive, "test.csv")

    expected_train = {"essay_id", "full_text", "score"}
    expected_test = {"essay_id", "full_text"}
    if not train_rows or set(train_rows[0]) != expected_train:
        raise ValueError(f"Unexpected train.csv columns; expected {sorted(expected_train)}")
    if not test_rows or set(test_rows[0]) != expected_test:
        raise ValueError(f"Unexpected test.csv columns; expected {sorted(expected_test)}")

    train_texts = [row["full_text"] for row in train_rows]
    test_texts = [row["full_text"] for row in test_rows]
    test_ids = [row["essay_id"] for row in test_rows]
    labels = np.asarray([int(row["score"]) for row in train_rows], dtype=np.int8)

    if any(not text.strip() for text in train_texts + test_texts):
        raise ValueError("Empty essay text found")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate essay_id found in test.csv")
    if not np.all((labels >= 1) & (labels <= 6)):
        raise ValueError("Training scores must be integers from 1 through 6")
    return train_texts, labels, test_texts, test_ids


def load_train_ids(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        train_rows = _read_csv_from_zip(archive, "train.csv")
    train_ids = [row["essay_id"] for row in train_rows]
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Duplicate essay_id found in train.csv")
    return train_ids


def cross_validation_splits(
    labels: np.ndarray,
    n_splits: int,
    seed: int,
    fold_ids: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if fold_ids is None:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        placeholder = np.zeros(len(labels), dtype=np.int8)
        return list(splitter.split(placeholder, labels))
    fold_ids = np.asarray(fold_ids)
    if fold_ids.shape != labels.shape:
        raise ValueError("fold_ids must have the same shape as labels")
    expected = set(range(n_splits))
    if set(map(int, np.unique(fold_ids))) != expected:
        raise ValueError(f"fold_ids must contain exactly {sorted(expected)}")
    all_indices = np.arange(len(labels))
    return [
        (all_indices[fold_ids != fold], all_indices[fold_ids == fold])
        for fold in range(n_splits)
    ]


def build_vectorizer(config: BaselineConfig) -> FeatureUnion:
    """Create complementary word- and character-level TF-IDF features."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        min_df=2,
        max_df=0.995,
        max_features=config.word_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        min_df=3,
        max_features=config.char_max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)], n_jobs=1)


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_score: int = 1,
    max_score: int = 6,
) -> float:
    """Compute quadratic weighted kappa without additional metric dependencies."""
    true = np.asarray(y_true, dtype=np.int64) - min_score
    pred = np.asarray(y_pred, dtype=np.int64) - min_score
    n_ratings = max_score - min_score + 1
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if np.any((true < 0) | (true >= n_ratings)):
        raise ValueError("y_true contains scores outside the configured range")
    if np.any((pred < 0) | (pred >= n_ratings)):
        raise ValueError("y_pred contains scores outside the configured range")

    observed = np.bincount(
        true * n_ratings + pred, minlength=n_ratings * n_ratings
    ).reshape(n_ratings, n_ratings)
    true_hist = np.bincount(true, minlength=n_ratings)
    pred_hist = np.bincount(pred, minlength=n_ratings)
    expected = np.outer(true_hist, pred_hist) / len(true)
    indices = np.arange(n_ratings)
    weights = (indices[:, None] - indices[None, :]) ** 2 / (n_ratings - 1) ** 2
    denominator = float(np.sum(weights * expected))
    if denominator == 0:
        return 1.0 if np.array_equal(true, pred) else 0.0
    return 1.0 - float(np.sum(weights * observed)) / denominator


def apply_thresholds(raw_predictions: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    ordered = np.asarray(list(thresholds), dtype=np.float64)
    if ordered.shape != (5,) or np.any(np.diff(ordered) <= 0):
        raise ValueError("Exactly five strictly increasing thresholds are required")
    return np.digitize(raw_predictions, ordered).astype(np.int8) + 1


def optimize_thresholds(
    y_true: np.ndarray, raw_predictions: np.ndarray
) -> tuple[np.ndarray, float]:
    """Coordinate-search five cut points using only out-of-fold predictions."""
    class_counts = np.bincount(y_true, minlength=7)[1:7]
    cumulative = np.cumsum(class_counts)[:-1] / len(y_true)
    quantile_start = np.quantile(raw_predictions, cumulative)
    starts = [quantile_start, np.arange(1.5, 6.0, 1.0)]

    def score(thresholds: np.ndarray) -> float:
        return quadratic_weighted_kappa(y_true, apply_thresholds(raw_predictions, thresholds))

    thresholds = max(starts, key=score).astype(np.float64)
    best_score = score(thresholds)
    # Coarse-to-fine coordinate ascent is deterministic and works well for the
    # discontinuous QWK objective, where gradient optimizers tend to stall.
    for radius in (0.60, 0.30, 0.15, 0.07, 0.03, 0.015, 0.007):
        for _ in range(3):
            improved = False
            for index in range(5):
                lower = thresholds[index - 1] + 1e-4 if index else 0.5
                upper = thresholds[index + 1] - 1e-4 if index < 4 else 6.5
                candidates = np.linspace(
                    max(lower, thresholds[index] - radius),
                    min(upper, thresholds[index] + radius),
                    25,
                )
                for candidate in candidates:
                    trial = thresholds.copy()
                    trial[index] = candidate
                    trial_score = score(trial)
                    if trial_score > best_score + 1e-12:
                        thresholds, best_score = trial, trial_score
                        improved = True
            if not improved:
                break
    return thresholds, best_score


def train_cross_validated_baseline(
    train_texts: list[str],
    labels: np.ndarray,
    test_texts: list[str],
    config: BaselineConfig,
    fold_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    oof = np.zeros(len(train_texts), dtype=np.float64)
    test_predictions = np.zeros(len(test_texts), dtype=np.float64)
    fold_metrics: list[dict[str, float | int]] = []
    texts = np.asarray(train_texts, dtype=object)

    splits = cross_validation_splits(
        labels, config.n_splits, config.seed, fold_ids=fold_ids
    )
    for fold, (train_index, valid_index) in enumerate(splits, 1):
        print(f"\n[fold {fold}/{config.n_splits}] fitting TF-IDF", flush=True)
        vectorizer = build_vectorizer(config)
        train_features = vectorizer.fit_transform(texts[train_index])
        valid_features = vectorizer.transform(texts[valid_index])
        test_features = vectorizer.transform(test_texts)
        print(
            f"[fold {fold}] feature shape={train_features.shape}, "
            f"nnz={train_features.nnz:,}",
            flush=True,
        )

        model = Ridge(
            alpha=config.ridge_alpha,
            solver="lsqr",
            tol=1e-4,
            max_iter=3_000,
        )
        model.fit(train_features, labels[train_index].astype(np.float32))
        valid_raw = model.predict(valid_features)
        oof[valid_index] = valid_raw
        test_predictions += model.predict(test_features) / config.n_splits

        valid_labels = np.clip(np.rint(valid_raw), 1, 6).astype(np.int8)
        fold_qwk = quadratic_weighted_kappa(labels[valid_index], valid_labels)
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "valid_rows": len(valid_index),
                "features": train_features.shape[1],
                "rounded_qwk": fold_qwk,
            }
        )
        print(f"[fold {fold}] rounded QWK={fold_qwk:.6f}", flush=True)

        del vectorizer, model, train_features, valid_features, test_features

    return oof, test_predictions, fold_metrics


def write_submission(path: Path, test_ids: list[str], predictions: np.ndarray) -> None:
    predictions = np.asarray(predictions)
    if len(test_ids) != len(predictions):
        raise ValueError("Test IDs and predictions have different lengths")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Test essay IDs are not unique")
    if predictions.dtype.kind not in "iu":
        raise ValueError("Submission scores must have an integer dtype")
    if not np.all((predictions >= 1) & (predictions <= 6)):
        raise ValueError("Submission scores must be between 1 and 6")

    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["essay_id", "score"])
        writer.writerows(zip(test_ids, predictions, strict=True))


def save_metrics(
    path: Path,
    config: BaselineConfig,
    fold_metrics: list[dict[str, float | int]],
    labels: np.ndarray,
    rounded_qwk: float,
    optimized_qwk: float,
    thresholds: np.ndarray,
    submission_predictions: np.ndarray,
) -> None:
    payload = {
        "config": asdict(config),
        "folds": fold_metrics,
        "label_distribution": dict(sorted(Counter(map(int, labels)).items())),
        "oof_rounded_qwk": rounded_qwk,
        "oof_optimized_qwk": optimized_qwk,
        "optimized_thresholds": thresholds.tolist(),
        "submission_distribution": dict(
            sorted(Counter(map(int, submission_predictions)).items())
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
