from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.baseline import (
    BaselineConfig,
    apply_thresholds,
    load_competition_data,
    optimize_thresholds,
    quadratic_weighted_kappa,
    train_cross_validated_baseline,
    write_submission,
)
from src.features import (
    FEATURE_NAMES,
    extract_feature_matrix,
    feature_target_correlations,
)
from src.stage2 import Stage2Config, optimize_blend, train_cross_validated_feature_model


DEFAULT_ARCHIVE = Path("data/learning-agency-lab-automated-essay-scoring-2.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and blend TF-IDF and engineered-feature essay models."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--submission", type=Path, default=Path("stage2_submission.csv")
    )
    parser.add_argument("--metrics", type=Path, default=Path("stage2_metrics.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=8.0)
    return parser.parse_args()


def optimized_score(labels: np.ndarray, predictions: np.ndarray) -> float:
    return optimize_thresholds(labels, predictions)[1]


def main() -> None:
    args = parse_args()
    baseline_config = BaselineConfig(
        n_splits=args.folds, seed=args.seed, ridge_alpha=args.alpha
    )
    stage2_config = Stage2Config()
    train_texts, labels, test_texts, test_ids = load_competition_data(args.data)
    print(
        f"Loaded {len(train_texts):,} train essays and {len(test_texts):,} test essays",
        flush=True,
    )

    print("\nExtracting engineered features", flush=True)
    train_features = extract_feature_matrix(train_texts)
    test_features = extract_feature_matrix(test_texts)
    print(f"Engineered feature shape={train_features.shape}", flush=True)

    feature_oof, feature_test, feature_fold_metrics = (
        train_cross_validated_feature_model(
            train_features,
            labels,
            test_features,
            baseline_config,
            stage2_config,
        )
    )
    print("\nTraining TF-IDF text branch", flush=True)
    text_oof, text_test, text_fold_metrics = train_cross_validated_baseline(
        train_texts, labels, test_texts, baseline_config
    )

    text_optimized_qwk = optimized_score(labels, text_oof)
    feature_optimized_qwk = optimized_score(labels, feature_oof)
    text_weight, thresholds, blended_qwk = optimize_blend(
        labels, text_oof, feature_oof
    )
    blended_test = text_weight * text_test + (1.0 - text_weight) * feature_test
    test_scores = apply_thresholds(blended_test, thresholds)
    write_submission(args.submission, test_ids, test_scores)

    rounded_blend = np.clip(
        np.rint(text_weight * text_oof + (1.0 - text_weight) * feature_oof), 1, 6
    ).astype(np.int8)
    metrics = {
        "baseline_config": asdict(baseline_config),
        "stage2_config": asdict(stage2_config),
        "feature_names": list(FEATURE_NAMES),
        "top_feature_target_correlations": feature_target_correlations(
            train_features, labels
        )[:15],
        "text_folds": text_fold_metrics,
        "feature_folds": feature_fold_metrics,
        "text_oof_optimized_qwk": text_optimized_qwk,
        "feature_oof_optimized_qwk": feature_optimized_qwk,
        "blend_oof_rounded_qwk": quadratic_weighted_kappa(labels, rounded_blend),
        "blend_oof_optimized_qwk": blended_qwk,
        "text_weight": text_weight,
        "feature_weight": 1.0 - text_weight,
        "optimized_thresholds": thresholds.tolist(),
        "submission_distribution": dict(
            sorted(Counter(map(int, test_scores)).items())
        ),
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\nStage 2 training complete", flush=True)
    print(f"Text-only optimized QWK:    {text_optimized_qwk:.6f}", flush=True)
    print(f"Feature-only optimized QWK: {feature_optimized_qwk:.6f}", flush=True)
    print(f"Blended optimized QWK:      {blended_qwk:.6f}", flush=True)
    print(
        f"Blend weights: text={text_weight:.4f}, features={1.0 - text_weight:.4f}",
        flush=True,
    )
    print(f"Thresholds: {np.round(thresholds, 6).tolist()}", flush=True)
    print(f"Wrote {args.submission} and {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
