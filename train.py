from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.baseline import (
    BaselineConfig,
    apply_thresholds,
    load_competition_data,
    optimize_thresholds,
    quadratic_weighted_kappa,
    save_metrics,
    train_cross_validated_baseline,
    write_submission,
)


DEFAULT_ARCHIVE = Path("data/learning-agency-lab-automated-essay-scoring-2.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-safe TF-IDF/Ridge essay-scoring baseline."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--metrics", type=Path, default=Path("baseline_metrics.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BaselineConfig(
        n_splits=args.folds,
        seed=args.seed,
        ridge_alpha=args.alpha,
    )
    print(f"Loading data from {args.data}", flush=True)
    train_texts, labels, test_texts, test_ids = load_competition_data(args.data)
    print(
        f"Loaded {len(train_texts):,} train essays and {len(test_texts):,} test essays",
        flush=True,
    )

    oof, test_raw, fold_metrics = train_cross_validated_baseline(
        train_texts, labels, test_texts, config
    )
    rounded = np.clip(np.rint(oof), 1, 6).astype(np.int8)
    rounded_qwk = quadratic_weighted_kappa(labels, rounded)
    thresholds, optimized_qwk = optimize_thresholds(labels, oof)
    test_scores = apply_thresholds(test_raw, thresholds)

    write_submission(args.submission, test_ids, test_scores)
    save_metrics(
        args.metrics,
        config,
        fold_metrics,
        labels,
        rounded_qwk,
        optimized_qwk,
        thresholds,
        test_scores,
    )
    print("\nTraining complete", flush=True)
    print(f"OOF rounded QWK:   {rounded_qwk:.6f}", flush=True)
    print(f"OOF optimized QWK: {optimized_qwk:.6f}", flush=True)
    print(f"Thresholds: {np.round(thresholds, 6).tolist()}", flush=True)
    print(f"Wrote {args.submission} and {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
