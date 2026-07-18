from __future__ import annotations

import argparse
from pathlib import Path

from src.baseline import load_competition_data, load_train_ids
from src.folds import (
    build_fixed_folds,
    build_fold_audit,
    detect_duplicate_groups,
    infer_topic_clusters,
    write_fold_manifest,
    write_json,
)


DEFAULT_ARCHIVE = Path("data/learning-agency-lab-automated-essay-scoring-2.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic stage-3 folds.")
    parser.add_argument("--data", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--folds-output", type=Path, default=Path("artifacts/folds.csv")
    )
    parser.add_argument(
        "--audit-output", type=Path, default=Path("artifacts/fold_audit.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    parser.add_argument("--topic-clusters", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_texts, labels, _, _ = load_competition_data(args.data)
    train_ids = load_train_ids(args.data)
    print(f"Auditing {len(train_texts):,} essays", flush=True)

    duplicate_groups, duplicate_audit = detect_duplicate_groups(
        train_texts, args.near_duplicate_threshold
    )
    print(f"Duplicate audit: {duplicate_audit}", flush=True)
    topic_clusters = infer_topic_clusters(
        train_texts, seed=args.seed, n_clusters=args.topic_clusters
    )
    fold_ids = build_fixed_folds(
        labels, duplicate_groups, n_splits=5, seed=args.seed
    )
    audit = build_fold_audit(
        labels, fold_ids, duplicate_groups, topic_clusters, duplicate_audit
    )
    write_fold_manifest(
        args.folds_output,
        train_ids,
        labels,
        fold_ids,
        duplicate_groups,
        topic_clusters,
    )
    write_json(args.audit_output, audit)
    print(f"Wrote {args.folds_output} and {args.audit_output}", flush=True)


if __name__ == "__main__":
    main()
