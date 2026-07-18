from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.baseline import optimize_thresholds, quadratic_weighted_kappa
from src.calibration import optimize_multimodel_blend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare completed stage-3 pilots.")
    parser.add_argument("run_names", nargs="+")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--stage2-predictions",
        type=Path,
        default=Path("artifacts/stage2_predictions.npz"),
    )
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage3"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/stage3/pilot_comparison.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2 = np.load(args.stage2_predictions)
    labels = stage2["labels"].astype(np.int8)
    fold_ids = stage2["fold_ids"].astype(np.int8)
    evaluation = fold_ids == args.fold
    valid_labels = labels[evaluation]
    valid_indices = np.flatnonzero(evaluation)
    stage2_fold_qwk = quadratic_weighted_kappa(
        valid_labels, stage2["crossfit_predictions"][evaluation].astype(np.int8)
    )
    results: list[dict[str, object]] = []
    for run_name in args.run_names:
        directory = args.root / run_name / f"fold_{args.fold}"
        prediction = np.load(directory / "predictions.npz")
        if not np.array_equal(prediction["valid_indices"], valid_indices):
            raise ValueError(f"Validation indices do not align for {run_name}")
        raw = prediction["valid_raw"]
        thresholds, optimized_qwk = optimize_thresholds(valid_labels, raw)
        rounded = np.clip(np.rint(raw), 1, 6).astype(np.int8)
        streams = np.vstack(
            (stage2["text_oof"][evaluation], stage2["feature_oof"][evaluation], raw)
        )
        weights, blend_thresholds, local_blend_qwk = optimize_multimodel_blend(
            valid_labels, streams
        )
        history = json.loads((directory / "history.json").read_text())
        results.append(
            {
                "run_name": run_name,
                "rmse": float(np.sqrt(np.mean((raw - valid_labels) ** 2))),
                "mae": float(np.mean(np.abs(raw - valid_labels))),
                "rounded_qwk": quadratic_weighted_kappa(valid_labels, rounded),
                "optimized_qwk": optimized_qwk,
                "optimized_improvement_over_stage2": optimized_qwk
                - stage2_fold_qwk,
                "reporting_thresholds": thresholds.tolist(),
                "local_three_way_blend_qwk_optimistic": local_blend_qwk,
                "local_three_way_weights": weights.tolist(),
                "local_blend_thresholds": blend_thresholds.tolist(),
                "history": history,
            }
        )
    payload = {
        "fold": args.fold,
        "stage2_fold_crossfit_qwk": stage2_fold_qwk,
        "pilot_gate": {
            "minimum_optimized_improvement": 0.005,
            "minimum_local_blend_improvement": 0.005,
            "warning": "Pilot thresholds and blend are fit on this fold; the full-CV gate remains authoritative.",
        },
        "warning": "Local pilot blend is optimistic; use only for relative selection.",
        "results": sorted(results, key=lambda row: row["optimized_qwk"], reverse=True),
    }
    for result in payload["results"]:
        result["local_blend_improvement_over_stage2"] = (
            result["local_three_way_blend_qwk_optimistic"] - stage2_fold_qwk
        )
        result["passes_pilot_gate"] = (
            result["optimized_improvement_over_stage2"] >= 0.005
            and result["local_blend_improvement_over_stage2"] >= 0.005
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
