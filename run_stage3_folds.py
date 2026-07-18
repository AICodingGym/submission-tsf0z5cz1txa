from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage-3 folds sequentially with GPU temperature protection."
    )
    parser.add_argument("--run-name", default="modernbert_reg_1024")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--max-temperature", type=int, default=84)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--cooldown-temperature", type=int, default=65)
    parser.add_argument("--cooldown-timeout", type=int, default=900)
    parser.add_argument("--mode", choices=["regression", "multitask"], default="regression")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/stage3"))
    return parser.parse_args()


def gpu_temperature(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def wait_until_cool(args: argparse.Namespace) -> None:
    started = time.monotonic()
    while True:
        temperature = gpu_temperature(args.gpu)
        if temperature <= args.cooldown_temperature:
            print(
                f"GPU {args.gpu} is ready at {temperature} C ",
                f"(cooldown target {args.cooldown_temperature} C).",
                flush=True,
            )
            return
        elapsed = time.monotonic() - started
        if elapsed >= args.cooldown_timeout:
            raise TimeoutError(
                f"GPU {args.gpu} stayed above the cooldown target for "
                f"{args.cooldown_timeout} seconds"
            )
        print(
            f"GPU {args.gpu} is {temperature} C; waiting for cooldown.",
            flush=True,
        )
        time.sleep(min(args.poll_seconds, 60))


def completed_fold_matches(args: argparse.Namespace, fold: int) -> bool:
    directory = args.output_root / args.run_name / f"fold_{fold}"
    config_path = directory / "config.json"
    predictions_path = directory / "predictions.npz"
    history_path = directory / "history.json"
    if not all(path.is_file() for path in (config_path, predictions_path, history_path)):
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "fold": fold,
        "run_name": args.run_name,
        "mode": args.mode,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Completed fold {fold} has incompatible configuration: {mismatches}"
        )
    with np.load(predictions_path) as predictions:
        required = {"valid_indices", "valid_raw", "test_indices", "test_raw"}
        if not required.issubset(predictions.files):
            raise ValueError(
                f"Completed fold {fold} is missing prediction arrays: "
                f"{sorted(required - set(predictions.files))}"
            )
        if len(predictions["valid_indices"]) != len(predictions["valid_raw"]):
            raise ValueError(f"Completed fold {fold} has misaligned validation output")
        if len(predictions["test_indices"]) != len(predictions["test_raw"]):
            raise ValueError(f"Completed fold {fold} has misaligned test output")
        if not (
            np.isfinite(predictions["valid_raw"]).all()
            and np.isfinite(predictions["test_raw"]).all()
        ):
            raise ValueError(f"Completed fold {fold} contains non-finite predictions")
    return True


def stop_process(process: subprocess.Popen[bytes]) -> None:
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_fold(args: argparse.Namespace, fold: int) -> None:
    wait_until_cool(args)
    command = [
        sys.executable,
        "train_stage3.py",
        "--fold",
        str(fold),
        "--run-name",
        args.run_name,
        "--mode",
        args.mode,
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--output-root",
        str(args.output_root),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    print(f"Starting fold {fold}: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    peak_temperature = 0
    while process.poll() is None:
        time.sleep(args.poll_seconds)
        temperature = gpu_temperature(args.gpu)
        peak_temperature = max(peak_temperature, temperature)
        print(
            f"Fold {fold}: GPU {args.gpu} temperature {temperature} C ",
            f"(peak {peak_temperature} C).",
            flush=True,
        )
        if temperature >= args.max_temperature:
            stop_process(process)
            raise RuntimeError(
                f"Stopped fold {fold}: GPU temperature reached {temperature} C "
                f"(limit {args.max_temperature} C)"
            )
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)
    if not completed_fold_matches(args, fold):
        raise RuntimeError(f"Fold {fold} exited successfully without complete artifacts")
    print(f"Completed fold {fold}; peak GPU temperature {peak_temperature} C.", flush=True)


def main() -> None:
    args = parse_args()
    if any(fold not in range(5) for fold in args.folds):
        raise ValueError("Fold IDs must be in [0, 4]")
    if args.max_temperature <= args.cooldown_temperature:
        raise ValueError("max-temperature must exceed cooldown-temperature")
    for fold in args.folds:
        if completed_fold_matches(args, fold):
            print(f"Skipping completed fold {fold}.", flush=True)
            continue
        run_fold(args, fold)
    print("All requested stage-3 folds are complete.", flush=True)


if __name__ == "__main__":
    main()
