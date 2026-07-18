from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.baseline import load_competition_data, load_train_ids


DEFAULT_ARCHIVE = Path("data/learning-agency-lab-automated-essay-scoring-2.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache untruncated essay token IDs.")
    parser.add_argument("--data", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--model-path", type=Path, default=Path("models/modernbert-base")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/token_cache/modernbert")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_split(
    output: Path,
    split: str,
    essay_ids: list[str],
    texts: list[str],
    tokenizer: AutoTokenizer,
) -> np.ndarray:
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    lengths = np.asarray([len(sequence) for sequence in encoded], dtype=np.int32)
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    flat = np.empty(int(offsets[-1]), dtype=np.int32)
    for index, sequence in enumerate(encoded):
        flat[offsets[index] : offsets[index + 1]] = sequence
    np.save(output / f"{split}_input_ids.npy", flat, allow_pickle=False)
    np.save(output / f"{split}_offsets.npy", offsets, allow_pickle=False)
    np.save(
        output / f"{split}_essay_ids.npy",
        np.asarray(essay_ids),
        allow_pickle=False,
    )
    return lengths


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_texts, _, test_texts, test_ids = load_competition_data(args.data)
    train_ids = load_train_ids(args.data)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, use_fast=True
    )
    train_lengths = cache_split(
        args.output, "train", train_ids, train_texts, tokenizer
    )
    test_lengths = cache_split(args.output, "test", test_ids, test_texts, tokenizer)
    all_lengths = np.concatenate((train_lengths, test_lengths))
    candidates = [512, 768, 1024, 1280, 1536, 2048]
    metadata = {
        "model_path": str(args.model_path),
        "model_sha256": sha256_file(args.model_path / "model.safetensors"),
        "tokenizer_sha256": sha256_file(args.model_path / "tokenizer.json"),
        "pad_token_id": tokenizer.pad_token_id,
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "token_length_quantiles": {
            str(quantile): int(np.quantile(all_lengths, quantile))
            for quantile in (0, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
        },
        "candidate_coverage": {
            str(length): float(np.mean(all_lengths <= length))
            for length in candidates
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
