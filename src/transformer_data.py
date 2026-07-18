from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def head_tail_truncate(input_ids: np.ndarray, max_length: int) -> np.ndarray:
    if len(input_ids) <= max_length:
        return input_ids
    head_length = max_length // 2
    tail_length = max_length - head_length
    return np.concatenate((input_ids[:head_length], input_ids[-tail_length:]))


class TokenCache:
    def __init__(self, directory: Path, split: str) -> None:
        self.directory = directory
        self.split = split
        self.input_ids = np.load(directory / f"{split}_input_ids.npy", mmap_mode="r")
        self.offsets = np.load(directory / f"{split}_offsets.npy", mmap_mode="r")
        self.essay_ids = np.load(directory / f"{split}_essay_ids.npy")
        if len(self.offsets) != len(self.essay_ids) + 1:
            raise ValueError(f"Invalid offsets for {split} token cache")

    def __len__(self) -> int:
        return len(self.essay_ids)

    def sequence(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.input_ids[start:end], dtype=np.int64)


class EssayTokenDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        cache: TokenCache,
        indices: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        index = int(self.indices[item])
        result: dict[str, object] = {
            "input_ids": self.cache.sequence(index),
            "index": index,
        }
        if self.labels is not None:
            result["label"] = float(self.labels[index])
        return result


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        sequences = [
            head_tail_truncate(np.asarray(row["input_ids"]), self.max_length)
            for row in rows
        ]
        batch_length = min(
            self.max_length,
            ((max(map(len, sequences)) + 7) // 8) * 8,
        )
        input_ids = torch.full(
            (len(rows), batch_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(rows), batch_length), dtype=torch.long)
        for row_index, sequence in enumerate(sequences):
            length = len(sequence)
            input_ids[row_index, :length] = torch.from_numpy(sequence.copy())
            attention_mask[row_index, :length] = 1
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "indices": torch.as_tensor([int(row["index"]) for row in rows]),
        }
        if "label" in rows[0]:
            batch["labels"] = torch.as_tensor(
                [float(row["label"]) for row in rows], dtype=torch.float32
            )
        return batch


def load_cache_metadata(directory: Path) -> dict[str, object]:
    return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
