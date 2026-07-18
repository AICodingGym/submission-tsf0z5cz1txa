import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.folds import build_fixed_folds, load_fold_manifest, write_fold_manifest


class FoldManifestTest(unittest.TestCase):
    def test_duplicate_groups_are_never_split(self) -> None:
        labels = np.tile(np.arange(1, 7), 20).astype(np.int8)
        groups = np.arange(len(labels), dtype=np.int32)
        groups[1] = groups[0]
        groups[31] = groups[30]
        folds = build_fixed_folds(labels, groups, n_splits=5, seed=42)
        self.assertEqual(set(map(int, folds)), set(range(5)))
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(folds[30], folds[31])

    def test_manifest_round_trip_uses_essay_id_order(self) -> None:
        essay_ids = ["essay_b", "essay_a", "essay_c", "essay_d", "essay_e"]
        labels = np.asarray([2, 1, 3, 4, 5], dtype=np.int8)
        folds = np.arange(5, dtype=np.int8)
        groups = np.arange(5, dtype=np.int32)
        topics = np.asarray([1, 0, 1, 2, 2], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            write_fold_manifest(path, essay_ids, labels, folds, groups, topics)
            reordered = list(reversed(essay_ids))
            loaded = load_fold_manifest(path, reordered)
        expected = np.asarray([4, 3, 2, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(loaded, expected)


if __name__ == "__main__":
    unittest.main()
