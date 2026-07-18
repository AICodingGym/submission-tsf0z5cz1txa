import unittest

import numpy as np

from src.transformer_data import DynamicPaddingCollator, head_tail_truncate


class TransformerDataTest(unittest.TestCase):
    def test_head_tail_truncation_preserves_both_ends(self) -> None:
        sequence = np.arange(20, dtype=np.int64)
        truncated = head_tail_truncate(sequence, 8)
        np.testing.assert_array_equal(
            truncated, np.asarray([0, 1, 2, 3, 16, 17, 18, 19])
        )

    def test_collator_dynamically_pads_and_returns_indices(self) -> None:
        rows = [
            {"input_ids": np.asarray([1, 2, 3]), "index": 7, "label": 2.0},
            {"input_ids": np.asarray([4, 5, 6, 7, 8]), "index": 9, "label": 4.0},
        ]
        batch = DynamicPaddingCollator(pad_token_id=0, max_length=16)(rows)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(batch["indices"].tolist(), [7, 9])
        self.assertEqual(batch["attention_mask"].sum(dim=1).tolist(), [3, 5])
        self.assertEqual(batch["labels"].tolist(), [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
