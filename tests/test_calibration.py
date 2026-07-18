import unittest

import numpy as np

from src.calibration import (
    blend_raw,
    crossfit_multimodel_blend,
    optimize_multimodel_blend,
    simplex_grid,
)


class MultiModelCalibrationTest(unittest.TestCase):
    def test_simplex_grid_weights_are_valid(self) -> None:
        weights = simplex_grid(model_count=3, denominator=4)
        self.assertEqual(len(weights), 15)
        for item in weights:
            self.assertTrue(np.all(item >= 0))
            self.assertAlmostEqual(float(item.sum()), 1.0)

    def test_three_way_crossfit_returns_valid_scores(self) -> None:
        labels = np.tile(np.arange(1, 7), 10).astype(np.int8)
        offsets = np.linspace(-0.2, 0.2, len(labels))
        streams = np.vstack((labels + offsets, labels - offsets, labels + offsets / 2))
        folds = np.arange(len(labels), dtype=np.int8) % 5
        predictions, score, parameters = crossfit_multimodel_blend(
            labels, streams, folds
        )
        self.assertGreater(score, 0.95)
        self.assertEqual(len(parameters), 5)
        self.assertTrue(np.all((predictions >= 1) & (predictions <= 6)))

    def test_optimizer_and_blend_shapes(self) -> None:
        labels = np.tile(np.arange(1, 7), 4).astype(np.int8)
        streams = np.vstack((labels, labels + 0.1, labels - 0.1))
        weights, thresholds, score = optimize_multimodel_blend(labels, streams)
        raw = blend_raw(streams, weights)
        self.assertEqual(raw.shape, labels.shape)
        self.assertEqual(thresholds.shape, (5,))
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
