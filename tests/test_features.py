import unittest

import numpy as np

from src.baseline import apply_thresholds
from src.features import FEATURE_NAMES, extract_essay_features, extract_feature_matrix
from src.stage2 import crossfit_blend, optimize_blend


class FeatureExtractionTest(unittest.TestCase):
    def test_feature_vector_is_named_and_finite(self) -> None:
        text = (
            "First of all, I believe this is useful.\n\n"
            "However, evidence should support the argument!"
        )
        features = extract_essay_features(text)
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(features[FEATURE_NAMES.index("word_count")], 10)
        self.assertEqual(features[FEATURE_NAMES.index("paragraph_count")], 2)
        self.assertGreater(
            features[FEATURE_NAMES.index("transition_phrases_per_100_words")], 0
        )

    def test_matrix_preserves_input_order(self) -> None:
        matrix = extract_feature_matrix(
            ["One short sentence.", "A much longer sentence with more words."]
        )
        self.assertEqual(matrix.shape, (2, len(FEATURE_NAMES)))
        word_count = FEATURE_NAMES.index("word_count")
        self.assertLess(matrix[0, word_count], matrix[1, word_count])


class BlendTest(unittest.TestCase):
    def test_blend_optimizer_returns_valid_parameters(self) -> None:
        labels = np.tile(np.arange(1, 7), 10).astype(np.int8)
        text = labels + np.linspace(-0.2, 0.2, len(labels))
        features = labels + np.linspace(0.2, -0.2, len(labels))
        weight, thresholds, score = optimize_blend(labels, text, features)
        self.assertGreaterEqual(weight, 0.0)
        self.assertLessEqual(weight, 1.0)
        self.assertTrue(np.all(np.diff(thresholds) > 0))
        self.assertGreater(score, 0.95)
        predictions = apply_thresholds(
            weight * text + (1.0 - weight) * features, thresholds
        )
        self.assertTrue(np.all((predictions >= 1) & (predictions <= 6)))

    def test_crossfit_blend_covers_each_row_once(self) -> None:
        labels = np.tile(np.arange(1, 7), 10).astype(np.int8)
        text = labels + np.linspace(-0.2, 0.2, len(labels))
        features = labels + np.linspace(0.2, -0.2, len(labels))
        folds = np.arange(len(labels), dtype=np.int8) % 5
        predictions, score, parameters = crossfit_blend(
            labels, text, features, folds
        )
        self.assertEqual(len(parameters), 5)
        self.assertGreater(score, 0.95)
        self.assertTrue(np.all((predictions >= 1) & (predictions <= 6)))


if __name__ == "__main__":
    unittest.main()
