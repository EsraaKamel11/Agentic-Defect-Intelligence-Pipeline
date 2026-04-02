"""Unit tests for scoring/bayesian_updater.py."""
import pytest

from adip.scoring.bayesian_updater import (
    DEFAULT_WEIGHTS,
    bayesian_update,
    get_current_weights,
    _current_weights,
)


class TestBayesianUpdater:
    def setup_method(self):
        """Reset weights before each test."""
        import adip.scoring.bayesian_updater as bu
        bu._current_weights = dict(DEFAULT_WEIGHTS)
        bu._update_count = 0

    def test_weights_sum_to_one_after_update(self):
        bayesian_update("defect_frequency", True)
        weights = get_current_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_correct_prediction_increases_weight(self):
        before = get_current_weights()["defect_frequency"]
        bayesian_update("defect_frequency", True)
        after = get_current_weights()["defect_frequency"]
        assert after > before

    def test_incorrect_prediction_decreases_weight(self):
        before = get_current_weights()["defect_frequency"]
        bayesian_update("defect_frequency", False)
        after = get_current_weights()["defect_frequency"]
        assert after < before

    def test_unknown_feature_ignored(self):
        before = get_current_weights()
        bayesian_update("nonexistent_feature", True)
        assert get_current_weights() == before

    def test_weights_stay_positive(self):
        for _ in range(100):
            bayesian_update("recency_decay", False)
        weights = get_current_weights()
        assert all(v > 0 for v in weights.values())
