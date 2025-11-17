"""Unit tests for evaluation metrics."""

import pytest

from aspect_extraction.core.aspect import Aspect, Sentiment
from aspect_extraction.evaluation.metrics import (
    calculate_f1,
    calculate_precision,
    calculate_recall,
    evaluate_extraction,
    evaluate_batch,
)


class TestMetricsFunctions:
    """Test individual metric functions."""

    def test_calculate_precision(self) -> None:
        """Should calculate precision correctly."""
        assert calculate_precision(80, 20) == 0.8
        assert calculate_precision(100, 0) == 1.0
        assert calculate_precision(0, 10) == 0.0
        assert calculate_precision(0, 0) == 0.0

    def test_calculate_precision_invalid_input(self) -> None:
        """Should raise ValueError for negative inputs."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_precision(-1, 10)

        with pytest.raises(ValueError, match="non-negative"):
            calculate_precision(10, -1)

    def test_calculate_recall(self) -> None:
        """Should calculate recall correctly."""
        assert calculate_recall(80, 20) == 0.8
        assert calculate_recall(100, 0) == 1.0
        assert calculate_recall(0, 10) == 0.0
        assert calculate_recall(0, 0) == 0.0

    def test_calculate_recall_invalid_input(self) -> None:
        """Should raise ValueError for negative inputs."""
        with pytest.raises(ValueError, match="non-negative"):
            calculate_recall(-1, 10)

    def test_calculate_f1(self) -> None:
        """Should calculate F1 score correctly."""
        # Perfect precision and recall
        assert calculate_f1(1.0, 1.0) == 1.0

        # Equal precision and recall
        assert calculate_f1(0.5, 0.5) == 0.5

        # Different values
        f1 = calculate_f1(0.8, 0.6)
        assert abs(f1 - 0.6857) < 0.001

        # Zero case
        assert calculate_f1(0.0, 0.0) == 0.0

    def test_calculate_f1_invalid_input(self) -> None:
        """Should raise ValueError for invalid inputs."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_f1(-0.1, 0.5)

        with pytest.raises(ValueError, match="between 0 and 1"):
            calculate_f1(0.5, 1.5)


class TestEvaluateExtraction:
    """Test evaluate_extraction function."""

    def test_perfect_match(self) -> None:
        """Should return perfect scores for identical predictions."""
        predicted = [
            Aspect(text="battery"),
            Aspect(text="camera"),
        ]
        ground_truth = [
            Aspect(text="battery"),
            Aspect(text="camera"),
        ]

        result = evaluate_extraction(predicted, ground_truth)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_partial_match(self) -> None:
        """Should handle partial matches correctly."""
        predicted = [
            Aspect(text="battery"),
            Aspect(text="camera"),
            Aspect(text="screen"),
        ]
        ground_truth = [
            Aspect(text="battery"),
            Aspect(text="camera"),
        ]

        result = evaluate_extraction(predicted, ground_truth)

        assert result.precision == 2 / 3
        assert result.recall == 1.0
        assert result.true_positives == 2
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_no_match(self) -> None:
        """Should return zero scores for no matches."""
        predicted = [Aspect(text="battery")]
        ground_truth = [Aspect(text="camera")]

        result = evaluate_extraction(predicted, ground_truth)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_case_insensitive_matching(self) -> None:
        """Should match aspects case-insensitively."""
        predicted = [Aspect(text="Battery Life")]
        ground_truth = [Aspect(text="battery life")]

        result = evaluate_extraction(predicted, ground_truth)

        assert result.true_positives == 1
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_sentiment_matching(self) -> None:
        """Should match sentiment when requested."""
        predicted = [
            Aspect(text="battery", sentiment=Sentiment.POSITIVE),
            Aspect(text="camera", sentiment=Sentiment.NEGATIVE),
        ]
        ground_truth = [
            Aspect(text="battery", sentiment=Sentiment.POSITIVE),
            Aspect(text="camera", sentiment=Sentiment.POSITIVE),
        ]

        # Without sentiment matching
        result1 = evaluate_extraction(predicted, ground_truth, match_sentiment=False)
        assert result1.true_positives == 2

        # With sentiment matching
        result2 = evaluate_extraction(predicted, ground_truth, match_sentiment=True)
        assert result2.true_positives == 1  # Only battery matches


class TestEvaluateBatch:
    """Test evaluate_batch function."""

    def test_batch_evaluation(self) -> None:
        """Should evaluate multiple predictions correctly."""
        predicted_batch = [
            [Aspect(text="battery"), Aspect(text="camera")],
            [Aspect(text="screen")],
        ]
        ground_truth_batch = [
            [Aspect(text="battery"), Aspect(text="camera")],
            [Aspect(text="screen"), Aspect(text="price")],
        ]

        result = evaluate_batch(predicted_batch, ground_truth_batch)

        assert result.true_positives == 3
        assert result.false_positives == 0
        assert result.false_negatives == 1

    def test_batch_length_mismatch_raises_error(self) -> None:
        """Should raise ValueError for mismatched batch lengths."""
        predicted_batch = [[Aspect(text="battery")]]
        ground_truth_batch = [
            [Aspect(text="battery")],
            [Aspect(text="camera")],
        ]

        with pytest.raises(ValueError, match="same length"):
            evaluate_batch(predicted_batch, ground_truth_batch)
