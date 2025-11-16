"""Evaluation metrics for aspect extraction."""

from dataclasses import dataclass
from typing import List, Set

from aspect_extraction.core.aspect import Aspect


@dataclass
class EvaluationResult:
    """Container for evaluation results.

    Attributes:
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        f1: F1 score (0-1)
        true_positives: Number of correct predictions
        false_positives: Number of incorrect predictions
        false_negatives: Number of missed aspects
    """

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def __str__(self) -> str:
        """Return human-readable representation."""
        return (
            f"EvaluationResult(\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall:    {self.recall:.4f}\n"
            f"  F1 Score:  {self.f1:.4f}\n"
            f"  TP: {self.true_positives}, "
            f"FP: {self.false_positives}, "
            f"FN: {self.false_negatives}\n"
            f")"
        )


def calculate_precision(true_positives: int, false_positives: int) -> float:
    """Calculate precision score.

    Precision = TP / (TP + FP)

    Args:
        true_positives: Number of correct predictions
        false_positives: Number of incorrect predictions

    Returns:
        Precision score between 0 and 1

    Raises:
        ValueError: If counts are negative

    Example:
        >>> calculate_precision(true_positives=80, false_positives=20)
        0.8
    """
    if true_positives < 0 or false_positives < 0:
        raise ValueError("Counts must be non-negative")

    total = true_positives + false_positives
    if total == 0:
        return 0.0

    return true_positives / total


def calculate_recall(true_positives: int, false_negatives: int) -> float:
    """Calculate recall score.

    Recall = TP / (TP + FN)

    Args:
        true_positives: Number of correct predictions
        false_negatives: Number of missed predictions

    Returns:
        Recall score between 0 and 1

    Raises:
        ValueError: If counts are negative

    Example:
        >>> calculate_recall(true_positives=80, false_negatives=20)
        0.8
    """
    if true_positives < 0 or false_negatives < 0:
        raise ValueError("Counts must be non-negative")

    total = true_positives + false_negatives
    if total == 0:
        return 0.0

    return true_positives / total


def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        precision: Precision value between 0 and 1
        recall: Recall value between 0 and 1

    Returns:
        F1 score between 0 and 1

    Raises:
        ValueError: If precision or recall not in [0, 1]

    Example:
        >>> calculate_f1(precision=0.8, recall=0.9)
        0.8470588235294118
    """
    if not (0 <= precision <= 1 and 0 <= recall <= 1):
        raise ValueError("Precision and recall must be between 0 and 1")

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def normalize_aspect_text(text: str) -> str:
    """Normalize aspect text for comparison.

    Args:
        text: Aspect text

    Returns:
        Normalized text (lowercase, stripped)

    Example:
        >>> normalize_aspect_text("  Battery Life  ")
        'battery life'
    """
    return text.lower().strip()


def aspects_match(predicted: Aspect, ground_truth: Aspect, match_sentiment: bool = False) -> bool:
    """Check if two aspects match.

    Args:
        predicted: Predicted aspect
        ground_truth: Ground truth aspect
        match_sentiment: Whether to also match sentiment

    Returns:
        True if aspects match

    Example:
        >>> from aspect_extraction.core.aspect import Aspect, Sentiment
        >>> pred = Aspect(text="battery life", sentiment=Sentiment.POSITIVE)
        >>> truth = Aspect(text="Battery Life", sentiment=Sentiment.POSITIVE)
        >>> aspects_match(pred, truth)
        True
    """
    # Normalize and compare text
    text_match = normalize_aspect_text(predicted.text) == normalize_aspect_text(ground_truth.text)

    if not match_sentiment:
        return text_match

    # Also check sentiment if required
    return text_match and predicted.sentiment == ground_truth.sentiment


def evaluate_extraction(
    predicted: List[Aspect],
    ground_truth: List[Aspect],
    match_sentiment: bool = False,
) -> EvaluationResult:
    """Evaluate aspect extraction results.

    Args:
        predicted: List of predicted aspects
        ground_truth: List of ground truth aspects
        match_sentiment: Whether to require sentiment match

    Returns:
        Evaluation results with precision, recall, F1

    Example:
        >>> from aspect_extraction.core.aspect import Aspect
        >>> pred = [Aspect(text="battery"), Aspect(text="camera")]
        >>> truth = [Aspect(text="battery"), Aspect(text="screen")]
        >>> result = evaluate_extraction(pred, truth)
        >>> print(f"F1: {result.f1:.2f}")
        F1: 0.67
    """
    # Create sets of normalized aspect texts for matching
    pred_texts = {normalize_aspect_text(a.text) for a in predicted}
    truth_texts = {normalize_aspect_text(a.text) for a in ground_truth}

    if match_sentiment:
        # For sentiment matching, we need to consider both text and sentiment
        matched_aspects = 0

        for pred_aspect in predicted:
            for truth_aspect in ground_truth:
                if aspects_match(pred_aspect, truth_aspect, match_sentiment=True):
                    matched_aspects += 1
                    break

        true_positives = matched_aspects
    else:
        # Simple text matching
        true_positives = len(pred_texts & truth_texts)

    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives

    # Calculate metrics
    precision = calculate_precision(true_positives, false_positives)
    recall = calculate_recall(true_positives, false_negatives)
    f1 = calculate_f1(precision, recall)

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def evaluate_batch(
    predicted_batch: List[List[Aspect]],
    ground_truth_batch: List[List[Aspect]],
    match_sentiment: bool = False,
) -> EvaluationResult:
    """Evaluate multiple extraction results.

    Args:
        predicted_batch: List of predicted aspect lists
        ground_truth_batch: List of ground truth aspect lists
        match_sentiment: Whether to require sentiment match

    Returns:
        Aggregated evaluation results

    Raises:
        ValueError: If batches have different lengths

    Example:
        >>> pred_batch = [[Aspect(text="battery")], [Aspect(text="camera")]]
        >>> truth_batch = [[Aspect(text="battery")], [Aspect(text="screen")]]
        >>> result = evaluate_batch(pred_batch, truth_batch)
        >>> print(result.f1)
        0.5
    """
    if len(predicted_batch) != len(ground_truth_batch):
        raise ValueError("Predicted and ground truth batches must have same length")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for predicted, ground_truth in zip(predicted_batch, ground_truth_batch):
        result = evaluate_extraction(predicted, ground_truth, match_sentiment=match_sentiment)
        total_tp += result.true_positives
        total_fp += result.false_positives
        total_fn += result.false_negatives

    # Calculate aggregated metrics
    precision = calculate_precision(total_tp, total_fp)
    recall = calculate_recall(total_tp, total_fn)
    f1 = calculate_f1(precision, recall)

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )
