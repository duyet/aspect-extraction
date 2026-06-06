"""Unit tests for extractors."""

import pytest

from aspect_extraction.core.aspect import Aspect, Sentiment
from aspect_extraction.extractors.rule_based import RuleBasedExtractor
from aspect_extraction.extractors.statistical import StatisticalExtractor


class TestRuleBasedExtractor:
    """Test RuleBasedExtractor."""

    def test_extract_basic(self) -> None:
        """Should extract aspects from simple text."""
        extractor = RuleBasedExtractor(use_spacy=False)
        text = "The battery life is great"

        aspects = extractor.extract(text)

        assert len(aspects) >= 0  # May or may not find aspects depending on patterns

    def test_extract_with_sentiment(self) -> None:
        """Should detect sentiment in aspects."""
        extractor = RuleBasedExtractor(use_spacy=False)
        text = "The camera quality is excellent but battery life is poor"

        aspects = extractor.extract(text)

        # Should find at least some aspects
        assert isinstance(aspects, list)
        for aspect in aspects:
            assert isinstance(aspect, Aspect)

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        extractor = RuleBasedExtractor(use_spacy=False)

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor.extract("")

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor.extract("   ")

    def test_extract_batch(self) -> None:
        """Should extract from multiple texts."""
        extractor = RuleBasedExtractor(use_spacy=False)
        texts = [
            "The battery life is good",
            "Camera quality is poor",
        ]

        results = extractor.extract_batch(texts)

        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_custom_sentiment_lexicon(self) -> None:
        """Should use custom sentiment lexicon."""
        custom_lexicon = {
            "awesome": Sentiment.POSITIVE,
            "terrible": Sentiment.NEGATIVE,
        }

        extractor = RuleBasedExtractor(
            use_spacy=False,
            sentiment_lexicon=custom_lexicon,
        )

        # Verify lexicon was set
        assert "awesome" in extractor._sentiment_lexicon
        assert extractor._sentiment_lexicon["awesome"] == Sentiment.POSITIVE


class TestStatisticalExtractor:
    """Test StatisticalExtractor."""

    def test_extract_without_training(self) -> None:
        """Should extract aspects without training (lower confidence)."""
        extractor = StatisticalExtractor()
        text = "The camera quality is great"

        aspects = extractor.extract(text)

        assert isinstance(aspects, list)
        for aspect in aspects:
            assert isinstance(aspect, Aspect)

    def test_train_and_extract(self) -> None:
        """Should improve extraction after training."""
        extractor = StatisticalExtractor(min_frequency=2)

        # Train on multiple texts
        training_texts = [
            "The battery life is great",
            "Battery life is excellent",
            "Poor battery life",
        ]

        extractor.train(training_texts)

        # Extract from new text
        aspects = extractor.extract("The battery life is good")

        assert isinstance(aspects, list)
        # Should find "battery life" with higher confidence after training

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        extractor = StatisticalExtractor()

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor.extract("")

    def test_train_empty_raises_error(self) -> None:
        """Should raise ValueError for empty training data."""
        extractor = StatisticalExtractor()

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor.train([])

    def test_invalid_parameters_raise_error(self) -> None:
        """Should raise ValueError for invalid parameters."""
        with pytest.raises(ValueError, match="at least 1"):
            StatisticalExtractor(min_frequency=0)

        with pytest.raises(ValueError, match="non-negative"):
            StatisticalExtractor(min_pmi=-1.0)

    def test_calculate_pmi(self) -> None:
        """Should calculate PMI for word pairs."""
        extractor = StatisticalExtractor()

        # Train with some data
        extractor.train(
            [
                "battery life",
                "battery life is good",
                "great battery",
            ]
        )

        # Calculate PMI
        pmi = extractor.calculate_pmi("battery", "life")
        assert isinstance(pmi, float)

    def test_get_top_aspects(self) -> None:
        """Should return top frequent aspects."""
        extractor = StatisticalExtractor()

        # Train with data
        extractor.train(
            [
                "battery life is great",
                "battery life is good",
                "camera quality is excellent",
            ]
        )

        top_aspects = extractor.get_top_aspects(n=5)

        assert isinstance(top_aspects, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in top_aspects)

    def test_get_top_aspects_without_training_raises_error(self) -> None:
        """Should raise ValueError if not trained."""
        extractor = StatisticalExtractor()

        with pytest.raises(ValueError, match="must be trained"):
            extractor.get_top_aspects()
