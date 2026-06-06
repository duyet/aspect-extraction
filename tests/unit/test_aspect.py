"""Unit tests for Aspect data model."""

import pytest

from aspect_extraction.core.aspect import Aspect, Sentiment


class TestAspect:
    """Test Aspect class."""

    def test_create_basic_aspect(self) -> None:
        """Should create aspect with minimal attributes."""
        aspect = Aspect(text="battery life")

        assert aspect.text == "battery life"
        assert aspect.category is None
        assert aspect.sentiment is None
        assert aspect.confidence == 1.0
        assert aspect.start_pos is None
        assert aspect.end_pos is None

    def test_create_full_aspect(self) -> None:
        """Should create aspect with all attributes."""
        aspect = Aspect(
            text="camera quality",
            category="hardware",
            sentiment=Sentiment.POSITIVE,
            confidence=0.95,
            start_pos=10,
            end_pos=24,
            context="The camera quality is great",
        )

        assert aspect.text == "camera quality"
        assert aspect.category == "hardware"
        assert aspect.sentiment == Sentiment.POSITIVE
        assert aspect.confidence == 0.95
        assert aspect.start_pos == 10
        assert aspect.end_pos == 24
        assert aspect.context == "The camera quality is great"

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Aspect(text="")

        with pytest.raises(ValueError, match="cannot be empty"):
            Aspect(text="   ")

    def test_invalid_confidence_raises_error(self) -> None:
        """Should raise ValueError for invalid confidence."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            Aspect(text="test", confidence=-0.1)

        with pytest.raises(ValueError, match="between 0 and 1"):
            Aspect(text="test", confidence=1.5)

    def test_invalid_positions_raise_error(self) -> None:
        """Should raise ValueError for invalid positions."""
        with pytest.raises(ValueError, match="non-negative"):
            Aspect(text="test", start_pos=-1, end_pos=5)

        with pytest.raises(ValueError, match="less than end_pos"):
            Aspect(text="test", start_pos=10, end_pos=5)

    def test_str_representation(self) -> None:
        """Should produce correct string representation."""
        aspect = Aspect(
            text="battery",
            category="hardware",
            sentiment=Sentiment.NEGATIVE,
            confidence=0.85,
        )

        str_repr = str(aspect)
        assert "battery" in str_repr
        assert "hardware" in str_repr
        assert "negative" in str_repr
        assert "0.85" in str_repr

    def test_to_dict(self) -> None:
        """Should convert to dictionary correctly."""
        aspect = Aspect(
            text="screen",
            sentiment=Sentiment.POSITIVE,
            confidence=0.9,
            start_pos=0,
            end_pos=6,
        )

        aspect_dict = aspect.to_dict()

        assert aspect_dict["text"] == "screen"
        assert aspect_dict["sentiment"] == "positive"
        assert aspect_dict["confidence"] == 0.9
        assert aspect_dict["start_pos"] == 0
        assert aspect_dict["end_pos"] == 6


class TestSentiment:
    """Test Sentiment enum."""

    def test_sentiment_values(self) -> None:
        """Should have correct sentiment values."""
        assert Sentiment.POSITIVE.value == "positive"
        assert Sentiment.NEGATIVE.value == "negative"
        assert Sentiment.NEUTRAL.value == "neutral"
        assert Sentiment.MIXED.value == "mixed"

    def test_sentiment_comparison(self) -> None:
        """Should support sentiment comparison."""
        assert Sentiment.POSITIVE == Sentiment.POSITIVE
        assert Sentiment.POSITIVE != Sentiment.NEGATIVE
