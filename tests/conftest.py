"""Pytest configuration and shared fixtures."""

import pytest

from aspect_extraction.core.aspect import Aspect, Sentiment


@pytest.fixture
def sample_aspects():
    """Provide sample aspects for testing."""
    return [
        Aspect(
            text="battery life",
            sentiment=Sentiment.NEGATIVE,
            confidence=0.95,
            start_pos=35,
            end_pos=47,
        ),
        Aspect(
            text="camera quality",
            sentiment=Sentiment.POSITIVE,
            confidence=0.92,
            start_pos=4,
            end_pos=18,
        ),
    ]


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "The camera quality is excellent but battery life is poor",
        "Great screen display and fast performance",
        "Terrible build quality and frequent crashes",
    ]


@pytest.fixture
def sample_review_texts():
    """Provide sample review texts."""
    return [
        "The food quality is excellent and service is outstanding",
        "Great ambiance but the price is too high",
        "Poor hygiene and rude waiters",
    ]
