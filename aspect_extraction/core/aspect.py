"""Core data models for aspects and sentiments."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Sentiment(str, Enum):
    """Sentiment polarity associated with an aspect.

    Attributes:
        POSITIVE: Positive sentiment
        NEGATIVE: Negative sentiment
        NEUTRAL: Neutral sentiment
        MIXED: Mixed sentiment (both positive and negative)
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass(frozen=True)
class Aspect:
    """Represents an extracted aspect from text.

    An aspect is a feature, attribute, or entity mentioned in text,
    optionally with associated sentiment and position information.

    Attributes:
        text: The aspect text (e.g., "battery life", "camera quality")
        category: Optional category label (e.g., "performance", "design")
        sentiment: Optional sentiment polarity
        confidence: Confidence score between 0 and 1
        start_pos: Starting character position in original text
        end_pos: Ending character position in original text
        context: Optional surrounding context

    Example:
        >>> aspect = Aspect(
        ...     text="battery life",
        ...     category="hardware",
        ...     sentiment=Sentiment.NEGATIVE,
        ...     confidence=0.95,
        ...     start_pos=35,
        ...     end_pos=47
        ... )
        >>> print(aspect)
        Aspect(text='battery life', sentiment='negative')
    """

    text: str
    category: Optional[str] = None
    sentiment: Optional[Sentiment] = None
    confidence: float = 1.0
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    context: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate aspect attributes after initialization."""
        if not self.text or not self.text.strip():
            raise ValueError("Aspect text cannot be empty")

        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

        if self.start_pos is not None and self.end_pos is not None:
            if self.start_pos < 0 or self.end_pos < 0:
                raise ValueError("Positions must be non-negative")
            if self.start_pos >= self.end_pos:
                raise ValueError("start_pos must be less than end_pos")

    def __str__(self) -> str:
        """Return human-readable string representation."""
        parts = [f"Aspect(text='{self.text}'"]
        if self.category:
            parts.append(f"category='{self.category}'")
        if self.sentiment:
            parts.append(f"sentiment='{self.sentiment.value}'")
        if self.confidence < 1.0:
            parts.append(f"confidence={self.confidence:.2f}")
        return ", ".join(parts) + ")"

    def to_dict(self) -> dict:
        """Convert aspect to dictionary representation.

        Returns:
            Dictionary containing all aspect attributes
        """
        return {
            "text": self.text,
            "category": self.category,
            "sentiment": self.sentiment.value if self.sentiment else None,
            "confidence": self.confidence,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "context": self.context,
        }
