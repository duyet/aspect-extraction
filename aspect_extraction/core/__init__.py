"""Core module containing base classes and interfaces for aspect extraction."""

from aspect_extraction.core.aspect import Aspect, Sentiment
from aspect_extraction.core.extractor import AspectExtractor

__all__ = ["Aspect", "Sentiment", "AspectExtractor"]
