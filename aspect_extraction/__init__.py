"""Aspect Extraction - Production-ready multi-model aspect extraction system.

This package provides a comprehensive toolkit for extracting aspects (features,
attributes, entities) from text using various methods including rule-based,
statistical, and transformer-based approaches.

Example:
    >>> from aspect_extraction import extract_aspects
    >>> text = "The camera quality is great but battery life is poor"
    >>> aspects = extract_aspects(text)
    >>> for aspect in aspects:
    ...     print(f"{aspect.text}: {aspect.sentiment}")
    camera quality: positive
    battery life: negative
"""

__version__ = "0.1.0"
__author__ = "aspect-extraction contributors"
__all__ = [
    "extract_aspects",
    "Aspect",
    "AspectExtractor",
    "RuleBasedExtractor",
    "TransformerExtractor",
]

from aspect_extraction.core.aspect import Aspect
from aspect_extraction.core.extractor import AspectExtractor
from aspect_extraction.core.factory import extract_aspects
from aspect_extraction.extractors.rule_based import RuleBasedExtractor
from aspect_extraction.extractors.transformer_based import TransformerExtractor
