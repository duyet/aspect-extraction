"""Aspect Extraction - Production-ready multi-model aspect extraction system.

This package provides a comprehensive toolkit for extracting aspects (features,
attributes, entities) from text using various methods including rule-based,
statistical, and transformer-based approaches.

Features high-performance Rust backend for 10-100x speedup (optional).

Example:
    >>> from aspect_extraction import extract_aspects
    >>> text = "The camera quality is great but battery life is poor"
    >>> # Auto-detects Rust backend if available
    >>> aspects = extract_aspects(text)
    >>> for aspect in aspects:
    ...     print(f"{aspect.text}: {aspect.sentiment}")
    camera quality: positive
    battery life: negative

    >>> # Force Rust backend for maximum performance
    >>> aspects = extract_aspects(text, backend="rust")

    >>> # Check if Rust is available
    >>> from aspect_extraction import is_rust_available
    >>> if is_rust_available():
    ...     print("🚀 Rust acceleration enabled!")
"""

__version__ = "0.1.0"
__author__ = "aspect-extraction contributors"
__all__ = [
    "extract_aspects",
    "extract_aspects_batch",
    "create_extractor",
    "is_rust_available",
    "Aspect",
    "AspectExtractor",
    "RuleBasedExtractor",
    "TransformerExtractor",
]

from aspect_extraction.core.aspect import Aspect
from aspect_extraction.core.extractor import AspectExtractor
from aspect_extraction.core.factory import (
    create_extractor,
    extract_aspects,
    extract_aspects_batch,
    is_rust_available,
)
from aspect_extraction.extractors.rule_based import RuleBasedExtractor
from aspect_extraction.extractors.transformer_based import TransformerExtractor
