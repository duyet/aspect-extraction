"""Factory function for creating aspect extractors."""

from typing import List, Literal

from aspect_extraction.core.aspect import Aspect
from aspect_extraction.core.extractor import AspectExtractor

# Method type hints
ExtractorMethod = Literal["rule", "statistical", "transformer", "auto"]
BackendType = Literal["auto", "rust", "python"]

# Check if Rust backend is available
_RUST_AVAILABLE = False
try:
    import aspect_extraction_core  # noqa: F401

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def is_rust_available() -> bool:
    """Check if Rust backend is available.

    Returns:
        True if Rust backend can be imported, False otherwise
    """
    return _RUST_AVAILABLE


def create_extractor(  # noqa: C901
    method: ExtractorMethod = "auto",
    backend: BackendType = "auto",
    **kwargs,  # type: ignore
) -> AspectExtractor:
    """Create an aspect extractor using the specified method and backend.

    Args:
        method: Extraction method to use:
            - "rule": Rule-based extraction (fast, interpretable)
            - "statistical": Statistical extraction (balanced)
            - "transformer": Transformer-based (accurate, slower)
            - "auto": Automatically select best available method
        backend: Backend to use:
            - "auto": Use Rust if available, otherwise Python (default)
            - "rust": Force Rust backend (raises error if not available)
            - "python": Force pure Python backend
        **kwargs: Additional arguments passed to the extractor

    Returns:
        Configured aspect extractor

    Raises:
        ValueError: If method is invalid
        ImportError: If rust backend is requested but not available

    Example:
        >>> # Use Rust backend if available, otherwise Python
        >>> extractor = create_extractor(method="rule", backend="auto")
        >>> aspects = extractor.extract("The camera quality is great")
        >>> print(aspects)

        >>> # Force Rust backend (10-100x faster)
        >>> extractor = create_extractor(method="rule", backend="rust")

        >>> # Force Python backend
        >>> extractor = create_extractor(method="rule", backend="python")
    """
    # Determine which backend to use
    use_rust = False
    if backend == "auto":
        use_rust = _RUST_AVAILABLE
    elif backend == "rust":
        if not _RUST_AVAILABLE:
            raise ImportError(
                "Rust backend requested but not available. "
                "Install with: pip install maturin && "
                "cd rust/aspect-extraction-core && maturin develop --release"
            )
        use_rust = True
    elif backend == "python":
        use_rust = False
    else:
        raise ValueError(f"Invalid backend '{backend}'. Choose from: 'auto', 'rust', 'python'")

    # Create extractor based on method and backend
    if method == "rule":
        if use_rust:
            from aspect_extraction_core import RuleBasedExtractor

            return RuleBasedExtractor(**kwargs)  # type: ignore
        else:
            from aspect_extraction.extractors.rule_based import RuleBasedExtractor

            return RuleBasedExtractor(**kwargs)

    elif method == "statistical":
        if use_rust:
            from aspect_extraction_core import StatisticalExtractor

            return StatisticalExtractor(**kwargs)  # type: ignore
        else:
            from aspect_extraction.extractors.statistical import StatisticalExtractor

            return StatisticalExtractor(**kwargs)

    elif method == "transformer":
        # Transformer only available in Python for now
        if use_rust:
            raise NotImplementedError(
                "Transformer method not yet available in Rust backend. "
                "Use backend='python' or method='rule'/'statistical'"
            )
        from aspect_extraction.extractors.transformer_based import TransformerExtractor

        return TransformerExtractor(**kwargs)

    elif method == "auto":
        # Try to use the best available method
        # Prefer Rust backend for rule/statistical if available
        if use_rust:
            from aspect_extraction_core import RuleBasedExtractor

            return RuleBasedExtractor(**kwargs)  # type: ignore

        try:
            from aspect_extraction.extractors.transformer_based import TransformerExtractor

            return TransformerExtractor(**kwargs)
        except ImportError:
            try:
                from aspect_extraction.extractors.rule_based import RuleBasedExtractor

                return RuleBasedExtractor(**kwargs)
            except ImportError:
                from aspect_extraction.extractors.statistical import StatisticalExtractor

                return StatisticalExtractor(**kwargs)

    else:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Choose from: 'rule', 'statistical', 'transformer', 'auto'"
        )


def extract_aspects(
    text: str,
    method: ExtractorMethod = "auto",
    backend: BackendType = "auto",
    **kwargs,  # type: ignore
) -> List[Aspect]:
    """Extract aspects from text using specified method and backend.

    This is a convenience function that creates an extractor
    and immediately extracts aspects.

    Args:
        text: Input text to analyze
        method: Extraction method (see create_extractor)
        backend: Backend to use ("auto", "rust", or "python")
        **kwargs: Additional arguments for the extractor

    Returns:
        List of extracted aspects

    Raises:
        ValueError: If text is empty or method is invalid

    Example:
        >>> # Auto-detect best backend (Rust if available)
        >>> aspects = extract_aspects(
        ...     "The battery life is great but camera quality is poor",
        ...     method="rule"
        ... )
        >>> for aspect in aspects:
        ...     print(f"{aspect.text}: {aspect.sentiment}")
        battery life: positive
        camera quality: negative

        >>> # Force Rust backend for maximum performance
        >>> aspects = extract_aspects(
        ...     "Great camera",
        ...     method="rule",
        ...     backend="rust"
        ... )
    """
    extractor = create_extractor(method=method, backend=backend, **kwargs)
    return extractor.extract(text)


def extract_aspects_batch(
    texts: List[str],
    method: ExtractorMethod = "auto",
    backend: BackendType = "auto",
    batch_size: int = 32,
    **kwargs,  # type: ignore
) -> List[List[Aspect]]:
    """Extract aspects from multiple texts efficiently.

    Args:
        texts: List of input texts
        method: Extraction method (see create_extractor)
        backend: Backend to use ("auto", "rust", or "python")
        batch_size: Batch size for processing
        **kwargs: Additional arguments for the extractor

    Returns:
        List of aspect lists, one for each input text

    Raises:
        ValueError: If texts is empty or method is invalid

    Example:
        >>> texts = [
        ...     "The camera is great",
        ...     "Battery life is poor"
        ... ]
        >>> # Use Rust backend for parallel processing (much faster)
        >>> results = extract_aspects_batch(texts, method="rule", backend="rust")
        >>> print(len(results))
        2
    """
    extractor = create_extractor(method=method, backend=backend, **kwargs)
    return extractor.extract_batch(texts, batch_size=batch_size)
