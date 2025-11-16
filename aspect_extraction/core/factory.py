"""Factory function for creating aspect extractors."""

from typing import List, Literal, Optional

from aspect_extraction.core.aspect import Aspect
from aspect_extraction.core.extractor import AspectExtractor

# Method type hints
ExtractorMethod = Literal["rule", "statistical", "transformer", "auto"]


def create_extractor(
    method: ExtractorMethod = "auto",
    **kwargs,  # type: ignore
) -> AspectExtractor:
    """Create an aspect extractor using the specified method.

    Args:
        method: Extraction method to use:
            - "rule": Rule-based extraction (fast, interpretable)
            - "statistical": Statistical extraction (balanced)
            - "transformer": Transformer-based (accurate, slower)
            - "auto": Automatically select best available method
        **kwargs: Additional arguments passed to the extractor

    Returns:
        Configured aspect extractor

    Raises:
        ValueError: If method is invalid

    Example:
        >>> extractor = create_extractor(method="rule")
        >>> aspects = extractor.extract("The camera quality is great")
        >>> print(aspects)
    """
    if method == "rule":
        from aspect_extraction.extractors.rule_based import RuleBasedExtractor

        return RuleBasedExtractor(**kwargs)

    elif method == "statistical":
        from aspect_extraction.extractors.statistical import StatisticalExtractor

        return StatisticalExtractor(**kwargs)

    elif method == "transformer":
        from aspect_extraction.extractors.transformer_based import TransformerExtractor

        return TransformerExtractor(**kwargs)

    elif method == "auto":
        # Try to use the best available method
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
    **kwargs,  # type: ignore
) -> List[Aspect]:
    """Extract aspects from text using specified method.

    This is a convenience function that creates an extractor
    and immediately extracts aspects.

    Args:
        text: Input text to analyze
        method: Extraction method (see create_extractor)
        **kwargs: Additional arguments for the extractor

    Returns:
        List of extracted aspects

    Raises:
        ValueError: If text is empty or method is invalid

    Example:
        >>> aspects = extract_aspects(
        ...     "The battery life is great but camera quality is poor",
        ...     method="rule"
        ... )
        >>> for aspect in aspects:
        ...     print(f"{aspect.text}: {aspect.sentiment}")
        battery life: positive
        camera quality: negative
    """
    extractor = create_extractor(method=method, **kwargs)
    return extractor.extract(text)


def extract_aspects_batch(
    texts: List[str],
    method: ExtractorMethod = "auto",
    batch_size: int = 32,
    **kwargs,  # type: ignore
) -> List[List[Aspect]]:
    """Extract aspects from multiple texts efficiently.

    Args:
        texts: List of input texts
        method: Extraction method (see create_extractor)
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
        >>> results = extract_aspects_batch(texts, method="rule")
        >>> print(len(results))
        2
    """
    extractor = create_extractor(method=method, **kwargs)
    return extractor.extract_batch(texts, batch_size=batch_size)
