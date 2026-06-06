"""Base interface for aspect extractors."""

from abc import ABC, abstractmethod
from typing import List

from aspect_extraction.core.aspect import Aspect


class AspectExtractor(ABC):
    """Abstract base class for all aspect extractors.

    All extractors must implement the extract method to provide
    aspect extraction functionality. This enables different extraction
    strategies (rule-based, statistical, transformer-based) to be used
    interchangeably.

    Example:
        >>> class MyExtractor(AspectExtractor):
        ...     def extract(self, text: str) -> List[Aspect]:
        ...         # Custom extraction logic
        ...         return [Aspect(text="example")]
        ...
        >>> extractor = MyExtractor()
        >>> aspects = extractor.extract("Some text")
    """

    @abstractmethod
    def extract(self, text: str) -> List[Aspect]:
        """Extract aspects from the given text.

        Args:
            text: Input text to extract aspects from

        Returns:
            List of extracted aspects

        Raises:
            ValueError: If text is empty or invalid
        """
        pass

    def extract_batch(self, texts: List[str], batch_size: int = 32) -> List[List[Aspect]]:
        """Extract aspects from multiple texts in batches.

        This default implementation processes texts sequentially.
        Subclasses can override for more efficient batch processing.

        Args:
            texts: List of input texts
            batch_size: Number of texts to process in each batch

        Returns:
            List of aspect lists, one for each input text

        Raises:
            ValueError: If texts is empty or contains invalid items
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        results: List[List[Aspect]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.extract(text))

        return results

    def __repr__(self) -> str:
        """Return string representation of the extractor."""
        return f"{self.__class__.__name__}()"
