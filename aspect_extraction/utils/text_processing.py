"""Text processing utilities for aspect extraction."""

import re
from typing import List


def clean_text(text: str, lowercase: bool = False, remove_punctuation: bool = False) -> str:
    """Clean and normalize text.

    Args:
        text: Input text to clean
        lowercase: Convert to lowercase if True
        remove_punctuation: Remove punctuation if True

    Returns:
        Cleaned text

    Raises:
        ValueError: If text is empty

    Example:
        >>> clean_text("  Hello, World!  ", lowercase=True)
        'hello, world!'
    """
    if not text:
        raise ValueError("Text cannot be empty")

    # Remove extra whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()

    # Lowercase if requested
    if lowercase:
        cleaned = cleaned.lower()

    # Remove punctuation if requested
    if remove_punctuation:
        cleaned = re.sub(r"[^\w\s]", "", cleaned)

    return cleaned


def tokenize(text: str, lowercase: bool = False) -> List[str]:
    """Tokenize text into words.

    Args:
        text: Input text to tokenize
        lowercase: Convert to lowercase if True

    Returns:
        List of tokens

    Raises:
        ValueError: If text is empty

    Example:
        >>> tokenize("Hello, world!")
        ['Hello', ',', 'world', '!']
    """
    if not text:
        raise ValueError("Text cannot be empty")

    # Clean text first
    text = clean_text(text, lowercase=lowercase)

    # Simple whitespace tokenization
    # In production, you might use spaCy or NLTK for better tokenization
    tokens = text.split()

    return tokens


def extract_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences

    Raises:
        ValueError: If text is empty

    Example:
        >>> extract_sentences("Hello world. How are you?")
        ['Hello world.', 'How are you?']
    """
    if not text:
        raise ValueError("Text cannot be empty")

    # Simple sentence splitting
    # In production, use spaCy or NLTK for better sentence segmentation
    sentences = re.split(r"[.!?]+", text)

    # Clean up and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def remove_stopwords(tokens: List[str], custom_stopwords: List[str] = None) -> List[str]:
    """Remove stopwords from token list.

    Args:
        tokens: List of tokens
        custom_stopwords: Optional custom stopword list

    Returns:
        Filtered token list

    Example:
        >>> remove_stopwords(["the", "camera", "is", "good"])
        ['camera', 'good']
    """
    # Common English stopwords
    default_stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
    }

    stopwords = default_stopwords
    if custom_stopwords:
        stopwords = stopwords.union(set(custom_stopwords))

    return [token for token in tokens if token.lower() not in stopwords]


def get_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams from token list.

    Args:
        tokens: List of tokens
        n: N-gram size (default: 2 for bigrams)

    Returns:
        List of n-grams as strings

    Raises:
        ValueError: If n < 1 or tokens is empty

    Example:
        >>> get_ngrams(["the", "camera", "quality"], n=2)
        ['the camera', 'camera quality']
    """
    if not tokens:
        raise ValueError("Tokens cannot be empty")
    if n < 1:
        raise ValueError("n must be at least 1")

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i : i + n])
        ngrams.append(ngram)

    return ngrams


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace

    Example:
        >>> normalize_whitespace("Hello    world\\n\\nFoo  bar")
        'Hello world Foo bar'
    """
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to add to truncated text

    Returns:
        Truncated text

    Raises:
        ValueError: If max_length is less than suffix length

    Example:
        >>> truncate_text("Hello world", max_length=8)
        'Hello...'
    """
    if max_length < len(suffix):
        raise ValueError("max_length must be at least as long as suffix")

    if len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix
