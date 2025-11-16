"""Unit tests for utility functions."""

import pytest

from aspect_extraction.utils.text_processing import (
    clean_text,
    tokenize,
    extract_sentences,
    remove_stopwords,
    get_ngrams,
    normalize_whitespace,
    truncate_text,
)


class TestCleanText:
    """Test clean_text function."""

    def test_clean_basic_text(self) -> None:
        """Should clean and normalize text."""
        text = "  Hello   World  "
        result = clean_text(text)
        assert result == "Hello World"

    def test_lowercase_option(self) -> None:
        """Should lowercase when requested."""
        text = "Hello World"
        result = clean_text(text, lowercase=True)
        assert result == "hello world"

    def test_remove_punctuation_option(self) -> None:
        """Should remove punctuation when requested."""
        text = "Hello, World!"
        result = clean_text(text, remove_punctuation=True)
        assert result == "Hello World"

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            clean_text("")


class TestTokenize:
    """Test tokenize function."""

    def test_basic_tokenization(self) -> None:
        """Should tokenize text into words."""
        text = "Hello world"
        tokens = tokenize(text)
        assert tokens == ["Hello", "world"]

    def test_lowercase_option(self) -> None:
        """Should lowercase tokens when requested."""
        text = "Hello World"
        tokens = tokenize(text, lowercase=True)
        assert tokens == ["hello", "world"]

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            tokenize("")


class TestExtractSentences:
    """Test extract_sentences function."""

    def test_basic_sentence_splitting(self) -> None:
        """Should split text into sentences."""
        text = "Hello world. How are you?"
        sentences = extract_sentences(text)
        assert len(sentences) == 2
        assert "Hello world" in sentences[0]
        assert "How are you" in sentences[1]

    def test_multiple_punctuation(self) -> None:
        """Should handle multiple punctuation marks."""
        text = "Hello! How are you? I'm good."
        sentences = extract_sentences(text)
        assert len(sentences) == 3

    def test_empty_text_raises_error(self) -> None:
        """Should raise ValueError for empty text."""
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_sentences("")


class TestRemoveStopwords:
    """Test remove_stopwords function."""

    def test_remove_common_stopwords(self) -> None:
        """Should remove common stopwords."""
        tokens = ["the", "camera", "is", "good"]
        filtered = remove_stopwords(tokens)
        assert "camera" in filtered
        assert "good" in filtered
        assert "the" not in filtered
        assert "is" not in filtered

    def test_custom_stopwords(self) -> None:
        """Should use custom stopwords."""
        tokens = ["hello", "world", "foo"]
        custom = ["hello"]
        filtered = remove_stopwords(tokens, custom_stopwords=custom)
        assert "hello" not in filtered
        assert "world" in filtered


class TestGetNgrams:
    """Test get_ngrams function."""

    def test_bigrams(self) -> None:
        """Should generate bigrams correctly."""
        tokens = ["the", "camera", "quality"]
        bigrams = get_ngrams(tokens, n=2)
        assert bigrams == ["the camera", "camera quality"]

    def test_trigrams(self) -> None:
        """Should generate trigrams correctly."""
        tokens = ["the", "camera", "quality", "is", "good"]
        trigrams = get_ngrams(tokens, n=3)
        assert len(trigrams) == 3
        assert "the camera quality" in trigrams

    def test_empty_tokens_raises_error(self) -> None:
        """Should raise ValueError for empty tokens."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_ngrams([])

    def test_invalid_n_raises_error(self) -> None:
        """Should raise ValueError for invalid n."""
        with pytest.raises(ValueError, match="at least 1"):
            get_ngrams(["a", "b"], n=0)


class TestNormalizeWhitespace:
    """Test normalize_whitespace function."""

    def test_normalize_spaces(self) -> None:
        """Should normalize multiple spaces."""
        text = "Hello    world"
        result = normalize_whitespace(text)
        assert result == "Hello world"

    def test_normalize_newlines(self) -> None:
        """Should normalize newlines."""
        text = "Hello\n\nworld"
        result = normalize_whitespace(text)
        assert result == "Hello world"


class TestTruncateText:
    """Test truncate_text function."""

    def test_truncate_long_text(self) -> None:
        """Should truncate text that exceeds max length."""
        text = "Hello world"
        result = truncate_text(text, max_length=8)
        assert result == "Hello..."
        assert len(result) == 8

    def test_dont_truncate_short_text(self) -> None:
        """Should not truncate text within max length."""
        text = "Hello"
        result = truncate_text(text, max_length=10)
        assert result == "Hello"

    def test_custom_suffix(self) -> None:
        """Should use custom suffix."""
        text = "Hello world"
        result = truncate_text(text, max_length=10, suffix=">>")
        assert result.endswith(">>")

    def test_invalid_max_length_raises_error(self) -> None:
        """Should raise ValueError if max_length < suffix length."""
        with pytest.raises(ValueError, match="at least as long"):
            truncate_text("Hello", max_length=2, suffix="...")
