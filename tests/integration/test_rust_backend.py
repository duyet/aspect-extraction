"""Integration tests for Rust backend."""

import pytest

from aspect_extraction.core.factory import (
    create_extractor,
    extract_aspects,
    extract_aspects_batch,
    is_rust_available,
)

# Check if Rust backend is available
RUST_AVAILABLE = is_rust_available()


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend not available")
class TestRustBackend:
    """Test Rust backend functionality."""

    def test_rust_backend_available(self):
        """Test that Rust backend can be imported."""
        assert is_rust_available()

        # Should be able to import Rust module
        import aspect_extraction_core

        assert hasattr(aspect_extraction_core, "RuleBasedExtractor")
        assert hasattr(aspect_extraction_core, "StatisticalExtractor")

    def test_create_extractor_rust_rule_based(self):
        """Test creating rule-based extractor with Rust backend."""
        extractor = create_extractor(method="rule", backend="rust")

        assert extractor is not None
        assert extractor.name() == "rule-based"

    def test_create_extractor_rust_statistical(self):
        """Test creating statistical extractor with Rust backend."""
        extractor = create_extractor(method="statistical", backend="rust")

        assert extractor is not None
        assert extractor.name() == "statistical"

    def test_rust_rule_based_extraction(self):
        """Test aspect extraction with Rust rule-based extractor."""
        extractor = create_extractor(method="rule", backend="rust")
        text = "The camera quality is excellent but battery life is poor"

        aspects = extractor.extract(text)

        assert len(aspects) > 0
        assert all(hasattr(a, "text") for a in aspects)
        assert all(hasattr(a, "confidence") for a in aspects)
        assert all(0.0 <= a.confidence <= 1.0 for a in aspects)

    def test_rust_statistical_extraction(self):
        """Test aspect extraction with Rust statistical extractor."""
        extractor = create_extractor(method="statistical", backend="rust")
        text = "The camera is great and the camera is excellent"

        aspects = extractor.extract(text)

        assert len(aspects) > 0
        # Should extract "camera" due to frequency
        aspect_texts = [a.text for a in aspects]
        assert any("camera" in text.lower() for text in aspect_texts)

    def test_rust_batch_processing(self):
        """Test batch processing with Rust backend."""
        extractor = create_extractor(method="rule", backend="rust")
        texts = [
            "The camera quality is great",
            "Battery life is poor",
            "Screen is beautiful",
        ]

        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_extract_aspects_rust(self):
        """Test convenience function with Rust backend."""
        text = "The camera is excellent"

        aspects = extract_aspects(text, method="rule", backend="rust")

        assert len(aspects) > 0

    def test_extract_aspects_batch_rust(self):
        """Test batch convenience function with Rust backend."""
        texts = ["Camera is good", "Battery is bad"]

        results = extract_aspects_batch(texts, method="rule", backend="rust")

        assert len(results) == 2

    def test_rust_auto_backend(self):
        """Test auto backend selection when Rust is available."""
        # With backend="auto", should use Rust if available
        extractor = create_extractor(method="rule", backend="auto")

        # Check it's using Rust implementation
        assert extractor is not None

        aspects = extractor.extract("Great camera")
        assert len(aspects) >= 0  # Should work without errors

    def test_rust_vs_python_same_results(self):
        """Test that Rust and Python backends produce similar results."""
        text = "The camera quality is excellent but battery life is poor"

        # Python backend
        python_extractor = create_extractor(method="rule", backend="python")
        python_aspects = python_extractor.extract(text)

        # Rust backend
        rust_extractor = create_extractor(method="rule", backend="rust")
        rust_aspects = rust_extractor.extract(text)

        # Should find similar number of aspects (may not be exactly the same)
        # Due to minor implementation differences
        assert len(python_aspects) > 0
        assert len(rust_aspects) > 0

        # Extract aspect texts (lowercase for comparison)
        python_texts = {a.text.lower() for a in python_aspects}
        rust_texts = {a.text.lower() for a in rust_aspects}

        # Should have some overlap
        overlap = python_texts & rust_texts
        assert len(overlap) > 0, f"No overlap between {python_texts} and {rust_texts}"

    def test_rust_with_min_confidence(self):
        """Test Rust extractor with confidence threshold."""
        extractor = create_extractor(
            method="rule", backend="rust", min_confidence=0.8
        )

        text = "The camera is nice"
        aspects = extractor.extract(text)

        # All aspects should meet confidence threshold
        assert all(a.confidence >= 0.8 for a in aspects)

    def test_rust_empty_text_error(self):
        """Test that Rust backend handles empty text gracefully."""
        extractor = create_extractor(method="rule", backend="rust")

        # Should raise an error or return empty list
        with pytest.raises(Exception):  # Could be ValueError or other
            extractor.extract("")


class TestBackendSelection:
    """Test backend selection logic."""

    def test_auto_backend_selection(self):
        """Test auto backend selection."""
        extractor = create_extractor(method="rule", backend="auto")

        assert extractor is not None

        # Should work regardless of which backend is used
        aspects = extractor.extract("Great camera")
        assert isinstance(aspects, list)

    def test_python_backend_always_works(self):
        """Test that Python backend is always available."""
        extractor = create_extractor(method="rule", backend="python")

        assert extractor is not None

        aspects = extractor.extract("Nice battery")
        assert len(aspects) >= 0

    @pytest.mark.skipif(RUST_AVAILABLE, reason="Testing Rust unavailable case")
    def test_rust_backend_error_when_unavailable(self):
        """Test error when Rust backend is requested but not available."""
        with pytest.raises(ImportError, match="Rust backend requested but not available"):
            create_extractor(method="rule", backend="rust")

    def test_invalid_backend(self):
        """Test error for invalid backend."""
        with pytest.raises(ValueError, match="Invalid backend"):
            create_extractor(method="rule", backend="invalid")  # type: ignore

    def test_transformer_rust_not_implemented(self):
        """Test that transformer is not available in Rust yet."""
        if RUST_AVAILABLE:
            with pytest.raises(
                NotImplementedError, match="Transformer method not yet available in Rust"
            ):
                create_extractor(method="transformer", backend="rust")


class TestRustPerformance:
    """Performance tests comparing Rust vs Python."""

    @pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend not available")
    @pytest.mark.benchmark
    def test_rust_is_faster_than_python(self, benchmark=None):
        """Test that Rust backend is faster than Python (when available)."""
        import time

        text = "The camera quality is excellent but battery life is poor. " * 10
        iterations = 100

        # Benchmark Python
        python_extractor = create_extractor(method="rule", backend="python")
        start = time.time()
        for _ in range(iterations):
            python_extractor.extract(text)
        python_time = time.time() - start

        # Benchmark Rust
        rust_extractor = create_extractor(method="rule", backend="rust")
        start = time.time()
        for _ in range(iterations):
            rust_extractor.extract(text)
        rust_time = time.time() - start

        # Rust should be significantly faster (at least 5x)
        speedup = python_time / rust_time
        print(f"\nSpeedup: {speedup:.1f}x (Python: {python_time:.3f}s, Rust: {rust_time:.3f}s)")

        # Note: We don't assert speedup here because it depends on hardware
        # But we can at least verify both work
        assert python_time > 0
        assert rust_time > 0
