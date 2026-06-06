#!/usr/bin/env python
"""
Demonstrate Rust acceleration for aspect extraction.

This example shows how to use the Rust backend for 10-100x performance
improvements over the pure Python implementation.

Usage:
    python examples/rust_acceleration.py
"""

import time
from typing import List

from aspect_extraction.core.aspect import Aspect
from aspect_extraction.core.factory import (
    create_extractor,
    extract_aspects,
    is_rust_available,
)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def print_aspects(aspects: List[Aspect]) -> None:
    """Print extracted aspects in a formatted way."""
    if not aspects:
        print("  No aspects found")
        return

    for i, aspect in enumerate(aspects, 1):
        sentiment_str = f" ({aspect.sentiment})" if aspect.sentiment else ""
        print(f"  {i}. {aspect.text}{sentiment_str} [confidence: {aspect.confidence:.2f}]")


def demo_basic_usage() -> None:
    """Demonstrate basic usage with automatic backend selection."""
    print_header("1. Basic Usage - Automatic Backend Selection")

    text = "The camera quality is excellent but battery life is poor"
    print(f"Text: '{text}'\n")

    # Auto-detect best backend (Rust if available, otherwise Python)
    print("Using backend='auto' (prefers Rust if available):")
    aspects = extract_aspects(text, method="rule", backend="auto")
    print_aspects(aspects)

    if is_rust_available():
        print("\n✓ Rust backend is being used!")
    else:
        print("\n✓ Python backend is being used (Rust not available)")


def demo_explicit_backends() -> None:
    """Demonstrate explicit backend selection."""
    print_header("2. Explicit Backend Selection")

    text = "The screen display is gorgeous and the design is beautiful"

    # Python backend
    print("A. Python Backend:")
    print(f"Text: '{text}'\n")
    python_aspects = extract_aspects(text, method="rule", backend="python")
    print_aspects(python_aspects)

    # Rust backend (if available)
    if is_rust_available():
        print("\n\nB. Rust Backend (10-100x faster):")
        print(f"Text: '{text}'\n")
        rust_aspects = extract_aspects(text, method="rule", backend="rust")
        print_aspects(rust_aspects)
    else:
        print("\n\nB. Rust Backend:")
        print("  ⚠️  Rust backend not available")
        print("  Install with: pip install maturin")
        print("                cd rust/aspect-extraction-core")
        print("                maturin develop --release")


def demo_batch_processing() -> None:
    """Demonstrate batch processing with Rust acceleration."""
    print_header("3. Batch Processing (Parallel in Rust)")

    texts = [
        "The camera quality is excellent",
        "Battery life is disappointing",
        "Screen size is perfect",
        "Design is beautiful but bulky",
        "Performance is outstanding",
    ]

    print(f"Processing {len(texts)} reviews...\n")

    if is_rust_available():
        # Rust backend with parallel processing
        extractor = create_extractor(method="rule", backend="rust")
        results = extractor.extract_batch(texts)

        for i, (text, aspects) in enumerate(zip(texts, results), 1):
            print(f"Review {i}: '{text}'")
            print_aspects(aspects)
            print()

        print("✓ Processed in parallel using Rust backend!")
    else:
        # Python backend (sequential)
        extractor = create_extractor(method="rule", backend="python")
        results = extractor.extract_batch(texts)

        for i, (text, aspects) in enumerate(zip(texts, results), 1):
            print(f"Review {i}: '{text}'")
            print_aspects(aspects)
            print()

        print("✓ Processed using Python backend (Rust not available)")


def benchmark_performance() -> None:
    """Benchmark Python vs Rust performance."""
    print_header("4. Performance Benchmark")

    # Sample text
    text = (
        "The camera quality is excellent but battery life is poor. "
        "The screen display is gorgeous and the design is beautiful. "
        "Performance is outstanding but the price is too high."
    )

    iterations = 1000
    print(f"Running {iterations} iterations...\n")

    # Benchmark Python
    print("A. Python Backend:")
    python_extractor = create_extractor(method="rule", backend="python")

    start_time = time.time()
    for _ in range(iterations):
        python_extractor.extract(text)
    python_time = time.time() - start_time

    print(f"   Time: {python_time:.3f} seconds")
    print(f"   Per extraction: {(python_time / iterations) * 1000:.2f} ms")

    # Benchmark Rust
    if is_rust_available():
        print("\nB. Rust Backend:")
        rust_extractor = create_extractor(method="rule", backend="rust")

        start_time = time.time()
        for _ in range(iterations):
            rust_extractor.extract(text)
        rust_time = time.time() - start_time

        print(f"   Time: {rust_time:.3f} seconds")
        print(f"   Per extraction: {(rust_time / iterations) * 1000:.2f} ms")

        # Calculate speedup
        speedup = python_time / rust_time
        print(f"\n🚀 Speedup: {speedup:.1f}x faster with Rust!")

        # Visual representation
        bar_length = 50
        python_bar = "█" * bar_length
        rust_bar = "█" * max(1, int(bar_length / speedup))

        print("\nVisual Comparison:")
        print(f"  Python: {python_bar}")
        print(f"  Rust:   {rust_bar}")
    else:
        print("\nB. Rust Backend: Not available")
        print("   Install Rust backend to see the performance difference!")


def demo_statistical_extractor() -> None:
    """Demonstrate statistical extractor with Rust."""
    print_header("5. Statistical Extractor with Rust")

    text = "camera camera quality battery battery life screen display"
    print(f"Text: '{text}'\n")

    if is_rust_available():
        print("Using Rust backend:")
        extractor = create_extractor(
            method="statistical", backend="rust", min_frequency=2
        )
        aspects = extractor.extract(text)
        print_aspects(aspects)

        print("\n✓ Statistical analysis completed using Rust!")
    else:
        print("Using Python backend:")
        extractor = create_extractor(
            method="statistical", backend="python", min_frequency=2
        )
        aspects = extractor.extract(text)
        print_aspects(aspects)

        print("\n✓ Statistical analysis completed using Python!")


def main() -> None:
    """Run all demonstrations."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        🦀 RUST ACCELERATION FOR ASPECT EXTRACTION 🚀              ║
║                                                                    ║
║  This demo shows how Rust can accelerate aspect extraction        ║
║  by 10-100x compared to pure Python implementation.               ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # Check Rust availability
    if is_rust_available():
        print("✓ Rust backend is available!")
        print("  You're getting maximum performance!\n")
    else:
        print("⚠️  Rust backend is NOT available")
        print("  Using pure Python implementation (slower)\n")
        print("To install Rust backend:")
        print("  1. Install Rust: https://rustup.rs/")
        print("  2. pip install maturin")
        print("  3. cd rust/aspect-extraction-core && maturin develop --release\n")

    # Run demonstrations
    try:
        demo_basic_usage()
        demo_explicit_backends()
        demo_batch_processing()
        demo_statistical_extractor()
        benchmark_performance()

        print_header("Summary")
        print("All demonstrations completed successfully!")

        if is_rust_available():
            print("\n🎉 You're using Rust acceleration!")
            print("   Enjoy 10-100x faster aspect extraction!")
        else:
            print("\n💡 Install Rust backend for maximum performance!")
            print("   See RUST_INTEGRATION.md for installation instructions.")

        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
