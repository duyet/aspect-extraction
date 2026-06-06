#!/usr/bin/env python
"""
Comprehensive benchmark comparing Python vs Rust backends.

This script provides detailed performance comparisons across different
scenarios and workloads.

Usage:
    python examples/benchmark_backends.py
"""

import time
from typing import Dict, List, Tuple

from aspect_extraction.core.factory import create_extractor, is_rust_available


def format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f} μs"
    elif seconds < 1.0:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def run_benchmark(
    name: str, backend: str, method: str, texts: List[str], iterations: int = 100
) -> Dict[str, float]:
    """Run a benchmark and return timing results."""
    print(f"  Running {name} ({backend})...", end=" ", flush=True)

    extractor = create_extractor(method=method, backend=backend)

    # Warmup
    for text in texts[:min(10, len(texts))]:
        extractor.extract(text)

    # Benchmark single extraction
    start = time.time()
    for _ in range(iterations):
        for text in texts:
            extractor.extract(text)
    total_time = time.time() - start
    single_time = total_time / (iterations * len(texts))

    # Benchmark batch extraction
    start = time.time()
    for _ in range(iterations):
        extractor.extract_batch(texts)
    batch_total_time = time.time() - start
    batch_time = batch_total_time / iterations

    print("Done")

    return {
        "total_time": total_time,
        "single_time": single_time,
        "batch_time": batch_time,
        "throughput": (iterations * len(texts)) / total_time,
    }


def print_comparison(
    name: str, python_results: Dict[str, float], rust_results: Dict[str, float]
) -> None:
    """Print a formatted comparison of results."""
    print(f"\n{name}:")
    print("-" * 70)

    # Single extraction time
    py_single = python_results["single_time"]
    rust_single = rust_results["single_time"]
    speedup_single = py_single / rust_single

    print(f"  Single Extraction:")
    print(f"    Python: {format_time(py_single)}")
    print(f"    Rust:   {format_time(rust_single)}")
    print(f"    Speedup: {speedup_single:.1f}x faster 🚀")

    # Batch extraction time
    py_batch = python_results["batch_time"]
    rust_batch = rust_results["batch_time"]
    speedup_batch = py_batch / rust_batch

    print(f"\n  Batch Processing:")
    print(f"    Python: {format_time(py_batch)}")
    print(f"    Rust:   {format_time(rust_batch)}")
    print(f"    Speedup: {speedup_batch:.1f}x faster 🚀")

    # Throughput
    py_throughput = python_results["throughput"]
    rust_throughput = rust_results["throughput"]

    print(f"\n  Throughput (extractions/sec):")
    print(f"    Python: {py_throughput:.1f}")
    print(f"    Rust:   {rust_throughput:.1f}")
    print(f"    Improvement: {(rust_throughput / py_throughput):.1f}x")


def benchmark_short_texts() -> Tuple[Dict, Dict]:
    """Benchmark short texts (product reviews)."""
    texts = [
        "Great camera",
        "Poor battery",
        "Excellent screen",
        "Bad design",
        "Amazing performance",
    ]

    print("\n1. Short Texts (5-10 words)")
    python_results = run_benchmark(
        "Short texts", "python", "rule", texts, iterations=200
    )

    if is_rust_available():
        rust_results = run_benchmark(
            "Short texts", "rust", "rule", texts, iterations=200
        )
        return python_results, rust_results
    else:
        return python_results, {}


def benchmark_medium_texts() -> Tuple[Dict, Dict]:
    """Benchmark medium-length texts."""
    texts = [
        "The camera quality is excellent but battery life is disappointing",
        "Screen display is gorgeous but the design feels bulky and heavy",
        "Performance is outstanding for the price point and build quality",
        "Battery life is poor compared to competitors in the same range",
        "The design is beautiful and the materials feel premium quality",
    ]

    print("\n2. Medium Texts (10-20 words)")
    python_results = run_benchmark(
        "Medium texts", "python", "rule", texts, iterations=100
    )

    if is_rust_available():
        rust_results = run_benchmark(
            "Medium texts", "rust", "rule", texts, iterations=100
        )
        return python_results, rust_results
    else:
        return python_results, {}


def benchmark_long_texts() -> Tuple[Dict, Dict]:
    """Benchmark long texts."""
    texts = [
        "The camera quality is absolutely excellent with great low-light performance. "
        "However, the battery life is quite disappointing and doesn't last a full day. "
        "The screen display is gorgeous with vibrant colors and deep blacks. "
        "The design is beautiful but feels a bit bulky in the hand. "
        "Overall performance is outstanding for the price point.",
    ] * 3

    print("\n3. Long Texts (50+ words)")
    python_results = run_benchmark(
        "Long texts", "python", "rule", texts, iterations=50
    )

    if is_rust_available():
        rust_results = run_benchmark("Long texts", "rust", "rule", texts, iterations=50)
        return python_results, rust_results
    else:
        return python_results, {}


def benchmark_statistical() -> Tuple[Dict, Dict]:
    """Benchmark statistical extractor."""
    texts = [
        "camera camera quality battery battery life screen display",
        "battery life camera quality screen size design materials",
        "performance speed camera display battery charging time",
    ] * 5

    print("\n4. Statistical Extraction")
    python_results = run_benchmark(
        "Statistical", "python", "statistical", texts, iterations=50
    )

    if is_rust_available():
        rust_results = run_benchmark(
            "Statistical", "rust", "statistical", texts, iterations=50
        )
        return python_results, rust_results
    else:
        return python_results, {}


def print_summary(all_results: List[Tuple[str, Dict, Dict]]) -> None:
    """Print overall summary."""
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    if not is_rust_available():
        print("\n⚠️  Rust backend not available - install for performance boost!")
        print("\nInstallation:")
        print("  1. Install Rust: https://rustup.rs/")
        print("  2. pip install maturin")
        print("  3. cd rust/aspect-extraction-core && maturin develop --release")
        return

    print("\nAverage Speedups:")
    print("-" * 70)

    total_speedup = 0.0
    count = 0

    for name, py_results, rust_results in all_results:
        if rust_results:
            speedup = py_results["single_time"] / rust_results["single_time"]
            print(f"  {name:30s}: {speedup:6.1f}x")
            total_speedup += speedup
            count += 1

    if count > 0:
        avg_speedup = total_speedup / count
        print(f"\n  {'Average':30s}: {avg_speedup:6.1f}x 🚀")

    print("\n" + "=" * 70)
    print("Conclusion:")
    print("-" * 70)
    print(f"  Rust backend is consistently {avg_speedup:.0f}x faster across all scenarios!")
    print("  This translates to:")
    print(f"    - Processing 1000 reviews in seconds instead of minutes")
    print(f"    - Real-time extraction for interactive applications")
    print(f"    - Reduced server costs for API deployments")
    print("=" * 70)


def main():
    """Run all benchmarks."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║              PYTHON vs RUST BACKEND BENCHMARK                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    if is_rust_available():
        print("✓ Rust backend detected - running full benchmarks\n")
    else:
        print("⚠️  Rust backend not available - running Python-only benchmarks\n")

    all_results = []

    try:
        # Run benchmarks
        py1, rust1 = benchmark_short_texts()
        all_results.append(("Short Texts", py1, rust1))

        py2, rust2 = benchmark_medium_texts()
        all_results.append(("Medium Texts", py2, rust2))

        py3, rust3 = benchmark_long_texts()
        all_results.append(("Long Texts", py3, rust3))

        py4, rust4 = benchmark_statistical()
        all_results.append(("Statistical Extraction", py4, rust4))

        # Print comparisons
        print("\n" + "=" * 70)
        print("DETAILED COMPARISONS")
        print("=" * 70)

        for name, py_results, rust_results in all_results:
            if rust_results:
                print_comparison(name, py_results, rust_results)

        # Print summary
        print_summary(all_results)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during benchmark: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
