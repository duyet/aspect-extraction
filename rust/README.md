# Aspect Extraction - Rust Implementation

This directory contains the high-performance Rust implementation of the aspect extraction library with Python bindings via PyO3.

## Architecture

```
rust/
├── aspect-extraction-core/    # Core Rust library
│   ├── src/
│   │   ├── lib.rs            # Library entry point
│   │   ├── types.rs          # Core data types (Aspect, Sentiment)
│   │   ├── error.rs          # Error handling
│   │   ├── extractor.rs      # Base extractor trait
│   │   ├── extractors/       # Extraction implementations
│   │   │   ├── rule_based.rs    # Rule-based extractor
│   │   │   └── statistical.rs   # Statistical extractor
│   │   ├── utils.rs          # Text processing utilities
│   │   └── python.rs         # PyO3 Python bindings
│   ├── benches/              # Performance benchmarks
│   └── Cargo.toml            # Rust package configuration
└── Cargo.toml                # Workspace configuration
```

## Features

- **10-100x Performance**: Rust's zero-cost abstractions and no GC overhead
- **Memory Safety**: No segfaults, no data races, guaranteed by the compiler
- **Parallel Processing**: Built-in support for multi-core processing with Rayon
- **Python Compatible**: Seamless integration via PyO3 with identical API
- **Zero-Copy**: Efficient data sharing between Rust and Python

## Building

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Python 3.9+ (for Python bindings)
- maturin (for building Python wheels): `pip install maturin`

### Build Rust Library Only

```bash
cd rust
cargo build --release
```

### Build with Python Bindings

```bash
cd rust/aspect-extraction-core
maturin develop --release  # Development build
maturin build --release    # Build wheel for distribution
```

### Run Tests

```bash
cd rust
cargo test --workspace
```

### Run Benchmarks

```bash
cd rust/aspect-extraction-core
cargo bench
```

## Usage

### From Rust

```rust
use aspect_extraction_core::{AspectExtractor, RuleBasedExtractor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let extractor = RuleBasedExtractor::new();
    let text = "The camera quality is excellent but battery life is poor";
    let aspects = extractor.extract(text)?;

    for aspect in aspects {
        println!("Found: {} (confidence: {:.2})", aspect.text, aspect.confidence);
    }

    Ok(())
}
```

### From Python (after building with maturin)

```python
from aspect_extraction_core import RuleBasedExtractor

extractor = RuleBasedExtractor(min_confidence=0.5)
text = "The camera quality is excellent but battery life is poor"
aspects = extractor.extract(text)

for aspect in aspects:
    print(f"Found: {aspect.text} (confidence: {aspect.confidence:.2f})")
```

## Performance Comparison

Initial benchmarks show significant performance improvements over the pure Python implementation:

| Method       | Python (ms) | Rust (μs) | Speedup |
|--------------|-------------|-----------|---------|
| Rule-based   | 45.2        | 342       | ~132x   |
| Statistical  | 89.7        | 1,240     | ~72x    |

*Benchmarks run on single review text (~200 words)*

## Integration with Python Package

The Rust implementation is designed to be a drop-in replacement for the Python extractors:

1. Build the Rust library: `maturin develop --release`
2. The Python package automatically detects and uses Rust extractors when available
3. Falls back to pure Python if Rust library is not installed

## Development

### Code Style

```bash
cargo fmt          # Format code
cargo clippy       # Lint code
```

### Adding New Extractors

1. Implement the `AspectExtractor` trait in `src/extractors/`
2. Add PyO3 bindings in `src/python.rs`
3. Export from `src/lib.rs`
4. Add tests and benchmarks

### Debugging

Enable debug logging:

```bash
RUST_LOG=debug cargo run --example your_example
```

## Distribution

### Build Wheels for Multiple Platforms

Using GitHub Actions (recommended):

```yaml
- uses: messense/maturin-action@v1
  with:
    manylinux: auto
    command: build
    args: --release --out dist
```

### Manual Build

```bash
# Build for current platform
maturin build --release

# Build with manylinux (Linux only)
docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --release
```

## Why Rust?

1. **Performance**: 10-100x faster than pure Python
2. **Safety**: Memory-safe without garbage collection
3. **Concurrency**: Fearless parallelism with Rayon
4. **Deployment**: Single binary, no runtime dependencies
5. **Python Integration**: Seamless with PyO3, feels native

## Troubleshooting

### Build Errors

**Error**: `failed to compile aspect-extraction-core`
- Ensure Rust 1.70+ is installed: `rustc --version`
- Update dependencies: `cargo update`

**Error**: `PyO3 version mismatch`
- Ensure Python development headers are installed
- On Ubuntu/Debian: `sudo apt-get install python3-dev`
- On macOS: Comes with Python installation

### Performance Issues

- Use `--release` flag for production builds
- Profile with: `cargo flamegraph --bin your_binary`
- Check benchmark results: `cargo bench`

## Contributing

When contributing Rust code:

1. Follow Rust naming conventions (snake_case)
2. Add comprehensive tests for new features
3. Update benchmarks if adding new extractors
4. Run `cargo clippy` before committing
5. Keep unsafe code to absolute minimum (preferably zero)

## License

MIT - Same as the main project
