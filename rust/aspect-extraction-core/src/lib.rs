//! Aspect Extraction Core Library
//!
//! A high-performance Rust library for extracting aspects from text with Python bindings.
//!
//! # Features
//!
//! - **Rule-based extraction**: Fast, interpretable pattern matching
//! - **Statistical extraction**: Frequency and collocation analysis
//! - **Python FFI**: Zero-copy integration with Python via PyO3
//! - **Memory safe**: No garbage collection, no memory leaks
//! - **Parallel processing**: Leverage all CPU cores with Rayon
//!
//! # Example
//!
//! ```rust
//! use aspect_extraction_core::{Aspect, RuleBasedExtractor, AspectExtractor};
//!
//! let extractor = RuleBasedExtractor::new();
//! let text = "The camera quality is excellent but battery life is poor";
//! let aspects = extractor.extract(text)?;
//!
//! for aspect in aspects {
//!     println!("Found aspect: {} (confidence: {:.2})", aspect.text, aspect.confidence);
//! }
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![warn(clippy::all)]

pub mod error;
pub mod extractor;
pub mod extractors;
pub mod types;
pub mod utils;

// Re-export main types
pub use error::{Error, Result};
pub use extractor::AspectExtractor;
pub use types::{Aspect, Sentiment};

// Re-export extractors
pub use extractors::rule_based::RuleBasedExtractor;
pub use extractors::statistical::StatisticalExtractor;

// Python bindings (optional feature)
#[cfg(feature = "python")]
pub mod python;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
