//! Error types for aspect extraction.

use std::fmt;
use thiserror::Error;

/// Result type alias using our Error type
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for aspect extraction operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum Error {
    /// Input validation error
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Text is empty or contains only whitespace
    #[error("Text cannot be empty")]
    EmptyText,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Extraction failed
    #[error("Extraction failed: {0}")]
    ExtractionFailed(String),

    /// Regex compilation error
    #[error("Regex error: {0}")]
    RegexError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(String),

    /// Feature not available
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),

    /// Invalid confidence score
    #[error("Confidence must be between 0.0 and 1.0, got {0}")]
    InvalidConfidence(f64),

    /// Invalid n-gram size
    #[error("N-gram size must be at least 1, got {0}")]
    InvalidNgramSize(usize),
}

impl From<regex::Error> for Error {
    fn from(err: regex::Error) -> Self {
        Error::RegexError(err.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::EmptyText;
        assert_eq!(err.to_string(), "Text cannot be empty");

        let err = Error::InvalidInput("test".to_string());
        assert_eq!(err.to_string(), "Invalid input: test");
    }

    #[test]
    fn test_error_equality() {
        assert_eq!(Error::EmptyText, Error::EmptyText);
        assert_ne!(
            Error::InvalidInput("a".to_string()),
            Error::InvalidInput("b".to_string())
        );
    }
}
