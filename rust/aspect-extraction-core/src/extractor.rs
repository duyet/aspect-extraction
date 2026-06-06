//! Base trait for aspect extractors.

use crate::error::Result;
use crate::types::Aspect;

/// Trait for aspect extraction implementations
pub trait AspectExtractor {
    /// Extract aspects from the given text
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to extract aspects from
    ///
    /// # Returns
    ///
    /// A vector of extracted aspects
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails or input is invalid
    fn extract(&self, text: &str) -> Result<Vec<Aspect>>;

    /// Extract aspects from multiple texts (batch processing)
    ///
    /// Default implementation processes texts sequentially.
    /// Implementations can override for parallel processing.
    ///
    /// # Arguments
    ///
    /// * `texts` - Vector of input texts
    ///
    /// # Returns
    ///
    /// A vector of vectors, where each inner vector contains aspects for one text
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails for any text
    fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Vec<Aspect>>> {
        texts.iter().map(|text| self.extract(text)).collect()
    }

    /// Get the name of this extractor
    fn name(&self) -> &'static str;

    /// Get a description of this extractor's approach
    fn description(&self) -> &'static str {
        "Aspect extraction implementation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock extractor for testing
    struct MockExtractor;

    impl AspectExtractor for MockExtractor {
        fn extract(&self, text: &str) -> Result<Vec<Aspect>> {
            if text.is_empty() {
                return Err(crate::error::Error::EmptyText);
            }

            // Simple mock: extract words as aspects
            let aspects = text
                .split_whitespace()
                .map(|word| Aspect::new(word))
                .collect();

            Ok(aspects)
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }

    #[test]
    fn test_extractor_trait() {
        let extractor = MockExtractor;
        let aspects = extractor.extract("camera battery").unwrap();

        assert_eq!(aspects.len(), 2);
        assert_eq!(aspects[0].text, "camera");
        assert_eq!(aspects[1].text, "battery");
    }

    #[test]
    fn test_extractor_batch() {
        let extractor = MockExtractor;
        let texts = vec!["camera", "battery screen"];
        let results = extractor.extract_batch(&texts).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 1);
        assert_eq!(results[1].len(), 2);
    }

    #[test]
    fn test_extractor_empty_text() {
        let extractor = MockExtractor;
        let result = extractor.extract("");

        assert!(result.is_err());
    }
}
