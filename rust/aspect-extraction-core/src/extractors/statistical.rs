//! Statistical aspect extraction using frequency and collocation analysis.
//!
//! This extractor identifies aspects based on:
//! - Word frequency (common nouns)
//! - Pointwise Mutual Information (PMI) for collocations
//! - Co-occurrence with opinion words

use crate::error::Result;
use crate::extractor::AspectExtractor;
use crate::types::{Aspect, Sentiment};
use crate::utils;
use hashbrown::HashMap;
use std::collections::HashSet;

/// Statistical aspect extractor
pub struct StatisticalExtractor {
    /// Minimum frequency threshold
    min_frequency: usize,

    /// Minimum PMI score for bigrams
    min_pmi: f64,

    /// Minimum confidence threshold
    min_confidence: f64,

    /// Trained word frequencies (optional)
    word_freq: HashMap<String, usize>,

    /// Trained bigram frequencies (optional)
    bigram_freq: HashMap<String, usize>,

    /// Total token count (for PMI calculation)
    total_tokens: usize,
}

impl StatisticalExtractor {
    /// Create a new statistical extractor with default configuration
    pub fn new() -> Self {
        Self {
            min_frequency: 2,
            min_pmi: 0.0,
            min_confidence: 0.5,
            word_freq: HashMap::new(),
            bigram_freq: HashMap::new(),
            total_tokens: 0,
        }
    }

    /// Create a new statistical extractor with custom configuration
    pub fn with_config(min_frequency: usize, min_pmi: f64, min_confidence: f64) -> Self {
        Self {
            min_frequency,
            min_pmi,
            min_confidence,
            word_freq: HashMap::new(),
            bigram_freq: HashMap::new(),
            total_tokens: 0,
        }
    }

    /// Train the extractor on a corpus of texts
    ///
    /// This builds word and bigram frequency distributions
    pub fn train(&mut self, texts: &[&str]) -> Result<()> {
        let mut word_freq = HashMap::new();
        let mut bigram_freq = HashMap::new();
        let mut total_tokens = 0;

        for text in texts {
            let tokens = utils::tokenize(text, true)?;
            let filtered = utils::remove_stopwords(tokens, None);

            // Count word frequencies
            for token in &filtered {
                *word_freq.entry(token.clone()).or_insert(0) += 1;
                total_tokens += 1;
            }

            // Count bigram frequencies
            let bigrams = utils::get_ngrams(&filtered, 2)?;
            for bigram in bigrams {
                *bigram_freq.entry(bigram).or_insert(0) += 1;
            }
        }

        self.word_freq = word_freq;
        self.bigram_freq = bigram_freq;
        self.total_tokens = total_tokens;

        Ok(())
    }

    /// Calculate PMI score for a bigram
    ///
    /// PMI(x, y) = log(P(x,y) / (P(x) * P(y)))
    fn calculate_pmi(&self, bigram: &str) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }

        let bigram_count = self.bigram_freq.get(bigram).copied().unwrap_or(0);
        if bigram_count == 0 {
            return 0.0;
        }

        let words: Vec<&str> = bigram.split_whitespace().collect();
        if words.len() != 2 {
            return 0.0;
        }

        let word1_count = self.word_freq.get(words[0]).copied().unwrap_or(0);
        let word2_count = self.word_freq.get(words[1]).copied().unwrap_or(0);

        if word1_count == 0 || word2_count == 0 {
            return 0.0;
        }

        let p_xy = bigram_count as f64 / self.total_tokens as f64;
        let p_x = word1_count as f64 / self.total_tokens as f64;
        let p_y = word2_count as f64 / self.total_tokens as f64;

        (p_xy / (p_x * p_y)).log2()
    }

    /// Extract candidate aspects based on frequency
    fn extract_candidates(&self, text: &str) -> Result<Vec<String>> {
        let tokens = utils::tokenize(text, true)?;
        let filtered = utils::remove_stopwords(tokens, None);

        let mut candidates = Vec::new();
        let freq = utils::word_frequency(&filtered);

        // Single words above frequency threshold
        for (word, count) in &freq {
            if *count >= self.min_frequency {
                candidates.push(word.clone());
            }
        }

        // Bigrams
        let bigrams = utils::get_ngrams(&filtered, 2)?;
        for bigram in bigrams {
            let pmi = self.calculate_pmi(&bigram);
            if pmi >= self.min_pmi {
                candidates.push(bigram);
            }
        }

        // Remove duplicates
        let unique: HashSet<_> = candidates.into_iter().collect();
        Ok(unique.into_iter().collect())
    }

    /// Calculate confidence score for an aspect candidate
    fn calculate_confidence(&self, aspect: &str) -> f64 {
        if self.total_tokens == 0 {
            // If not trained, use simpler heuristics
            let word_count = aspect.split_whitespace().count();
            return if word_count >= 2 { 0.7 } else { 0.6 };
        }

        // Use frequency-based confidence
        let words: Vec<&str> = aspect.split_whitespace().collect();

        if words.len() == 1 {
            let freq = self.word_freq.get(aspect).copied().unwrap_or(0);
            let normalized = (freq as f64 / self.total_tokens as f64).sqrt();
            (normalized * 100.0).min(1.0)
        } else if words.len() == 2 {
            let pmi = self.calculate_pmi(aspect);
            ((pmi + 5.0) / 10.0).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}

impl Default for StatisticalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AspectExtractor for StatisticalExtractor {
    fn extract(&self, text: &str) -> Result<Vec<Aspect>> {
        let candidates = self.extract_candidates(text)?;

        let mut aspects = Vec::new();

        for candidate in candidates {
            let confidence = self.calculate_confidence(&candidate);

            if confidence < self.min_confidence {
                continue;
            }

            let aspect = Aspect::new(candidate)
                .sentiment(Sentiment::Neutral)
                .confidence(confidence)?;

            aspects.push(aspect);
        }

        // Sort by confidence (descending)
        aspects.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(aspects)
    }

    fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Vec<Aspect>>> {
        // Parallel processing
        use rayon::prelude::*;

        texts
            .par_iter()
            .map(|text| self.extract(text))
            .collect()
    }

    fn name(&self) -> &'static str {
        "statistical"
    }

    fn description(&self) -> &'static str {
        "Frequency and collocation-based extraction using PMI"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistical_extractor_new() {
        let extractor = StatisticalExtractor::new();
        assert_eq!(extractor.min_frequency, 2);
    }

    #[test]
    fn test_extract_without_training() {
        let extractor = StatisticalExtractor::new();
        let text = "The camera quality is great and the camera is good";

        let aspects = extractor.extract(text).unwrap();

        // Should extract "camera" due to frequency
        assert!(!aspects.is_empty());
    }

    #[test]
    fn test_train_and_extract() {
        let mut extractor = StatisticalExtractor::new();

        let training_texts = vec![
            "The camera quality is excellent",
            "Camera quality matters a lot",
            "Battery life is important",
            "Battery life is crucial",
        ];

        extractor.train(&training_texts).unwrap();

        assert!(extractor.total_tokens > 0);
        assert!(!extractor.word_freq.is_empty());

        let text = "The camera quality and battery life are good";
        let aspects = extractor.extract(text).unwrap();

        assert!(!aspects.is_empty());
    }

    #[test]
    fn test_pmi_calculation() {
        let mut extractor = StatisticalExtractor::new();

        let texts = vec!["camera quality", "camera quality", "camera phone"];

        extractor.train(&texts).unwrap();

        let pmi = extractor.calculate_pmi("camera quality");
        assert!(pmi > 0.0);
    }

    #[test]
    fn test_confidence_scores() {
        let extractor = StatisticalExtractor::with_config(1, 0.0, 0.3);
        let text = "camera battery screen camera battery";

        let aspects = extractor.extract(text).unwrap();

        // All aspects should meet confidence threshold
        assert!(aspects.iter().all(|a| a.confidence >= 0.3));

        // Should be sorted by confidence
        for i in 1..aspects.len() {
            assert!(aspects[i - 1].confidence >= aspects[i].confidence);
        }
    }

    #[test]
    fn test_extract_batch() {
        let extractor = StatisticalExtractor::new();
        let texts = vec!["camera quality good", "battery life poor"];

        let results = extractor.extract_batch(&texts).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_extract_empty_text() {
        let extractor = StatisticalExtractor::new();
        let result = extractor.extract("");

        assert!(result.is_err());
    }
}
