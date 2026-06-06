//! Rule-based aspect extraction using linguistic patterns.
//!
//! This extractor uses regex patterns and simple sentiment analysis
//! to extract aspects from text without requiring ML models.

use crate::error::{Error, Result};
use crate::extractor::AspectExtractor;
use crate::types::{Aspect, Sentiment};
use crate::utils;
use regex::Regex;
use std::collections::HashMap;

/// Rule-based aspect extractor
pub struct RuleBasedExtractor {
    /// Minimum confidence threshold (0.0 to 1.0)
    min_confidence: f64,

    /// Regex patterns for aspect extraction
    patterns: Vec<Regex>,

    /// Sentiment lexicon (positive words)
    positive_words: HashMap<String, f64>,

    /// Sentiment lexicon (negative words)
    negative_words: HashMap<String, f64>,
}

impl RuleBasedExtractor {
    /// Create a new rule-based extractor with default configuration
    pub fn new() -> Self {
        Self::with_config(0.0)
    }

    /// Create a new rule-based extractor with custom configuration
    pub fn with_config(min_confidence: f64) -> Self {
        Self {
            min_confidence,
            patterns: Self::default_patterns(),
            positive_words: Self::default_positive_words(),
            negative_words: Self::default_negative_words(),
        }
    }

    /// Get default extraction patterns
    fn default_patterns() -> Vec<Regex> {
        vec![
            // Noun phrases: "the <aspect>"
            Regex::new(r"\b(the|a|an)\s+([a-z]+(?:\s+[a-z]+){0,2})\b").unwrap(),
            // Adjective + noun: "<adj> <aspect>"
            Regex::new(r"\b([a-z]+)\s+(quality|life|size|speed|performance|design|price)\b")
                .unwrap(),
            // Possessive: "its <aspect>", "the <aspect>"
            Regex::new(r"\b(its|their|my|your)\s+([a-z]+)\b").unwrap(),
        ]
    }

    /// Get default positive sentiment words
    fn default_positive_words() -> HashMap<String, f64> {
        let words = vec![
            ("good", 0.7),
            ("great", 0.9),
            ("excellent", 1.0),
            ("amazing", 1.0),
            ("fantastic", 1.0),
            ("awesome", 0.9),
            ("wonderful", 0.9),
            ("nice", 0.6),
            ("love", 1.0),
            ("perfect", 1.0),
            ("best", 1.0),
            ("outstanding", 1.0),
            ("superb", 0.9),
            ("brilliant", 0.9),
            ("positive", 0.7),
        ];

        words
            .into_iter()
            .map(|(w, s)| (w.to_string(), s))
            .collect()
    }

    /// Get default negative sentiment words
    fn default_negative_words() -> HashMap<String, f64> {
        let words = vec![
            ("bad", 0.7),
            ("poor", 0.8),
            ("terrible", 1.0),
            ("awful", 1.0),
            ("horrible", 1.0),
            ("disappointing", 0.8),
            ("worst", 1.0),
            ("hate", 1.0),
            ("useless", 0.9),
            ("pathetic", 0.9),
            ("negative", 0.7),
            ("mediocre", 0.6),
            ("inadequate", 0.7),
        ];

        words
            .into_iter()
            .map(|(w, s)| (w.to_string(), s))
            .collect()
    }

    /// Analyze sentiment for a given context
    fn analyze_sentiment(&self, context: &str) -> Option<Sentiment> {
        let tokens = utils::tokenize(context, true).ok()?;

        let mut pos_score = 0.0;
        let mut neg_score = 0.0;

        for token in &tokens {
            if let Some(score) = self.positive_words.get(token) {
                pos_score += score;
            }
            if let Some(score) = self.negative_words.get(token) {
                neg_score += score;
            }
        }

        if pos_score > neg_score && pos_score > 0.0 {
            Some(Sentiment::Positive)
        } else if neg_score > pos_score && neg_score > 0.0 {
            Some(Sentiment::Negative)
        } else if pos_score == 0.0 && neg_score == 0.0 {
            None
        } else {
            Some(Sentiment::Neutral)
        }
    }

    /// Extract context window around a match
    fn extract_context(&self, text: &str, start: usize, end: usize, window_size: usize) -> String {
        let context_start = start.saturating_sub(window_size);
        let context_end = (end + window_size).min(text.len());

        text[context_start..context_end].to_string()
    }

    /// Calculate confidence score based on pattern match and context
    fn calculate_confidence(&self, aspect_text: &str, context: &str) -> f64 {
        let mut confidence = 0.7; // Base confidence for pattern match

        // Boost if sentiment words are present
        let tokens = utils::tokenize(context, true).unwrap_or_default();
        let has_sentiment = tokens.iter().any(|t| {
            self.positive_words.contains_key(t) || self.negative_words.contains_key(t)
        });

        if has_sentiment {
            confidence += 0.2;
        }

        // Boost for longer aspects (more specific)
        let word_count = aspect_text.split_whitespace().count();
        if word_count >= 2 {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }
}

impl Default for RuleBasedExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl AspectExtractor for RuleBasedExtractor {
    fn extract(&self, text: &str) -> Result<Vec<Aspect>> {
        let cleaned = utils::clean_text(text, true)?;

        let mut aspects = Vec::new();
        let mut seen = HashMap::new();

        for pattern in &self.patterns {
            for cap in pattern.captures_iter(&cleaned) {
                // Get the aspect text (usually the last capture group)
                let aspect_text = cap
                    .get(cap.len() - 1)
                    .map(|m| m.as_str())
                    .unwrap_or_default()
                    .trim()
                    .to_string();

                if aspect_text.is_empty() || aspect_text.len() < 2 {
                    continue;
                }

                // Skip if we've already seen this aspect
                if seen.contains_key(&aspect_text) {
                    continue;
                }

                // Get match position
                let match_obj = cap.get(0).unwrap();
                let start = match_obj.start();
                let end = match_obj.end();

                // Extract context
                let context = self.extract_context(&cleaned, start, end, 50);

                // Analyze sentiment
                let sentiment = self.analyze_sentiment(&context);

                // Calculate confidence
                let confidence = self.calculate_confidence(&aspect_text, &context);

                // Skip if below confidence threshold
                if confidence < self.min_confidence {
                    continue;
                }

                let aspect = Aspect::new(aspect_text.clone())
                    .sentiment(sentiment.unwrap_or(Sentiment::Neutral))
                    .position(start, end)
                    .context(context)
                    .confidence(confidence)?;

                seen.insert(aspect_text, ());
                aspects.push(aspect);
            }
        }

        Ok(aspects)
    }

    fn extract_batch(&self, texts: &[&str]) -> Result<Vec<Vec<Aspect>>> {
        // Parallel processing using rayon
        use rayon::prelude::*;

        texts
            .par_iter()
            .map(|text| self.extract(text))
            .collect()
    }

    fn name(&self) -> &'static str {
        "rule-based"
    }

    fn description(&self) -> &'static str {
        "Fast rule-based extraction using regex patterns and sentiment lexicon"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_based_extractor_new() {
        let extractor = RuleBasedExtractor::new();
        assert_eq!(extractor.min_confidence, 0.0);
    }

    #[test]
    fn test_extract_basic() {
        let extractor = RuleBasedExtractor::new();
        let text = "The camera quality is excellent but the battery life is poor";

        let aspects = extractor.extract(text).unwrap();

        assert!(!aspects.is_empty());
        // Should find "camera quality" and "battery life"
        let aspect_texts: Vec<&str> = aspects.iter().map(|a| a.text.as_str()).collect();
        assert!(aspect_texts.contains(&"camera quality") || aspect_texts.contains(&"quality"));
    }

    #[test]
    fn test_sentiment_analysis() {
        let extractor = RuleBasedExtractor::new();

        assert_eq!(
            extractor.analyze_sentiment("this is excellent"),
            Some(Sentiment::Positive)
        );
        assert_eq!(
            extractor.analyze_sentiment("this is terrible"),
            Some(Sentiment::Negative)
        );
        assert_eq!(extractor.analyze_sentiment("this is something"), None);
    }

    #[test]
    fn test_min_confidence_threshold() {
        let extractor = RuleBasedExtractor::with_config(0.95);
        let text = "The camera is nice";

        let aspects = extractor.extract(text).unwrap();

        // With high threshold, might filter out some aspects
        assert!(aspects.iter().all(|a| a.confidence >= 0.95));
    }

    #[test]
    fn test_extract_batch() {
        let extractor = RuleBasedExtractor::new();
        let texts = vec!["The camera is good", "The battery is bad"];

        let results = extractor.extract_batch(&texts).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_extract_empty_text() {
        let extractor = RuleBasedExtractor::new();
        let result = extractor.extract("");

        assert!(result.is_err());
    }

    #[test]
    fn test_no_duplicate_aspects() {
        let extractor = RuleBasedExtractor::new();
        let text = "The battery is good and the battery is excellent";

        let aspects = extractor.extract(text).unwrap();

        // Should only extract "battery" once
        let aspect_texts: Vec<&str> = aspects.iter().map(|a| a.text.as_str()).collect();
        let battery_count = aspect_texts.iter().filter(|&&t| t == "battery").count();

        assert!(battery_count <= 1);
    }
}
