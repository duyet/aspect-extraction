//! Core data types for aspect extraction.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{Error, Result};

/// Sentiment polarity of an aspect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sentiment {
    /// Positive sentiment
    Positive,
    /// Negative sentiment
    Negative,
    /// Neutral sentiment (neither positive nor negative)
    Neutral,
}

impl fmt::Display for Sentiment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sentiment::Positive => write!(f, "positive"),
            Sentiment::Negative => write!(f, "negative"),
            Sentiment::Neutral => write!(f, "neutral"),
        }
    }
}

impl Sentiment {
    /// Convert from string representation
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "positive" => Some(Sentiment::Positive),
            "negative" => Some(Sentiment::Negative),
            "neutral" => Some(Sentiment::Neutral),
            _ => None,
        }
    }

    /// Get numeric score: -1.0 for negative, 0.0 for neutral, 1.0 for positive
    pub fn score(&self) -> f64 {
        match self {
            Sentiment::Positive => 1.0,
            Sentiment::Negative => -1.0,
            Sentiment::Neutral => 0.0,
        }
    }
}

/// An aspect extracted from text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Aspect {
    /// The aspect text
    pub text: String,

    /// Category of the aspect (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,

    /// Sentiment polarity (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentiment: Option<Sentiment>,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Start position in original text (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_pos: Option<usize>,

    /// End position in original text (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_pos: Option<usize>,

    /// Context window around the aspect (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

impl Aspect {
    /// Create a new aspect with just text and default confidence
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            category: None,
            sentiment: None,
            confidence: 1.0,
            start_pos: None,
            end_pos: None,
            context: None,
        }
    }

    /// Create a new aspect with text and confidence
    pub fn with_confidence(text: impl Into<String>, confidence: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(Error::InvalidConfidence(confidence));
        }

        Ok(Self {
            text: text.into(),
            category: None,
            sentiment: None,
            confidence,
            start_pos: None,
            end_pos: None,
            context: None,
        })
    }

    /// Builder pattern: set category
    pub fn category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Builder pattern: set sentiment
    pub fn sentiment(mut self, sentiment: Sentiment) -> Self {
        self.sentiment = Some(sentiment);
        self
    }

    /// Builder pattern: set confidence
    pub fn confidence(mut self, confidence: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(Error::InvalidConfidence(confidence));
        }
        self.confidence = confidence;
        Ok(self)
    }

    /// Builder pattern: set position
    pub fn position(mut self, start: usize, end: usize) -> Self {
        self.start_pos = Some(start);
        self.end_pos = Some(end);
        self
    }

    /// Builder pattern: set context
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    /// Check if this aspect matches another (case-insensitive text comparison)
    pub fn matches(&self, other: &Aspect, match_sentiment: bool) -> bool {
        let text_match = self.text.to_lowercase() == other.text.to_lowercase();

        if !match_sentiment {
            return text_match;
        }

        text_match && self.sentiment == other.sentiment
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(Into::into)
    }
}

impl fmt::Display for Aspect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)?;
        if let Some(sentiment) = &self.sentiment {
            write!(f, " ({})", sentiment)?;
        }
        write!(f, " [conf: {:.2}]", self.confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_from_string() {
        assert_eq!(Sentiment::from_str("positive"), Some(Sentiment::Positive));
        assert_eq!(Sentiment::from_str("NEGATIVE"), Some(Sentiment::Negative));
        assert_eq!(Sentiment::from_str("neutral"), Some(Sentiment::Neutral));
        assert_eq!(Sentiment::from_str("invalid"), None);
    }

    #[test]
    fn test_sentiment_score() {
        assert_eq!(Sentiment::Positive.score(), 1.0);
        assert_eq!(Sentiment::Negative.score(), -1.0);
        assert_eq!(Sentiment::Neutral.score(), 0.0);
    }

    #[test]
    fn test_aspect_new() {
        let aspect = Aspect::new("battery life");
        assert_eq!(aspect.text, "battery life");
        assert_eq!(aspect.confidence, 1.0);
        assert_eq!(aspect.sentiment, None);
    }

    #[test]
    fn test_aspect_builder() {
        let aspect = Aspect::new("camera quality")
            .category("hardware")
            .sentiment(Sentiment::Positive)
            .position(5, 20)
            .context("The camera quality is excellent")
            .confidence(0.95)
            .unwrap();

        assert_eq!(aspect.text, "camera quality");
        assert_eq!(aspect.category, Some("hardware".to_string()));
        assert_eq!(aspect.sentiment, Some(Sentiment::Positive));
        assert_eq!(aspect.confidence, 0.95);
        assert_eq!(aspect.start_pos, Some(5));
        assert_eq!(aspect.end_pos, Some(20));
    }

    #[test]
    fn test_aspect_invalid_confidence() {
        let result = Aspect::with_confidence("test", 1.5);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidConfidence(_)));
    }

    #[test]
    fn test_aspect_matches() {
        let aspect1 = Aspect::new("Battery Life").sentiment(Sentiment::Positive);
        let aspect2 = Aspect::new("battery life").sentiment(Sentiment::Positive);
        let aspect3 = Aspect::new("battery life").sentiment(Sentiment::Negative);

        // Without sentiment matching
        assert!(aspect1.matches(&aspect2, false));
        assert!(aspect1.matches(&aspect3, false));

        // With sentiment matching
        assert!(aspect1.matches(&aspect2, true));
        assert!(!aspect1.matches(&aspect3, true));
    }

    #[test]
    fn test_aspect_serialization() {
        let aspect = Aspect::new("camera")
            .sentiment(Sentiment::Positive)
            .confidence(0.9)
            .unwrap();

        let json = aspect.to_json().unwrap();
        let deserialized = Aspect::from_json(&json).unwrap();

        assert_eq!(aspect, deserialized);
    }

    #[test]
    fn test_aspect_display() {
        let aspect = Aspect::new("battery")
            .sentiment(Sentiment::Negative)
            .confidence(0.85)
            .unwrap();

        let display = format!("{}", aspect);
        assert!(display.contains("battery"));
        assert!(display.contains("negative"));
        assert!(display.contains("0.85"));
    }
}
