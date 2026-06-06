//! Utility functions for text processing.

use crate::error::{Error, Result};
use unicode_segmentation::UnicodeSegmentation;

/// Normalize whitespace in text (collapse multiple spaces, newlines, etc.)
pub fn normalize_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Clean text by normalizing whitespace and optionally lowercasing
pub fn clean_text(text: &str, lowercase: bool) -> Result<String> {
    let trimmed = text.trim();

    if trimmed.is_empty() {
        return Err(Error::EmptyText);
    }

    let normalized = normalize_whitespace(trimmed);

    Ok(if lowercase {
        normalized.to_lowercase()
    } else {
        normalized
    })
}

/// Tokenize text into words
pub fn tokenize(text: &str, lowercase: bool) -> Result<Vec<String>> {
    if text.trim().is_empty() {
        return Err(Error::EmptyText);
    }

    let tokens: Vec<String> = text
        .unicode_words()
        .map(|word| {
            if lowercase {
                word.to_lowercase()
            } else {
                word.to_string()
            }
        })
        .collect();

    Ok(tokens)
}

/// Extract sentences from text
pub fn extract_sentences(text: &str) -> Result<Vec<String>> {
    if text.trim().is_empty() {
        return Err(Error::EmptyText);
    }

    // Simple sentence splitting on common punctuation
    // For production, consider using a proper sentence tokenizer
    let sentences: Vec<String> = text
        .split(|c| c == '.' || c == '!' || c == '?')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    Ok(sentences)
}

/// Generate n-grams from tokens
pub fn get_ngrams(tokens: &[String], n: usize) -> Result<Vec<String>> {
    if tokens.is_empty() {
        return Err(Error::InvalidInput("Tokens cannot be empty".to_string()));
    }

    if n == 0 {
        return Err(Error::InvalidNgramSize(n));
    }

    if n > tokens.len() {
        return Ok(Vec::new());
    }

    let ngrams = tokens
        .windows(n)
        .map(|window| window.join(" "))
        .collect();

    Ok(ngrams)
}

/// Remove common English stopwords from tokens
pub fn remove_stopwords(tokens: Vec<String>, custom_stopwords: Option<&[&str]>) -> Vec<String> {
    // Common English stopwords
    const DEFAULT_STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
        "it", "its", "of", "on", "that", "the", "to", "was", "will", "with",
    ];

    let stopwords = custom_stopwords.unwrap_or(DEFAULT_STOPWORDS);

    tokens
        .into_iter()
        .filter(|token| {
            !stopwords
                .iter()
                .any(|sw| sw.eq_ignore_ascii_case(token))
        })
        .collect()
}

/// Truncate text to a maximum length, adding a suffix if truncated
pub fn truncate_text(text: &str, max_length: usize, suffix: &str) -> Result<String> {
    if max_length < suffix.len() {
        return Err(Error::InvalidInput(format!(
            "max_length ({}) must be at least as long as suffix length ({})",
            max_length,
            suffix.len()
        )));
    }

    if text.len() <= max_length {
        return Ok(text.to_string());
    }

    let truncate_at = max_length - suffix.len();
    Ok(format!("{}{}", &text[..truncate_at], suffix))
}

/// Calculate word frequency from a list of tokens
pub fn word_frequency(tokens: &[String]) -> hashbrown::HashMap<String, usize> {
    let mut freq = hashbrown::HashMap::new();

    for token in tokens {
        *freq.entry(token.clone()).or_insert(0) += 1;
    }

    freq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("hello\n\nworld"), "hello world");
        assert_eq!(normalize_whitespace("hello\tworld"), "hello world");
    }

    #[test]
    fn test_clean_text() {
        assert_eq!(clean_text("  Hello  World  ", false).unwrap(), "Hello World");
        assert_eq!(clean_text("  Hello  World  ", true).unwrap(), "hello world");
    }

    #[test]
    fn test_clean_text_empty() {
        assert!(clean_text("", false).is_err());
        assert!(clean_text("   ", false).is_err());
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello world!", false).unwrap();
        assert_eq!(tokens, vec!["Hello", "world"]);

        let tokens = tokenize("Hello World!", true).unwrap();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_extract_sentences() {
        let sentences = extract_sentences("Hello world. How are you?").unwrap();
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "Hello world");
        assert_eq!(sentences[1], "How are you");
    }

    #[test]
    fn test_get_ngrams() {
        let tokens = vec!["the", "camera", "quality"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let bigrams = get_ngrams(&tokens, 2).unwrap();
        assert_eq!(bigrams, vec!["the camera", "camera quality"]);

        let trigrams = get_ngrams(&tokens, 3).unwrap();
        assert_eq!(trigrams, vec!["the camera quality"]);
    }

    #[test]
    fn test_get_ngrams_invalid() {
        let tokens: Vec<String> = vec![];
        assert!(get_ngrams(&tokens, 2).is_err());

        let tokens = vec!["a".to_string()];
        assert!(get_ngrams(&tokens, 0).is_err());
    }

    #[test]
    fn test_remove_stopwords() {
        let tokens = vec!["the", "camera", "is", "good"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let filtered = remove_stopwords(tokens, None);
        assert_eq!(filtered, vec!["camera", "good"]);
    }

    #[test]
    fn test_remove_stopwords_custom() {
        let tokens = vec!["hello", "world", "foo"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let filtered = remove_stopwords(tokens, Some(&["hello"]));
        assert_eq!(filtered, vec!["world", "foo"]);
    }

    #[test]
    fn test_truncate_text() {
        assert_eq!(truncate_text("Hello world", 8, "...").unwrap(), "Hello...");
        assert_eq!(truncate_text("Hello", 10, "...").unwrap(), "Hello");
    }

    #[test]
    fn test_truncate_text_invalid() {
        assert!(truncate_text("Hello", 2, "...").is_err());
    }

    #[test]
    fn test_word_frequency() {
        let tokens = vec!["camera", "battery", "camera"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();

        let freq = word_frequency(&tokens);
        assert_eq!(freq.get("camera"), Some(&2));
        assert_eq!(freq.get("battery"), Some(&1));
    }
}
