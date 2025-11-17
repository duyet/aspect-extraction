//! Aspect extraction implementations.
//!
//! This module contains different strategies for extracting aspects from text:
//!
//! - [`rule_based`]: Fast, interpretable rule-based extraction
//! - [`statistical`]: Frequency and collocation-based extraction

pub mod rule_based;
pub mod statistical;
