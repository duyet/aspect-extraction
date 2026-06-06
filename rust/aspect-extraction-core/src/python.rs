//! Python bindings using PyO3.
//!
//! This module exposes the Rust aspect extraction functionality to Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::extractor::AspectExtractor as RustAspectExtractor;
use crate::extractors::{RuleBasedExtractor as RustRuleBasedExtractor, StatisticalExtractor as RustStatisticalExtractor};
use crate::types::{Aspect as RustAspect, Sentiment as RustSentiment};

/// Python-facing Sentiment enum
#[pyclass(name = "Sentiment")]
#[derive(Clone)]
pub struct PySentiment {
    inner: RustSentiment,
}

#[pymethods]
impl PySentiment {
    #[new]
    fn new(value: &str) -> PyResult<Self> {
        let inner = RustSentiment::from_str(value)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid sentiment: {}", value)))?;

        Ok(Self { inner })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("Sentiment('{}')", self.inner)
    }

    #[getter]
    fn score(&self) -> f64 {
        self.inner.score()
    }
}

/// Python-facing Aspect class
#[pyclass(name = "Aspect")]
#[derive(Clone)]
pub struct PyAspect {
    #[pyo3(get, set)]
    text: String,

    #[pyo3(get, set)]
    category: Option<String>,

    #[pyo3(get, set)]
    confidence: f64,

    #[pyo3(get, set)]
    start_pos: Option<usize>,

    #[pyo3(get, set)]
    end_pos: Option<usize>,

    #[pyo3(get, set)]
    context: Option<String>,

    sentiment: Option<RustSentiment>,
}

impl From<RustAspect> for PyAspect {
    fn from(aspect: RustAspect) -> Self {
        Self {
            text: aspect.text,
            category: aspect.category,
            confidence: aspect.confidence,
            start_pos: aspect.start_pos,
            end_pos: aspect.end_pos,
            context: aspect.context,
            sentiment: aspect.sentiment,
        }
    }
}

#[pymethods]
impl PyAspect {
    #[new]
    #[pyo3(signature = (text, category=None, sentiment=None, confidence=1.0))]
    fn new(
        text: String,
        category: Option<String>,
        sentiment: Option<&PySentiment>,
        confidence: f64,
    ) -> PyResult<Self> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(PyValueError::new_err(format!(
                "Confidence must be between 0.0 and 1.0, got {}",
                confidence
            )));
        }

        Ok(Self {
            text,
            category,
            confidence,
            start_pos: None,
            end_pos: None,
            context: None,
            sentiment: sentiment.map(|s| s.inner),
        })
    }

    #[getter]
    fn sentiment(&self) -> Option<PySentiment> {
        self.sentiment.map(|s| PySentiment { inner: s })
    }

    #[setter]
    fn set_sentiment(&mut self, sentiment: Option<&PySentiment>) {
        self.sentiment = sentiment.map(|s| s.inner);
    }

    fn __str__(&self) -> String {
        if let Some(sentiment) = &self.sentiment {
            format!("{} ({}) [conf: {:.2}]", self.text, sentiment, self.confidence)
        } else {
            format!("{} [conf: {:.2}]", self.text, self.confidence)
        }
    }

    fn __repr__(&self) -> String {
        format!("Aspect(text='{}', confidence={:.2})", self.text, self.confidence)
    }

    fn to_dict(&self) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("text", &self.text)?;
            dict.set_item("confidence", self.confidence)?;

            if let Some(category) = &self.category {
                dict.set_item("category", category)?;
            }

            if let Some(sentiment) = &self.sentiment {
                dict.set_item("sentiment", sentiment.to_string())?;
            }

            if let Some(start_pos) = self.start_pos {
                dict.set_item("start_pos", start_pos)?;
            }

            if let Some(end_pos) = self.end_pos {
                dict.set_item("end_pos", end_pos)?;
            }

            if let Some(context) = &self.context {
                dict.set_item("context", context)?;
            }

            Ok(dict.into())
        })
    }
}

/// Python-facing RuleBasedExtractor
#[pyclass(name = "RuleBasedExtractor")]
pub struct PyRuleBasedExtractor {
    inner: RustRuleBasedExtractor,
}

#[pymethods]
impl PyRuleBasedExtractor {
    #[new]
    #[pyo3(signature = (min_confidence=0.0))]
    fn new(min_confidence: f64) -> Self {
        Self {
            inner: RustRuleBasedExtractor::with_config(min_confidence),
        }
    }

    fn extract(&self, text: &str) -> PyResult<Vec<PyAspect>> {
        let aspects = self
            .inner
            .extract(text)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(aspects.into_iter().map(PyAspect::from).collect())
    }

    fn extract_batch(&self, texts: Vec<&str>) -> PyResult<Vec<Vec<PyAspect>>> {
        let results = self
            .inner
            .extract_batch(&texts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|aspects| aspects.into_iter().map(PyAspect::from).collect())
            .collect())
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }
}

/// Python-facing StatisticalExtractor
#[pyclass(name = "StatisticalExtractor")]
pub struct PyStatisticalExtractor {
    inner: RustStatisticalExtractor,
}

#[pymethods]
impl PyStatisticalExtractor {
    #[new]
    #[pyo3(signature = (min_frequency=2, min_pmi=0.0, min_confidence=0.5))]
    fn new(min_frequency: usize, min_pmi: f64, min_confidence: f64) -> Self {
        Self {
            inner: RustStatisticalExtractor::with_config(min_frequency, min_pmi, min_confidence),
        }
    }

    fn extract(&self, text: &str) -> PyResult<Vec<PyAspect>> {
        let aspects = self
            .inner
            .extract(text)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(aspects.into_iter().map(PyAspect::from).collect())
    }

    fn extract_batch(&self, texts: Vec<&str>) -> PyResult<Vec<Vec<PyAspect>>> {
        let results = self
            .inner
            .extract_batch(&texts)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(results
            .into_iter()
            .map(|aspects| aspects.into_iter().map(PyAspect::from).collect())
            .collect())
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }
}

/// Module initialization
#[pymodule]
fn aspect_extraction_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySentiment>()?;
    m.add_class::<PyAspect>()?;
    m.add_class::<PyRuleBasedExtractor>()?;
    m.add_class::<PyStatisticalExtractor>()?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
