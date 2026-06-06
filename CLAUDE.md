# CLAUDE.md - Project Philosophy & Guidelines

> "Simplicity is the ultimate sophistication." - Leonardo da Vinci

## The Vision

**aspect-extraction** is a production-ready, multi-model NLP system for extracting aspects (features, attributes, entities) from text. It's designed for opinion mining, sentiment analysis, and understanding what people truly care about in reviews and feedback.

This isn't just another research project. It's a tool that should feel inevitable - elegant, powerful, and actually useful in production.

## Core Principles

### 1. Elegance Over Complexity
- Every abstraction should feel natural
- If you can't explain it simply, you don't understand it well enough
- The best code is code you don't have to write

### 2. Production-First Mindset
- Everything is tested (>90% coverage)
- Everything is typed (mypy strict mode)
- Everything is documented
- Performance matters from day one

### 3. User-Centric Design
- APIs should be intuitive
- Error messages should be helpful
- Defaults should be sensible
- Documentation should teach, not just describe

### 4. Craft Excellence
- Code is read more than written - optimize for readability
- Consistency matters - follow established patterns
- Details matter - from docstrings to error handling
- Leave it better than you found it

## Architecture Philosophy

### Multi-Model Approach
We support multiple extraction methods because different use cases need different trade-offs:
- **Rule-based**: Fast, interpretable, no dependencies
- **Statistical**: Balanced performance, moderate resource usage
- **Transformer-based**: State-of-the-art accuracy, higher resource requirements

### Modular Design
```
aspect_extraction/
├── core/           # Core interfaces and base classes
├── extractors/     # Different extraction strategies
├── models/         # Data models and schemas
├── evaluation/     # Metrics and benchmarking
├── api/           # REST API (FastAPI)
├── cli/           # Command-line interface
└── utils/         # Shared utilities
```

### API Design Principles
1. **Consistency**: Same patterns across all endpoints
2. **Validation**: Strong input validation with helpful errors
3. **Performance**: Async by default, caching where appropriate
4. **Observability**: Logging, metrics, tracing built-in

## Code Quality Standards

### Type Hints
```python
# ✓ Good
def extract_aspects(text: str, method: str = "auto") -> list[Aspect]:
    """Extract aspects from text using specified method."""
    ...

# ✗ Bad
def extract_aspects(text, method="auto"):
    ...
```

### Documentation
```python
# ✓ Good
def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.

    Args:
        precision: Precision value between 0 and 1
        recall: Recall value between 0 and 1

    Returns:
        F1 score between 0 and 1

    Raises:
        ValueError: If precision or recall not in [0, 1]
    """
    if not (0 <= precision <= 1 and 0 <= recall <= 1):
        raise ValueError("Precision and recall must be between 0 and 1")

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
```

### Error Handling
```python
# ✓ Good - Specific, helpful errors
if not text.strip():
    raise ValueError("Input text cannot be empty")

# ✗ Bad - Generic, unhelpful
if not text.strip():
    raise Exception("Error")
```

### Testing
```python
# ✓ Good - Clear, comprehensive
def test_extract_aspects_empty_input():
    """Should raise ValueError for empty input."""
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_aspects("")

def test_extract_aspects_basic():
    """Should extract aspects from simple review."""
    text = "The camera quality is great but battery life is poor"
    aspects = extract_aspects(text)

    assert len(aspects) >= 2
    assert any(a.text == "camera quality" for a in aspects)
    assert any(a.text == "battery life" for a in aspects)
```

## Development Workflow

### Before Committing
```bash
# Format code
black aspect_extraction tests
isort aspect_extraction tests

# Type check
mypy aspect_extraction

# Lint
flake8 aspect_extraction tests
pylint aspect_extraction

# Test
pytest --cov=aspect_extraction --cov-report=term-missing
```

### Commit Messages
Follow conventional commits:
```
feat: add transformer-based aspect extractor
fix: handle empty input gracefully in rule-based extractor
docs: add examples for API usage
test: add integration tests for FastAPI endpoints
refactor: simplify aspect scoring logic
perf: cache model loading to improve throughput
```

## Dependencies Philosophy

### Be Selective
- Only add dependencies that provide significant value
- Prefer standard library when possible
- Consider maintenance status and community health
- Pin versions for reproducibility

### Core Stack (Justified)
- **FastAPI**: Best-in-class async web framework, auto docs
- **Pydantic**: Runtime validation, perfect FastAPI integration
- **spaCy**: Production-ready NLP, efficient
- **transformers**: State-of-the-art models, extensive ecosystem
- **pytest**: Industry standard, powerful fixtures
- **black**: Opinionated formatting, zero config needed

## Performance Considerations

### Model Loading
```python
# ✓ Good - Lazy loading, singleton pattern
class TransformerExtractor:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModel.from_pretrained(...)
        return cls._model
```

### Batch Processing
```python
# ✓ Good - Process in batches
def extract_aspects_batch(texts: list[str], batch_size: int = 32) -> list[list[Aspect]]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(self._process_batch(batch))
    return results
```

### Async Where It Matters
```python
# ✓ Good - I/O-bound operations
@app.post("/extract")
async def extract_endpoint(request: ExtractionRequest):
    # Async I/O operations
    result = await process_extraction(request.text)
    return result
```

## Documentation Standards

### README
- Clear project description
- Quick start guide
- Installation instructions
- Usage examples
- API reference link
- Contributing guidelines

### Code Comments
Use comments to explain **why**, not **what**:
```python
# ✓ Good
# Use sliding window to handle texts longer than model's max length
# without losing context at boundaries
chunks = create_overlapping_chunks(text, window_size=512, overlap=50)

# ✗ Bad
# Create chunks
chunks = create_overlapping_chunks(text, 512, 50)
```

## The Ultrathink Mindset

When approaching any task:

1. **Question Everything**: Why this way? What if we started fresh?
2. **See the Whole**: How does this fit into the larger system?
3. **Simplify Relentlessly**: What can we remove without losing power?
4. **Craft With Care**: Is this the most elegant solution?
5. **Test Rigorously**: How do we know it works?
6. **Iterate Fearlessly**: Good enough isn't good enough

## Success Criteria

A feature is complete when:
- [ ] It solves the real problem elegantly
- [ ] It has comprehensive tests (>90% coverage)
- [ ] It passes all linters and type checks
- [ ] It's documented with examples
- [ ] It handles errors gracefully
- [ ] It performs acceptably
- [ ] It's consistent with existing patterns
- [ ] You're proud to show it to others

---

*"The people who are crazy enough to think they can change the world are the ones who do."*
Let's build something insanely great.
