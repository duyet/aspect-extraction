"""Request models for the API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from aspect_extraction.core.factory import ExtractorMethod


class ExtractionRequest(BaseModel):
    """Request model for aspect extraction.

    Attributes:
        text: Text to extract aspects from
        method: Extraction method to use
        include_sentiment: Whether to include sentiment analysis
        min_confidence: Minimum confidence threshold (0-1)
    """

    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    method: ExtractorMethod = Field(
        default="auto", description="Extraction method: rule, statistical, transformer, or auto"
    )
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate that text is not empty after stripping."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "text": "The camera quality is excellent but battery life is poor",
                "method": "auto",
                "include_sentiment": True,
                "min_confidence": 0.5,
            }
        }


class BatchExtractionRequest(BaseModel):
    """Request model for batch aspect extraction.

    Attributes:
        texts: List of texts to extract aspects from
        method: Extraction method to use
        include_sentiment: Whether to include sentiment analysis
        min_confidence: Minimum confidence threshold (0-1)
        batch_size: Batch size for processing
    """

    texts: List[str] = Field(..., min_length=1, max_length=100, description="Texts to analyze")
    method: ExtractorMethod = Field(
        default="auto", description="Extraction method: rule, statistical, transformer, or auto"
    )
    include_sentiment: bool = Field(default=True, description="Include sentiment analysis")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    batch_size: int = Field(default=32, ge=1, le=100, description="Batch size for processing")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        """Validate that all texts are non-empty."""
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty or whitespace only")
        return v

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "texts": [
                    "The camera quality is excellent",
                    "Battery life is poor",
                ],
                "method": "auto",
                "include_sentiment": True,
                "min_confidence": 0.5,
                "batch_size": 32,
            }
        }


class EvaluationRequest(BaseModel):
    """Request model for evaluating aspect extraction.

    Attributes:
        predicted_aspects: List of predicted aspect texts
        ground_truth_aspects: List of ground truth aspect texts
        match_sentiment: Whether to require sentiment match
    """

    predicted_aspects: List[str] = Field(..., description="Predicted aspect texts")
    ground_truth_aspects: List[str] = Field(..., description="Ground truth aspect texts")
    match_sentiment: bool = Field(default=False, description="Require sentiment match for evaluation")

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "predicted_aspects": ["battery life", "camera quality"],
                "ground_truth_aspects": ["battery life", "screen quality"],
                "match_sentiment": False,
            }
        }
