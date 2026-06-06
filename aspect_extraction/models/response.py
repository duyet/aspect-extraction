"""Response models for the API."""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from aspect_extraction.core.aspect import Sentiment


class AspectResponse(BaseModel):
    """Response model for a single aspect.

    Attributes:
        text: The aspect text
        category: Optional category label
        sentiment: Optional sentiment polarity
        confidence: Confidence score
        start_pos: Starting character position
        end_pos: Ending character position
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "battery life",
                "category": "hardware",
                "sentiment": "negative",
                "confidence": 0.95,
                "start_pos": 35,
                "end_pos": 47,
            }
        }
    )

    text: str = Field(..., description="Aspect text")
    category: Optional[str] = Field(None, description="Aspect category")
    sentiment: Optional[Sentiment] = Field(None, description="Sentiment polarity")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    start_pos: Optional[int] = Field(None, description="Start position in text")
    end_pos: Optional[int] = Field(None, description="End position in text")


class ExtractionResponse(BaseModel):
    """Response model for aspect extraction.

    Attributes:
        aspects: List of extracted aspects
        method_used: Extraction method that was used
        processing_time_ms: Processing time in milliseconds
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "aspects": [
                    {
                        "text": "camera quality",
                        "sentiment": "positive",
                        "confidence": 0.92,
                    },
                    {
                        "text": "battery life",
                        "sentiment": "negative",
                        "confidence": 0.88,
                    },
                ],
                "method_used": "rule",
                "processing_time_ms": 45.2,
            }
        }
    )

    aspects: List[AspectResponse] = Field(..., description="Extracted aspects")
    method_used: str = Field(..., description="Extraction method used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class BatchExtractionResponse(BaseModel):
    """Response model for batch aspect extraction.

    Attributes:
        results: List of extraction results
        total_texts: Total number of texts processed
        total_aspects: Total number of aspects extracted
        method_used: Extraction method that was used
        processing_time_ms: Total processing time in milliseconds
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    [{"text": "camera quality", "sentiment": "positive", "confidence": 0.92}],
                    [{"text": "battery life", "sentiment": "negative", "confidence": 0.88}],
                ],
                "total_texts": 2,
                "total_aspects": 2,
                "method_used": "rule",
                "processing_time_ms": 78.5,
            }
        }
    )

    results: List[List[AspectResponse]] = Field(..., description="Extraction results for each text")
    total_texts: int = Field(..., description="Total number of texts processed")
    total_aspects: int = Field(..., description="Total number of aspects extracted")
    method_used: str = Field(..., description="Extraction method used")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class EvaluationResponse(BaseModel):
    """Response model for evaluation results.

    Attributes:
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        true_positives: Number of correct predictions
        false_positives: Number of incorrect predictions
        false_negatives: Number of missed predictions
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "precision": 0.85,
                "recall": 0.78,
                "f1_score": 0.81,
                "true_positives": 15,
                "false_positives": 3,
                "false_negatives": 4,
            }
        }
    )

    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    true_positives: int = Field(..., ge=0, description="True positives")
    false_positives: int = Field(..., ge=0, description="False positives")
    false_negatives: int = Field(..., ge=0, description="False negatives")


class HealthResponse(BaseModel):
    """Response model for health check.

    Attributes:
        status: Service status
        version: API version
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "0.1.0",
            }
        }
    )

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
