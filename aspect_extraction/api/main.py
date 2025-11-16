"""FastAPI application for aspect extraction."""

import time
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aspect_extraction import __version__
from aspect_extraction.core.aspect import Aspect
from aspect_extraction.core.factory import create_extractor
from aspect_extraction.evaluation.metrics import evaluate_extraction
from aspect_extraction.models.request import (
    BatchExtractionRequest,
    EvaluationRequest,
    ExtractionRequest,
)
from aspect_extraction.models.response import (
    AspectResponse,
    BatchExtractionResponse,
    EvaluationResponse,
    ExtractionResponse,
    HealthResponse,
)

# Create FastAPI app
app = FastAPI(
    title="Aspect Extraction API",
    description="Production-ready API for extracting aspects from text using multiple methods",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache extractors to avoid recreating them
_extractors_cache: Dict[str, object] = {}


def _aspect_to_response(aspect: Aspect) -> AspectResponse:
    """Convert Aspect to AspectResponse.

    Args:
        aspect: Aspect instance

    Returns:
        AspectResponse instance
    """
    return AspectResponse(
        text=aspect.text,
        category=aspect.category,
        sentiment=aspect.sentiment,
        confidence=aspect.confidence,
        start_pos=aspect.start_pos,
        end_pos=aspect.end_pos,
    )


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root endpoint returning service health.

    Returns:
        Health status and version
    """
    return HealthResponse(status="healthy", version=__version__)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status and version
    """
    return HealthResponse(status="healthy", version=__version__)


@app.post("/extract", response_model=ExtractionResponse, status_code=status.HTTP_200_OK)
async def extract_aspects(request: ExtractionRequest) -> ExtractionResponse:
    """Extract aspects from text.

    Args:
        request: Extraction request containing text and parameters

    Returns:
        Extraction results with aspects

    Raises:
        HTTPException: If extraction fails
    """
    try:
        start_time = time.time()

        # Create or get cached extractor
        cache_key = f"{request.method}"
        if cache_key not in _extractors_cache:
            _extractors_cache[cache_key] = create_extractor(method=request.method)

        extractor = _extractors_cache[cache_key]

        # Extract aspects
        aspects = extractor.extract(request.text)  # type: ignore

        # Filter by confidence
        if request.min_confidence > 0:
            aspects = [a for a in aspects if a.confidence >= request.min_confidence]

        # Remove sentiment if not requested
        if not request.include_sentiment:
            aspects = [
                Aspect(
                    text=a.text,
                    category=a.category,
                    sentiment=None,
                    confidence=a.confidence,
                    start_pos=a.start_pos,
                    end_pos=a.end_pos,
                    context=a.context,
                )
                for a in aspects
            ]

        # Convert to response format
        aspect_responses = [_aspect_to_response(a) for a in aspects]

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        return ExtractionResponse(
            aspects=aspect_responses,
            method_used=request.method,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}",
        )


@app.post(
    "/extract/batch",
    response_model=BatchExtractionResponse,
    status_code=status.HTTP_200_OK,
)
async def extract_aspects_batch(request: BatchExtractionRequest) -> BatchExtractionResponse:
    """Extract aspects from multiple texts.

    Args:
        request: Batch extraction request

    Returns:
        Batch extraction results

    Raises:
        HTTPException: If extraction fails
    """
    try:
        start_time = time.time()

        # Create or get cached extractor
        cache_key = f"{request.method}"
        if cache_key not in _extractors_cache:
            _extractors_cache[cache_key] = create_extractor(method=request.method)

        extractor = _extractors_cache[cache_key]

        # Extract aspects for all texts
        all_aspects = extractor.extract_batch(request.texts, batch_size=request.batch_size)  # type: ignore

        # Filter by confidence and sentiment
        results = []
        total_aspects = 0

        for aspects in all_aspects:
            # Filter by confidence
            if request.min_confidence > 0:
                aspects = [a for a in aspects if a.confidence >= request.min_confidence]

            # Remove sentiment if not requested
            if not request.include_sentiment:
                aspects = [
                    Aspect(
                        text=a.text,
                        category=a.category,
                        sentiment=None,
                        confidence=a.confidence,
                        start_pos=a.start_pos,
                        end_pos=a.end_pos,
                        context=a.context,
                    )
                    for a in aspects
                ]

            aspect_responses = [_aspect_to_response(a) for a in aspects]
            results.append(aspect_responses)
            total_aspects += len(aspect_responses)

        processing_time = (time.time() - start_time) * 1000

        return BatchExtractionResponse(
            results=results,
            total_texts=len(request.texts),
            total_aspects=total_aspects,
            method_used=request.method,
            processing_time_ms=processing_time,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch extraction failed: {str(e)}",
        )


@app.post("/evaluate", response_model=EvaluationResponse, status_code=status.HTTP_200_OK)
async def evaluate(request: EvaluationRequest) -> EvaluationResponse:
    """Evaluate aspect extraction results.

    Args:
        request: Evaluation request with predictions and ground truth

    Returns:
        Evaluation metrics

    Raises:
        HTTPException: If evaluation fails
    """
    try:
        # Convert string lists to Aspect objects
        predicted = [Aspect(text=text) for text in request.predicted_aspects]
        ground_truth = [Aspect(text=text) for text in request.ground_truth_aspects]

        # Evaluate
        result = evaluate_extraction(predicted, ground_truth, match_sentiment=request.match_sentiment)

        return EvaluationResponse(
            precision=result.precision,
            recall=result.recall,
            f1_score=result.f1,
            true_positives=result.true_positives,
            false_positives=result.false_positives,
            false_negatives=result.false_negatives,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@app.exception_handler(404)
async def not_found_handler(request, exc) -> JSONResponse:  # type: ignore
    """Handle 404 errors.

    Args:
        request: Request object
        exc: Exception

    Returns:
        JSON response with error message
    """
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. See /docs for available endpoints."},
    )


def run_server() -> None:
    """Run the API server.

    This is used by the aspect-api CLI command.
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run_server()
