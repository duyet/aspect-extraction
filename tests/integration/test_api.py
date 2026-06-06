"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from aspect_extraction.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self) -> None:
        """Should return health status from root."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_endpoint(self) -> None:
        """Should return health status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestExtractionEndpoint:
    """Test /extract endpoint."""

    def test_extract_basic(self) -> None:
        """Should extract aspects from text."""
        payload = {
            "text": "The camera quality is excellent but battery life is poor",
            "method": "rule",
        }

        response = client.post("/extract", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "aspects" in data
        assert "method_used" in data
        assert "processing_time_ms" in data
        assert isinstance(data["aspects"], list)

    def test_extract_with_confidence_filter(self) -> None:
        """Should filter aspects by confidence."""
        payload = {
            "text": "The camera is great",
            "method": "rule",
            "min_confidence": 0.9,
        }

        response = client.post("/extract", json=payload)
        assert response.status_code == 200

        data = response.json()
        # All returned aspects should meet confidence threshold
        for aspect in data["aspects"]:
            assert aspect["confidence"] >= 0.9

    def test_extract_empty_text_fails(self) -> None:
        """Should fail for empty text."""
        payload = {
            "text": "",
            "method": "rule",
        }

        response = client.post("/extract", json=payload)
        assert response.status_code == 422  # Validation error

    def test_extract_invalid_method_fails(self) -> None:
        """Should fail for invalid method."""
        payload = {
            "text": "Some text",
            "method": "invalid_method",
        }

        response = client.post("/extract", json=payload)
        assert response.status_code == 422  # Validation error


class TestBatchExtractionEndpoint:
    """Test /extract/batch endpoint."""

    def test_extract_batch_basic(self) -> None:
        """Should extract aspects from multiple texts."""
        payload = {
            "texts": [
                "The camera is great",
                "Battery life is poor",
            ],
            "method": "rule",
        }

        response = client.post("/extract/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "total_texts" in data
        assert "total_aspects" in data
        assert len(data["results"]) == 2
        assert data["total_texts"] == 2

    def test_extract_batch_empty_list_fails(self) -> None:
        """Should fail for empty text list."""
        payload = {
            "texts": [],
            "method": "rule",
        }

        response = client.post("/extract/batch", json=payload)
        assert response.status_code == 422  # Validation error


class TestEvaluationEndpoint:
    """Test /evaluate endpoint."""

    def test_evaluate_basic(self) -> None:
        """Should evaluate predictions."""
        payload = {
            "predicted_aspects": ["battery", "camera"],
            "ground_truth_aspects": ["battery", "screen"],
        }

        response = client.post("/evaluate", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "precision" in data
        assert "recall" in data
        assert "f1_score" in data
        assert "true_positives" in data
        assert "false_positives" in data
        assert "false_negatives" in data

        # Verify metrics are in valid range
        assert 0 <= data["precision"] <= 1
        assert 0 <= data["recall"] <= 1
        assert 0 <= data["f1_score"] <= 1


class TestNotFoundHandler:
    """Test 404 handler."""

    def test_not_found_returns_helpful_message(self) -> None:
        """Should return helpful message for 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "docs" in data["detail"].lower()
