.PHONY: help install install-dev test test-cov lint format type-check clean docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install        - Install package"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run all linters"
	@echo "  make format         - Format code with black and isort"
	@echo "  make type-check     - Run mypy type checking"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest

test-cov:
	pytest --cov=aspect_extraction --cov-report=term-missing --cov-report=html

lint:
	black --check aspect_extraction tests
	isort --check-only aspect_extraction tests
	flake8 aspect_extraction tests
	mypy aspect_extraction || true
	pylint aspect_extraction || true

format:
	black aspect_extraction tests examples
	isort aspect_extraction tests examples

type-check:
	mypy aspect_extraction

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build:
	docker build -t aspect-extraction .

docker-run:
	docker run -p 8000:8000 aspect-extraction

docker-compose-up:
	docker-compose up

docker-compose-down:
	docker-compose down
