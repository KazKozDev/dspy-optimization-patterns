.PHONY: install dev-install test lint format clean run-api optimize-rag help

# Default target
.DEFAULT_GOAL := help

# Environment variables
PYTHON := python3
POETRY := poetry
PORT := 8000

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

install: ## Install production dependencies
	@echo "📦 Installing production dependencies..."
	$(POETRY) install --without dev

dev-install: ## Install all dependencies (including dev tools)
	@echo "📦 Installing all dependencies..."
	$(POETRY) install --with dev --extras all
	@echo "✓ Installation complete!"

install-phoenix: ## Install Phoenix for observability
	@echo "📦 Installing Phoenix..."
	$(POETRY) install --extras phoenix

# ============================================================================
# CODE QUALITY
# ============================================================================

lint: ## Run code linters (ruff + mypy)
	@echo "🔍 Running linters..."
	$(POETRY) run ruff check src/ tests/
	$(POETRY) run mypy src/

format: ## Format code with black and ruff
	@echo "✨ Formatting code..."
	$(POETRY) run black src/ tests/
	$(POETRY) run ruff check --fix src/ tests/
	@echo "✓ Code formatted!"

check: lint ## Alias for lint
	@echo "✓ Code check complete!"

# ============================================================================
# TESTING
# ============================================================================

test: ## Run tests with coverage
	@echo "🧪 Running tests..."
	$(POETRY) run pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests without coverage
	@echo "🧪 Running fast tests..."
	$(POETRY) run pytest tests/ -v

test-watch: ## Run tests in watch mode
	@echo "🧪 Running tests in watch mode..."
	$(POETRY) run pytest-watch tests/ -v

# ============================================================================
# DSPY OPTIMIZATION (THE CORE!)
# ============================================================================

optimize-rag: ## Optimize RAG module with MIPRO
	@echo "🔥 Optimizing RAG module..."
	@echo "⚠️  This requires:"
	@echo "   1. Dataset at data/processed/qa_dataset.jsonl"
	@echo "   2. OPENAI_API_KEY in .env"
	@echo ""
	$(POETRY) run python -m src.pipeline.optimizer \
		--module SimpleRAG \
		--data data/processed/qa_dataset.jsonl \
		--task-type rag \
		--metric rag_quality \
		--optimizer mipro \
		--output artifacts/compiled_programs/rag_v1_mipro.json

optimize-classifier: ## Optimize document classifier
	@echo "🔥 Optimizing classifier..."
	$(POETRY) run python -m src.pipeline.optimizer \
		--module DocumentClassifier \
		--data data/processed/classification_dataset.jsonl \
		--task-type classification \
		--metric classification_reasoning \
		--optimizer bootstrap \
		--output artifacts/compiled_programs/classifier_v1_bootstrap.json

optimize-custom: ## Optimize with custom parameters (set ARGS="...")
	@echo "🔥 Running custom optimization..."
	$(POETRY) run python -m src.pipeline.optimizer $(ARGS)

# ============================================================================
# API SERVER
# ============================================================================

run-api: ## Start FastAPI server
	@echo "🚀 Starting API server on port $(PORT)..."
	@echo "📖 Docs will be available at http://localhost:$(PORT)/docs"
	$(POETRY) run uvicorn src.app.main:app --host 0.0.0.0 --port $(PORT) --reload

run-api-prod: ## Start API in production mode (no reload)
	@echo "🚀 Starting API server in production mode..."
	$(POETRY) run uvicorn src.app.main:app --host 0.0.0.0 --port $(PORT) --workers 4

# ============================================================================
# OBSERVABILITY
# ============================================================================

run-phoenix: ## Start Phoenix observability server
	@echo "🔍 Starting Phoenix..."
	$(POETRY) run python -c "import phoenix as px; px.launch_app()"

# ============================================================================
# DATA MANAGEMENT
# ============================================================================

prepare-sample-data: ## Create sample datasets for testing
	@echo "📊 Creating sample datasets..."
	@mkdir -p data/raw data/processed
	@echo '{"question": "What is DSPy?", "answer": "DSPy is a framework for programming with LLMs."}' > data/raw/sample.jsonl
	@echo '{"question": "How does optimization work?", "answer": "DSPy uses teacher-student optimization."}' >> data/raw/sample.jsonl
	@cp data/raw/sample.jsonl data/processed/qa_dataset.jsonl
	@echo "✓ Sample data created at data/processed/"

# ============================================================================
# DOCKER
# ============================================================================

docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t dspy-production:latest .

docker-run: ## Run Docker container
	@echo "🐳 Running Docker container..."
	docker run -p $(PORT):$(PORT) \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-v $(PWD)/artifacts:/app/artifacts \
		dspy-production:latest

# ============================================================================
# CLEANUP
# ============================================================================

clean: ## Clean temporary files and caches
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ htmlcov/ .coverage
	@echo "✓ Cleanup complete!"

clean-artifacts: ## Remove all compiled programs (use with caution!)
	@echo "⚠️  This will delete all compiled DSPy programs!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf artifacts/compiled_programs/*; \
		echo "✓ Artifacts deleted"; \
	else \
		echo "Cancelled"; \
	fi

# ============================================================================
# UTILITIES
# ============================================================================

inspect-artifact: ## Inspect compiled program (usage: make inspect-artifact ARTIFACT=path/to/file.json)
	@echo "🔍 Inspecting artifact: $(ARTIFACT)"
	$(POETRY) run python -c "from src.utils.tracing import inspect_compiled_program; inspect_compiled_program('$(ARTIFACT)')"

shell: ## Start IPython shell with imports
	@echo "🐍 Starting IPython shell..."
	$(POETRY) run ipython -i -c "import dspy; from src.core import *; from src.pipeline import *"

notebook: ## Start Jupyter notebook
	@echo "📓 Starting Jupyter..."
	$(POETRY) run jupyter notebook

# ============================================================================
# HELP
# ============================================================================

help: ## Show this help message
	@echo "DSPy Production - Available Commands"
	@echo "====================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make dev-install          # Install everything"
	@echo "  make prepare-sample-data  # Create test data"
	@echo "  make optimize-rag         # Compile RAG module"
	@echo "  make run-api              # Start API server"
	@echo ""
