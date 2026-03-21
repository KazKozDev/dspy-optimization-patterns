# DSPy Production Framework

Production-oriented DSPy starter for compiling, serving, and operating LLM programs with a teacher-student workflow.

## Highlights

- Compiles prompts into versioned artifacts
- Separates optimization from inference cost
- Exposes FastAPI inference endpoints
- Includes Docker and Kubernetes deployment
- Provides tracing, tests, and examples

## Demo

Add demo GIF or screenshot.

## Overview

This repository shows how to treat DSPy programs like deployable assets rather than ad hoc prompts. It provides reusable DSPy modules, an optimization pipeline that saves compiled JSON artifacts, and a FastAPI service that loads those artifacts at startup for production inference. The project is aimed at teams building RAG, QA, or classification systems that need a cleaner path from offline optimization to online serving.

## Motivation

Most LLM applications stop at prompt iteration, which makes behavior hard to reproduce, version, and deploy. DSPy changes that model by compiling instructions and few-shot examples against a metric. This project packages that idea into a practical workflow: prepare data, optimize with a stronger teacher model, save the compiled artifact, and serve requests with a cheaper student model. The result is a more disciplined way to ship LLM systems with measurable optimization and lower inference cost.

## Features

- Provides DSPy modules for QA, RAG, classification, and multi-hop reasoning.
- Saves compiled programs to `artifacts/compiled_programs/` for reuse in production.
- Runs optimization through configurable teacher and student model settings in `config/`.
- Exposes `/qa`, `/rag`, `/classify`, `/health`, and `/artifacts` endpoints via FastAPI.
- Includes sample data preparation, example scripts, and pytest coverage for core behavior.
- Supports local runs, Docker Compose, Phoenix observability, and Kubernetes manifests.

## Architecture

Core components:

- `src/core/` defines signatures, metrics, and DSPy modules.
- `src/pipeline/` loads datasets, splits train/dev/test sets, and runs compilation.
- `src/app/` serves optimized or fallback modules through REST endpoints.
- `src/integrations/` contains vector database integration points.
- `artifacts/compiled_programs/` stores versioned compiled DSPy states.

Flow:

`dataset -> metric -> optimizer -> compiled artifact -> FastAPI service -> inference request`

## Tech Stack

- Python 3.10+
- DSPy
- FastAPI + Uvicorn
- Poetry
- Pytest, Ruff, Black, MyPy
- Docker Compose
- Kubernetes
- Arize Phoenix

## Quick Start

1. Clone the repo and install dependencies.

```bash
git clone https://github.com/KazKozDev/dspy-optimization-patterns.git
cd dspy-optimization-patterns
poetry install --with dev
```

2. Create local environment variables.

```bash
cp .env.example .env
```

Required for most flows: `OPENAI_API_KEY`. Optional model configuration lives in `config/models.yaml`.

3. Generate sample data.

```bash
make prepare-sample-data
```

4. Compile a RAG module into a reusable artifact.

```bash
make optimize-rag
```

5. Start the API.

```bash
make run-api
```

Open `http://localhost:8000/docs` for the interactive API docs.

## Usage

Run the service locally and send a QA request:

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DSPy?",
    "context": "DSPy is a framework for programming with LLMs."
  }'
```

Compile a classifier artifact:

```bash
make optimize-classifier
```

Explore example workflows:

```bash
python examples/01_basic_usage.py
python examples/02_optimization_workflow.py
python examples/03_rag_with_vector_db.py
```

## Project Structure

```text
config/      model and optimizer settings
data/        raw and processed datasets
artifacts/   compiled DSPy programs
src/core/    signatures, modules, metrics
src/pipeline/ data loading and optimization
src/app/     FastAPI serving layer
tests/       API and module tests
```

## Testing

```bash
make test
make lint
```

## More Docs

- [Quick Start](QUICKSTART.md)
- [Kubernetes Deployment](k8s/README.md)
- [Frontend Prototype](frontend/README-PRO.md)

---

MIT - see LICENSE

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[LinkedIn](https://www.linkedin.com/in/kazkozdev/)
[Email](mailto:kazkozdev@gmail.com)
