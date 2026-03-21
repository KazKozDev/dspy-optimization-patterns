# Quick Start

Use this guide when you want the fastest path from clone to a working local DSPy workflow. For project context, architecture, and deployment overview, see [README.md](README.md).

## What You Will Do

- Install dependencies
- Add your API key
- Create sample data
- Compile one DSPy artifact
- Start the API and send a request

## Prerequisites

- Python 3.10+
- Poetry
- `OPENAI_API_KEY`
- Docker and Docker Compose only if you want the containerized path

## Local Path

1. Install dependencies.

```bash
git clone https://github.com/KazKozDev/dspy-optimization-patterns.git
cd dspy-optimization-patterns
poetry install --with dev
```

2. Create environment variables.

```bash
cp .env.example .env
export OPENAI_API_KEY=sk-...
```

3. Create a small sample dataset.

```bash
make prepare-sample-data
```

4. Compile a RAG artifact.

```bash
make optimize-rag
```

This writes a compiled DSPy program into `artifacts/compiled_programs/`.

5. Start the API.

```bash
make run-api
```

6. Verify the service.

```bash
curl http://localhost:8000/health
```

7. Send a QA request.

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DSPy?",
    "context": "DSPy is a framework for programming with LLMs."
  }'
```

Open `http://localhost:8000/docs` for the interactive API UI.

## Docker Path

Use this path if you want the API and supporting services in containers.

```bash
docker-compose up -d
docker-compose ps
docker-compose logs -f api
```

The default API endpoint is `http://localhost:8000`, and the docs are at `http://localhost:8000/docs`.

## Common Next Steps

Run the example scripts:

```bash
python examples/01_basic_usage.py
python examples/02_optimization_workflow.py
python examples/03_rag_with_vector_db.py
```

Compile a classifier instead of RAG:

```bash
make optimize-classifier
```

Run tests and linting:

```bash
make test
make lint
```

## Troubleshooting

If optimization fails immediately, check that `OPENAI_API_KEY` is exported in your shell.

If the API starts but answers are poor, confirm that a compiled artifact exists in `artifacts/compiled_programs/`; otherwise the service falls back to unoptimized modules.

If you need deployment details, observability setup, or a broader explanation of the repository, continue with [README.md](README.md) and [k8s/README.md](k8s/README.md).
