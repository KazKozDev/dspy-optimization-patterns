<p align="center">
  <img width="240" height="" alt="logo" src="https://github.com/user-attachments/assets/a82845a4-511b-4203-b931-950463d64b6d" />
</p>
<h3 align="center">DSPy Production Framework</h3>

Production-grade implementation of DSPy for building, optimizing, and deploying LLM applications.

This framework demonstrates production best practices for DSPy. Instead of manually crafting prompts, DSPy compiles programs that automatically optimize prompts and select few-shot examples.

### Core Principles

- **Programming over Prompting**: Define logic, not prompts
- **Compilation**: Automatic optimization of prompts and examples
- **Teacher-Student Pattern**: Use expensive models for optimization, deploy with cheaper ones
- **Versioned Artifacts**: Compiled programs are versioned like ML models

![imag](https://github.com/user-attachments/assets/ff5a8818-aa7b-4e55-b89f-4802432c556d)

## Architecture

```
project-root/
├── config/               # Model and optimizer configurations
├── data/                 # Training, development, and test datasets
├── artifacts/            # Compiled programs (JSON artifacts)
├── src/
│   ├── core/            # DSPy signatures, modules, and metrics
│   ├── pipeline/        # Data loading and optimization
│   ├── utils/           # Tracing, logging, cost tracking
│   └── app/             # FastAPI server
└── tests/               # Unit and integration tests
```

## Quick Start

### Installation

```bash
# Using Poetry
poetry install --with dev

# Using Make
make dev-install
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key (optional)
- `TEACHER_MODEL`: Model for optimization (default: gpt-5)
- `STUDENT_MODEL`: Model for production (default: gpt-5-mini)

### Data Preparation

```bash
make prepare-sample-data
```

### Optimization

Compile optimal prompts and few-shot examples:

```bash
make optimize-rag
```

Or customize:

```bash
python -m src.pipeline.optimizer \
  --module SimpleRAG \
  --data data/processed/qa_dataset.jsonl \
  --metric rag_quality \
  --optimizer mipro \
  --output artifacts/compiled_programs/rag_v1.json
```

### API Deployment

```bash
make run-api
```

API documentation available at `http://localhost:8000/docs`

## Core Components

### Signatures

Signatures define the contract between your logic and the LLM.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions based on context."""

    context: str = dspy.InputField(desc="Relevant information")
    question: str = dspy.InputField(desc="User's question")
    answer: str = dspy.OutputField(desc="Concise answer")
```

Best practices:
- Clear docstrings (used in prompts)
- Descriptive field names
- Type hints for validation

### Modules

Modules contain business logic and operate in two modes:

1. **Zero-shot** (development): Default prompts
2. **Optimized** (production): Load compiled state from artifacts

```python
# Development mode
rag = SimpleRAG()

# Production mode
rag = SimpleRAG()
rag.load_compiled_state("artifacts/compiled_programs/rag_v1.json")
```

Module pattern:

```python
class MyModule(BaseModule):
    def __init__(self, compiled_state_path: Optional[str] = None):
        super().__init__(compiled_state_path)

        self.chain = dspy.ChainOfThought(MySignature)

        if compiled_state_path:
            self.load_compiled_state(compiled_state_path)
```

### Metrics

Metrics define success criteria. Without metrics, optimization is impossible.

```python
def rag_quality_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Evaluates retrieval quality, answer faithfulness, and correctness.
    Returns float in [0.0, 1.0].
    """
    pass
```

Metric types:
- **Heuristic**: Exact match, substring match (fast, brittle)
- **Semantic**: Embedding similarity (robust, moderate cost)
- **LLM-as-Judge**: Use GPT-5 to evaluate (flexible, expensive)
- **Hybrid**: Combine multiple metrics (recommended)

### Pipeline

Data loading:

```python
from src.pipeline.loader import load_and_split

trainset, devset, testset = load_and_split(
    "data/processed/qa_dataset.jsonl",
    task_type="rag",
    train_size=50,
    dev_size=100,
    test_size=200
)
```

Optimization:

```python
from src.pipeline.optimizer import OptimizationPipeline

pipeline = OptimizationPipeline()
compiled_module, dev_score, test_score = pipeline.run(
    module_class=SimpleRAG,
    data_path="data/processed/qa_dataset.jsonl",
    metric_name="rag_quality",
    optimizer_name="mipro",
    output_path="artifacts/compiled_programs/rag_v2.json"
)
```

Available optimizers:
- `bootstrap`: Generate few-shot examples
- `mipro`: Multi-prompt instruction optimization
- `signature_opt`: Refine field descriptions

## Optimization Process

### Compilation Steps

1. **Teacher Model** (GPT-5) generates high-quality examples
2. **Optimizer** tries different instruction phrasings, few-shot combinations, and signature descriptions
3. **Validation** on dev set selects best program
4. **Compiled State** saved as JSON with optimized instructions and examples

### Teacher-Student Pattern

```yaml
# config/models.yaml
teacher:
  model: "gpt-5"
  temperature: 0.0

student:
  model: "gpt-5-mini"
  temperature: 0.0
```

Cost example:
- Optimization (one-time): $5 using GPT-5
- Production (per request): $0.0001 using gpt-5-mini
- Break-even: 5000 requests

## API Usage

### Start Server

```bash
make run-api
# Or: uvicorn src.app.main:app --reload
```

### Question Answering

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DSPy?",
    "context": "DSPy is a framework for programming with LLMs."
  }'
```

### RAG

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does DSPy optimization work?",
    "top_k": 5
  }'
```

### Classification

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Research paper about transformers."
  }'
```

## Observability

### Phoenix Integration

```python
from src.utils.tracing import setup_phoenix_tracing

setup_phoenix_tracing(project_name="dspy_production")
```

Phoenix UI available at `http://localhost:6006`

### Custom Tracing

```python
from src.utils.tracing import DSPyTracer

tracer = DSPyTracer()
with tracer.trace_module("SimpleRAG", {"question": "..."}):
    result = rag_module(question="...")
```

### Cost Tracking

```python
from src.utils.tracing import CostTracker

tracker = CostTracker()
tracker.log_call(
    model="gpt-5-mini",
    input_tokens=500,
    output_tokens=200
)
tracker.save_report("cost_report.json")
```

## Testing

```bash
# All tests with coverage
make test

# Fast tests
make test-fast

# Watch mode
make test-watch
```

## Docker Deployment

### Build and Run

```bash
# Build image
make docker-build

# Run with docker-compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Services

- **API**: FastAPI server on port 8000
- **Qdrant**: Vector database on ports 6333, 6334
- **Phoenix**: Observability on port 6006 (optional)
- **Jupyter**: Development notebooks on port 8888 (optional)

### Profiles

```bash
# With observability
docker-compose --profile observability up -d

# With Jupyter
docker-compose --profile development up jupyter
```

### Single Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/artifacts:/app/artifacts \
  dspy-production:latest
```

## Kubernetes Deployment

Full manifests in `k8s/` directory.

```bash
# Deploy all
kubectl apply -f k8s/

# Check status
kubectl get all -n dspy-production

# Scale
kubectl scale deployment dspy-api --replicas=10 -n dspy-production
```

Features:
- Horizontal Pod Autoscaling
- Health checks and rolling updates
- Persistent storage for artifacts and vector DB
- Ingress with TLS
- Resource limits and requests

See [k8s/README.md](k8s/README.md) for details.

## Production Workflow

### 1. Data Collection

Collect edge cases where model fails:

```bash
# Production logs -> data/raw/failures.jsonl
# Human annotations -> data/processed/qa_dataset.jsonl
```

### 2. Offline Optimization

```bash
make optimize-rag
# Artifact saved: artifacts/compiled_programs/rag_v2_20240115.json
```

### 3. Evaluation

```bash
python -m src.pipeline.optimizer \
  --module SimpleRAG \
  --data data/processed/qa_dataset.jsonl
```

Test set is automatically held out for evaluation.

### 4. Deployment

```bash
# Rolling deployment
kubectl rollout restart deployment/dspy-api
```

### 5. Monitoring

- Track latency, cost, error rates
- Use Phoenix for request debugging
- A/B test compiled programs

## Configuration

### Models

`config/models.yaml`:

```yaml
teacher:
  provider: "openai"
  model: "gpt-5"
  temperature: 0.0

student:
  provider: "openai"
  model: "gpt-5-mini"
  temperature: 0.0

anthropic_teacher:
  provider: "anthropic"
  model: "claude-sonnet-4.5"
  temperature: 0.0
```

### Optimizers

`config/optimizers.yaml`:

```yaml
mipro:
  type: "MIPRO"
  num_candidates: 10
  max_bootstrapped_demos: 8
  metric_threshold: 0.80

run:
  train_size: 50
  dev_size: 100
  test_size: 200
  cache_dir: ".cache/dspy"
```

## Advanced Patterns

### Adaptive Routing

```python
class AdaptiveModule(BaseModule):
    """Route to different strategies based on input complexity."""

    def forward(self, question: str, context: str):
        routing = self.route(question=question)

        if "simple" in routing.complexity:
            return self.simple_qa(question=question, context=context)
        else:
            return self.complex_qa(question=question, context=context)
```

### Multi-Hop Reasoning

```python
class MultiHopReasoner(BaseModule):
    """Iterative reasoning for complex questions."""

    def forward(self, question: str, retriever_fn: callable):
        result = self.react(question=question, context=initial_context)
        return result
```

### Ensemble Optimization

```python
bootstrap_program = optimize_with_bootstrap(module, trainset, devset)
mipro_program = optimize_with_mipro(module, trainset, devset)

# Evaluate both, use best in production
```

## Vector Database Integration

### Qdrant

```python
from src.integrations.vector_db import create_retriever

retriever = create_retriever(
    provider="qdrant",
    host="localhost",
    port=6333,
    collection_name="documents"
)

retriever.upsert(
    texts=["Document 1", "Document 2"],
    metadata=[{"source": "web"}, {"source": "pdf"}]
)

results = retriever.search("What is DSPy?", top_k=5)
```

### Supported Providers

- **Qdrant**: Open-source, self-hosted, high performance
- **Pinecone**: Managed service, auto-scaling
- **Chroma**: Lightweight, development-friendly

### RAG Integration

```python
from src.core.modules import SimpleRAG

rag = SimpleRAG()
rag.load_compiled_state("artifacts/compiled_programs/rag_v1.json")

def retriever_fn(query: str):
    results = retriever.search(query, top_k=5)
    return [r.text for r in results]

result = rag.forward(
    question="How does DSPy work?",
    retriever_fn=retriever_fn
)
```

## CI/CD

GitHub Actions workflow in `.github/workflows/ci.yml`:

**On every push:**
- Linting (Ruff + Black)
- Type checking (MyPy)
- Unit tests with coverage
- Docker build test
- Security scan (Trivy)

**On main branch:**
- Build and push Docker image
- Deploy to staging
- Run integration tests

## Examples

```bash
# Basic usage
python examples/01_basic_usage.py

# Optimization workflow
python examples/02_optimization_workflow.py

# RAG with vector DB
python examples/03_rag_with_vector_db.py
```

## Sample Datasets

Located in `data/raw/`:

- `qa_dataset_sample.jsonl`: 20 Q&A pairs about DSPy
- `classification_dataset_sample.jsonl`: 20 tech/business/science articles
- `rag_dataset_sample.jsonl`: 10 complex RAG examples

All datasets ready for optimization.

## Resources

- [Quick Start Guide](QUICKSTART.md)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [Paper: DSPy - Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714)

---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE)
