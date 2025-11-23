# 🚀 DSPy Production-Ready Framework

A production-grade implementation of DSPy for building, optimizing, and deploying LLM-based applications.

## 📖 Overview

This repository demonstrates best practices for using DSPy in production environments. DSPy fundamentally changes how we build with LLMs: instead of manually crafting prompts, we **compile** programs that automatically optimize prompts and select few-shot examples.

### Key Concepts

- **Programming vs Prompting**: Define logic, not prompts
- **Compilation**: Automatic optimization of prompts and examples
- **Teacher-Student**: Use expensive models for optimization, deploy with cheap ones
- **Artifacts as Truth**: Compiled programs are versioned like ML models

## 🏗️ Architecture

```
project-root/
├── config/               # YAML configs for models & optimizers
├── data/                 # Training, dev, and test datasets
├── artifacts/            # Compiled programs (JSON) - the "models"
├── src/
│   ├── core/            # DSPy signatures, modules, and metrics
│   ├── pipeline/        # Data loading and optimization logic
│   ├── utils/           # Tracing, logging, cost tracking
│   └── app/             # FastAPI serving layer
└── tests/               # Unit and integration tests
```

## 🚦 Quick Start

### 1. Installation

```bash
# Using Poetry (recommended)
make dev-install

# Or manually
poetry install --with dev --extras all
```

### 2. Setup Environment

```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=sk-...
```

### 3. Prepare Sample Data

```bash
make prepare-sample-data
```

### 4. Run Optimization (Compile Your Program)

```bash
# This is the CORE of DSPy - compile optimal prompts
make optimize-rag

# Or customize:
python -m src.pipeline.optimizer \
  --module SimpleRAG \
  --data data/processed/qa_dataset.jsonl \
  --metric rag_quality \
  --optimizer mipro \
  --output artifacts/compiled_programs/rag_v1.json
```

### 5. Deploy API

```bash
make run-api
# API docs: http://localhost:8000/docs
```

## 📚 Core Components

### Signatures (`src/core/signatures.py`)

Signatures define the **contract** between your logic and the LLM.

```python
class GenerateAnswer(dspy.Signature):
    """Answer questions based on context."""

    context: str = dspy.InputField(desc="Relevant information")
    question: str = dspy.InputField(desc="User's question")
    answer: str = dspy.OutputField(desc="Concise answer")
```

**Best Practices:**
- Clear docstrings (used in prompts!)
- Descriptive field names
- Type hints for validation

### Modules (`src/core/modules.py`)

Modules contain your **business logic**. They work in two modes:

1. **Zero-shot** (development): Default prompts
2. **Optimized** (production): Load compiled state from artifacts

```python
# Development
rag = SimpleRAG()

# Production
rag = SimpleRAG()
rag.load_compiled_state("artifacts/compiled_programs/rag_v1.json")
```

**Key Pattern:**

```python
class MyModule(BaseModule):
    def __init__(self, compiled_state_path: Optional[str] = None):
        super().__init__(compiled_state_path)

        # Define sub-modules
        self.chain = dspy.ChainOfThought(MySignature)

        # Auto-load if path provided
        if compiled_state_path:
            self.load_compiled_state(compiled_state_path)
```

### Metrics (`src/core/metrics.py`)

Metrics define **what "good" means**. Without metrics, optimization is impossible.

```python
def rag_quality_metric(example: dspy.Example, prediction: dspy.Prediction) -> float:
    """
    Evaluates:
    1. Retrieval quality (are relevant docs retrieved?)
    2. Answer faithfulness (is answer grounded in context?)
    3. Answer correctness (does it match expected answer?)
    """
    # Implementation...
```

**Metric Types:**
- **Heuristic**: Exact match, substring match (fast, brittle)
- **Semantic**: Embedding similarity (robust, moderate cost)
- **LLM-as-Judge**: Use GPT-5o to evaluate (flexible, expensive, best quality)
- **Hybrid**: Combine multiple metrics (recommended)

### Pipeline (`src/pipeline/`)

#### Data Loading

```python
from src.pipeline.loader import load_and_split

trainset, devset, testset = load_and_split(
    "data/processed/qa_dataset.jsonl",
    task_type="rag",
    train_size=50,   # For generating few-shot examples
    dev_size=100,    # For validation during optimization
    test_size=200,   # Hold-out for final evaluation
)
```

#### Optimization

```python
from src.pipeline.optimizer import OptimizationPipeline

pipeline = OptimizationPipeline()
compiled_module, dev_score, test_score = pipeline.run(
    module_class=SimpleRAG,
    data_path="data/processed/qa_dataset.jsonl",
    metric_name="rag_quality",
    optimizer_name="mipro",
    output_path="artifacts/compiled_programs/rag_v2.json",
)
```

**Available Optimizers:**
- `bootstrap`: Generate few-shot examples from training data
- `mipro`: Multi-prompt instruction optimization (best for complex tasks)
- `signature_opt`: Refine field descriptions

## 🔬 Optimization Deep Dive

### What Happens During Compilation?

1. **Teacher Model** (GPT-5o) generates high-quality examples on training set
2. **Optimizer** tries different:
   - Instruction phrasings
   - Few-shot example combinations
   - Signature descriptions
3. **Validation** on dev set selects best program
4. **Compiled State** saved as JSON with:
   - Optimized instructions
   - Best few-shot examples
   - Signature metadata

### Teacher-Student Pattern

```yaml
# config/models.yaml
teacher:
  model: "gpt-5o"        # Next-gen model - for optimization

student:
  model: "gpt-5o-mini"   # Cost-effective - for production
```

**Cost Savings Example:**
- Optimization (one-time): $5 using GPT-5o teacher
- Production (per request): $0.0001 using gpt-5o-mini student
- **ROI**: After 5000 requests, you've broken even

## 🌐 API Usage

### Start Server

```bash
make run-api
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

### RAG Endpoint

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
    "text": "This is a research paper about transformers."
  }'
```

## 📊 Observability

### Phoenix Integration

```python
from src.utils.tracing import setup_phoenix_tracing

# Start Phoenix (http://localhost:6006)
setup_phoenix_tracing(project_name="dspy_production")

# Now all DSPy calls are automatically traced!
```

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
    model="gpt-5o-mini",
    input_tokens=500,
    output_tokens=200
)
tracker.save_report("cost_report.json")
```

## 🧪 Testing

```bash
# Run all tests with coverage
make test

# Fast tests (no coverage)
make test-fast

# Watch mode
make test-watch
```

## 🐳 Docker Deployment

### Build Image

```bash
make docker-build
```

### Run Container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -v $(pwd)/artifacts:/app/artifacts \
  dspy-production:latest
```

### Kubernetes Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dspy-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: dspy-production:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        volumeMounts:
        - name: artifacts
          mountPath: /app/artifacts
          readOnly: true
      volumes:
      - name: artifacts
        persistentVolumeClaim:
          claimName: dspy-artifacts-pvc
```

## 📈 Production Workflow

### 1. Data Collection

Continuously collect edge cases where your model fails:

```bash
# Production logs → data/raw/failures.jsonl
# Human annotations → data/processed/qa_dataset.jsonl
```

### 2. Offline Optimization

```bash
# Run weekly or when you have enough new data
make optimize-rag

# Artifact saved: artifacts/compiled_programs/rag_v2_20240115.json
```

### 3. Evaluation

```bash
# Always evaluate on hold-out test set
python -m src.pipeline.optimizer \
  --module SimpleRAG \
  --data data/processed/qa_dataset.jsonl \
  # ... (test set is automatically held out)
```

### 4. Deployment

```bash
# Update production to use new artifact
# artifacts/compiled_programs/rag_v2_20240115.json

# Rolling deployment (zero downtime)
kubectl rollout restart deployment/dspy-api
```

### 5. Monitoring

- Track latency, cost, error rates
- Use Phoenix to debug individual requests
- A/B test new compiled programs

## 🔧 Configuration

### Model Configuration (`config/models.yaml`)

```yaml
teacher:
  provider: "openai"
  model: "gpt-5o"  # Next-generation model
  temperature: 0.0

student:
  provider: "openai"
  model: "gpt-5o-mini"  # Cost-effective
  temperature: 0.0

# Alternative configurations
gpt5o_teacher:
  provider: "openai"
  model: "gpt-5o-2025-01-20"

# Or use Anthropic
anthropic_teacher:
  provider: "anthropic"
  model: "claude-sonnet-5"
```

### Optimizer Configuration (`config/optimizers.yaml`)

```yaml
mipro:
  type: "MIPRO"
  num_candidates: 10        # Instructions to try
  max_bootstrapped_demos: 8 # Few-shot examples
  metric_threshold: 0.80    # Minimum acceptable score

run:
  train_size: 50
  dev_size: 100
  test_size: 200
  cache_dir: ".cache/dspy"  # Cache LLM calls to save money
```

## 💡 Advanced Patterns

### Adaptive Routing

```python
class AdaptiveModule(BaseModule):
    """Route to different strategies based on input complexity."""

    def forward(self, question: str, context: str):
        # Determine complexity
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
        # ReAct pattern: Thought → Action → Observation
        result = self.react(question=question, context=initial_context)
        return result
```

### Ensemble of Optimizers

```python
# Optimize same module with different strategies
bootstrap_program = optimize_with_bootstrap(module, trainset, devset)
mipro_program = optimize_with_mipro(module, trainset, devset)

# Evaluate both on test set
# Use best one in production
```

## 🗄️ Vector Database Integration

Support for multiple vector DB providers for production RAG:

### Qdrant (Recommended for Production)

```python
from src.integrations.vector_db import create_retriever

# Create retriever
retriever = create_retriever(
    provider="qdrant",
    host="localhost",
    port=6333,
    collection_name="documents"
)

# Ingest documents
retriever.upsert(
    texts=["Document 1...", "Document 2..."],
    metadata=[{"source": "web"}, {"source": "pdf"}]
)

# Search
results = retriever.search("What is DSPy?", top_k=5)
```

### Supported Providers

- **Qdrant** - Open-source, self-hosted, high performance
- **Pinecone** - Managed service, auto-scaling
- **Chroma** - Lightweight, perfect for development

### Using with RAG

```python
from src.core.modules import SimpleRAG

rag = SimpleRAG()
rag.load_compiled_state("artifacts/compiled_programs/rag_v1.json")

# Create retriever function
def retriever_fn(query: str):
    results = retriever.search(query, top_k=5)
    return [r.text for r in results]

# Run RAG
result = rag.forward(
    question="How does DSPy work?",
    retriever_fn=retriever_fn
)
```

## 🐳 Local Development with Docker Compose

Complete development environment with one command:

```bash
# Start everything: API + Qdrant + Phoenix
docker-compose up -d

# Start with observability
docker-compose --profile observability up -d

# Start Jupyter for experimentation
docker-compose --profile development up jupyter

# View logs
docker-compose logs -f api

# Stop everything
docker-compose down
```

Services included:
- **API**: FastAPI server on :8000
- **Qdrant**: Vector DB on :6333
- **Phoenix**: Observability on :6006
- **Jupyter**: Notebooks on :8888

## ☸️ Production Kubernetes Deployment

Full K8s manifests included in `k8s/`:

```bash
# Quick deploy
kubectl apply -f k8s/

# Or step-by-step
kubectl apply -f k8s/deployment.yaml  # API deployment
kubectl apply -f k8s/qdrant.yaml      # Vector DB

# Check status
kubectl get all -n dspy-production

# Scale
kubectl scale deployment dspy-api --replicas=10 -n dspy-production
```

Features:
- Horizontal autoscaling (HPA)
- Health checks and rolling updates
- Persistent storage for artifacts and vector DB
- Ingress with TLS support
- Resource limits and requests
- Multi-replica deployment

See [k8s/README.md](k8s/README.md) for complete guide.

## 🔄 CI/CD Pipeline

GitHub Actions workflow included (`.github/workflows/ci.yml`):

**On every push:**
- Linting (Ruff + Black)
- Type checking (MyPy)
- Unit tests with coverage
- Docker build test
- Security scan (Trivy)

**On main branch:**
- Build and push Docker image
- Deploy to staging environment
- Run integration tests

Customize for your needs!

## 📝 Example Scripts

Quick start examples in `examples/`:

```bash
# Basic usage
python examples/01_basic_usage.py

# Full optimization workflow
python examples/02_optimization_workflow.py

# RAG with vector DB
python examples/03_rag_with_vector_db.py
```

Helper scripts in `scripts/`:

```bash
# Prepare data
python scripts/prepare_data.py \
  --input data/raw/qa_dataset_sample.jsonl \
  --task-type qa

# Quick optimization
./scripts/optimize_model.sh SimpleRAG data/processed/qa_dataset.jsonl qa
```

## 📊 Sample Datasets

Production-quality datasets included in `data/raw/`:

- **qa_dataset_sample.jsonl** - 20 Q&A pairs about DSPy (with context)
- **classification_dataset_sample.jsonl** - 20 tech/business/science articles
- **rag_dataset_sample.jsonl** - 10 complex RAG examples with contexts

All ready to use for optimization!

## 📚 Resources

- [Quick Start Guide](QUICKSTART.md) - Get running in 5 minutes
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
- [Paper: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"](https://arxiv.org/abs/2310.03714)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Make changes and test (`make test`)
4. Format code (`make format`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push and open PR

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- DSPy team at Stanford for the incredible framework
- Arize for Phoenix observability
- The open-source community

---

**Built with ❤️ using DSPy**

For questions or issues, please open a GitHub issue or reach out via email.
