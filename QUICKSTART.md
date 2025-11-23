# 🚀 DSPy Quick Start Guide

Get up and running with DSPy in 5 minutes!

## Prerequisites

- Python 3.10+
- OpenAI API key
- Docker (optional, for vector DB)

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/dspy-optimization-patterns
cd dspy-optimization-patterns

# Install with Poetry
make dev-install

# Or with pip
pip install -e .
```

## Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
# OPENAI_API_KEY=sk-your-key-here
```

## Option 1: Docker Compose (Easiest)

```bash
# Start everything (API + Qdrant + Phoenix)
docker-compose up -d

# Check logs
docker-compose logs -f api

# Visit API docs
open http://localhost:8000/docs
```

## Option 2: Local Development

### Step 1: Prepare Sample Data

```bash
# Copy sample data to processed folder
python scripts/prepare_data.py \
  --input data/raw/qa_dataset_sample.jsonl \
  --output data/processed/ \
  --task-type qa
```

### Step 2: Run Basic Example

```python
# examples/01_basic_usage.py
import dspy

# Configure
lm = dspy.LM(model="openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)

# Create predictor
from src.core.signatures import GenerateAnswer
qa = dspy.ChainOfThought(GenerateAnswer)

# Ask question
result = qa(
    context="Paris is the capital of France.",
    question="What is the capital of France?"
)

print(result.answer)  # "Paris"
```

### Step 3: Optimize (Compile) Your Program

```bash
# Quick optimization
./scripts/optimize_model.sh SimpleRAG data/processed/qa_dataset.jsonl qa

# Or detailed:
python -m src.pipeline.optimizer \
  --module SimpleRAG \
  --data data/processed/qa_dataset.jsonl \
  --task-type qa \
  --metric hybrid_qa \
  --optimizer mipro \
  --output artifacts/compiled_programs/rag_v1.json
```

This will:
- Use GPT-4 to generate optimized prompts
- Try different instruction phrasings
- Select best few-shot examples
- Save compiled program to `artifacts/`

### Step 4: Deploy API

```bash
# Start FastAPI server
make run-api

# Test with curl
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is DSPy?",
    "context": "DSPy is a framework for programming with LLMs."
  }'
```

## 🎯 What Just Happened?

1. **Defined Logic** → You created signatures and modules
2. **Compiled** → DSPy optimized prompts automatically
3. **Deployed** → Loaded compiled program in production API

No manual prompt engineering! 🎉

## Next Steps

### 1. Try RAG with Vector DB

```python
# examples/03_rag_with_vector_db.py
from src.integrations.vector_db import create_retriever
from src.core.modules import SimpleRAG

# Setup vector DB
retriever = create_retriever("chroma")
retriever.upsert(
    texts=["DSPy optimizes prompts...", "MIPRO is an optimizer..."],
    metadata=[{"source": "docs"}] * 2
)

# Use with RAG
rag = SimpleRAG()
result = rag.forward(
    question="What is MIPRO?",
    retriever_fn=lambda q: [r.text for r in retriever.search(q)]
)
```

### 2. Customize Metrics

```python
# src/core/metrics.py
def my_custom_metric(example, prediction):
    # Your evaluation logic
    return score >= 0.8
```

### 3. Add Your Own Module

```python
# src/core/modules.py
class MyCustomModule(BaseModule):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(MySignature)

    def forward(self, input_text):
        return self.predictor(text=input_text)
```

## 📊 Monitoring

### Phoenix Observability

```bash
# Start with docker-compose
docker-compose --profile observability up -d

# Or standalone
docker run -p 6006:6006 arizephoenix/phoenix

# View traces
open http://localhost:6006
```

### Cost Tracking

```python
from src.utils.tracing import CostTracker

tracker = CostTracker()
# ... make API calls ...
tracker.save_report("cost_report.json")
```

## 🐳 Production Deployment

### Kubernetes

```bash
# Configure secrets
kubectl create secret generic dspy-secrets \
  --from-literal=OPENAI_API_KEY=sk-xxx

# Deploy
kubectl apply -f k8s/

# Check status
kubectl get pods -n dspy-production
```

See [k8s/README.md](k8s/README.md) for full guide.

## 🎓 Learning Resources

- [Full README](README.md) - Comprehensive documentation
- [Examples](examples/) - Complete working examples
- [Tests](tests/) - See how components work
- [DSPy Docs](https://dspy-docs.vercel.app/) - Official documentation

## 💡 Pro Tips

1. **Start Small**: Use 20-50 examples for initial optimization
2. **Iterate**: Collect production failures → add to training data → re-optimize
3. **Measure**: Track metrics over time to detect degradation
4. **A/B Test**: Always test new compiled programs before full rollout
5. **Cache**: Enable caching during optimization to save money

## 🆘 Troubleshooting

### API Key Not Found

```bash
echo $OPENAI_API_KEY  # Should print your key
export OPENAI_API_KEY=sk-...
```

### Module Not Found

```bash
# Make sure you're in the project root
pwd  # Should end with dspy-optimization-patterns

# Install in editable mode
pip install -e .
```

### Optimization Too Slow

```bash
# Use cheaper optimizer
--optimizer bootstrap  # Instead of mipro

# Reduce dataset size
--train-size 20 --dev-size 50

# Enable caching
# Caching is automatic in config/optimizers.yaml
```

## 🎉 Success!

You now have a production-ready DSPy setup!

Next challenge: **Beat your baseline by 20%** through optimization. Good luck! 🚀
