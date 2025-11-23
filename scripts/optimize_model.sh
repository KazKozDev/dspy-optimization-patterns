#!/bin/bash
# Quick optimization script
# Usage: ./scripts/optimize_model.sh <module> <data_file> <task_type>

set -e

MODULE=${1:-SimpleRAG}
DATA_FILE=${2:-data/processed/qa_dataset.jsonl}
TASK_TYPE=${3:-qa}

echo "============================================"
echo "DSPy Optimization Script"
echo "============================================"
echo "Module: $MODULE"
echo "Data: $DATA_FILE"
echo "Task Type: $TASK_TYPE"
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Error: Data file not found: $DATA_FILE"
    echo ""
    echo "Creating sample data..."
    poetry run python scripts/prepare_data.py \
        --input data/raw/qa_dataset_sample.jsonl \
        --output data/processed/ \
        --task-type qa
    DATA_FILE="data/processed/qa_dataset.jsonl"
fi

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set"
    echo "   export OPENAI_API_KEY=sk-..."
    exit 1
fi

# Run optimization
echo "Starting optimization..."
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="artifacts/compiled_programs/${MODULE,,}_${TIMESTAMP}.json"

poetry run python -m src.pipeline.optimizer \
    --module "$MODULE" \
    --data "$DATA_FILE" \
    --task-type "$TASK_TYPE" \
    --metric hybrid_qa \
    --optimizer mipro \
    --output "$OUTPUT_FILE"

echo ""
echo "============================================"
echo "✓ Optimization Complete!"
echo "============================================"
echo "Compiled program saved to: $OUTPUT_FILE"
echo ""
echo "To deploy:"
echo "  1. Test locally: make run-api"
echo "  2. Deploy to K8s: kubectl apply -f k8s/"
echo ""
