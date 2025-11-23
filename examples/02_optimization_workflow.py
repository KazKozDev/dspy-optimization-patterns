"""
Example 2: Complete Optimization Workflow

This example shows the full optimization process:
1. Load dataset
2. Split into train/dev/test
3. Run optimization
4. Evaluate results
5. Save compiled program
"""

import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from src.core.modules import SimpleRAG
from src.core.metrics import hybrid_qa_metric
from src.pipeline.loader import QADatasetLoader, split_dataset

# 1. Configure models
teacher_lm = dspy.LM(model="openai/gpt-5o", temperature=0.0)  # Best for optimization
student_lm = dspy.LM(model="openai/gpt-5o-mini", temperature=0.0)  # Cheap for production

# 2. Load dataset
print("Loading dataset...")
examples = QADatasetLoader.load("data/raw/qa_dataset_sample.jsonl", has_context=True)

# 3. Split data
trainset, devset, testset = split_dataset(
    examples,
    train_ratio=0.3,
    dev_ratio=0.4,
    test_ratio=0.3,
    random_seed=42,
)

# 4. Create unoptimized module
print("\nCreating module...")
dspy.settings.configure(lm=teacher_lm)

def mock_retriever(query: str):
    """Mock retriever for demo"""
    return ["Mock context 1", "Mock context 2"]

# We'll use a simpler module for this example
from src.core.signatures import GenerateAnswer
qa_module = dspy.ChainOfThought(GenerateAnswer)

# 5. Run optimization
print("\nStarting optimization...")
optimizer = BootstrapFewShotWithRandomSearch(
    metric=hybrid_qa_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=2,
    num_candidate_programs=5,
    teacher_settings=dict(lm=teacher_lm),
)

compiled_module = optimizer.compile(
    qa_module,
    trainset=trainset[:10],  # Use small subset for demo
    valset=devset[:20],
)

# 6. Evaluate
print("\nEvaluating on test set...")
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=testset[:10],
    metric=hybrid_qa_metric,
    num_threads=1,
)

score = evaluator(compiled_module)
print(f"\nTest Score: {score:.2%}")

# 7. Save
output_path = "artifacts/compiled_programs/qa_demo.json"
compiled_module.save(output_path)
print(f"\nSaved to: {output_path}")
