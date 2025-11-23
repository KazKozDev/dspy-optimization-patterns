"""
Example 1: Basic DSPy Usage

This example demonstrates:
1. Creating a simple signature
2. Using ChainOfThought predictor
3. Basic evaluation with metrics
"""

import dspy
from src.core.signatures import GenerateAnswer
from src.core.metrics import exact_match

# Configure LLM
lm = dspy.LM(model="openai/gpt-5o-mini", temperature=0.0)
dspy.settings.configure(lm=lm)

# Create predictor
qa = dspy.ChainOfThought(GenerateAnswer)

# Test prediction
result = qa(
    context="Paris is the capital and largest city of France.",
    question="What is the capital of France?",
)

print("Question:", result.question)
print("Answer:", result.answer)
print("Reasoning:", result.reasoning)

# Evaluate with metric
example = dspy.Example(
    context="Paris is the capital of France.",
    question="What is the capital of France?",
    answer="Paris",
).with_inputs("context", "question")

prediction = qa(context=example.context, question=example.question)
score = exact_match(example, prediction)
print(f"\nMetric Score: {score}")
