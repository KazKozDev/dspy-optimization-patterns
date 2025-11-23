"""
DSPy Metrics - The Heart of Optimization

Without good metrics, DSPy cannot optimize effectively.
A metric is a function that takes (example, prediction) and returns:
- bool: True if prediction is acceptable
- float: Score between 0.0 and 1.0

Best Practices:
1. Start with simple heuristics (exact match, substring)
2. Add semantic similarity for robustness
3. Use LLM-as-Judge for complex reasoning tasks
4. Combine multiple metrics (hybrid approach)
5. Log metric failures to improve training data

Critical: The metric defines what "good" means for your task.
"""

from typing import Optional, List, Dict, Any
import re

import dspy
from dspy.evaluate import SemanticF1

from .signatures import EvaluateAnswer


# ============================================================================
# BASIC HEURISTIC METRICS
# ============================================================================


def exact_match(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    """
    Strictest metric: Prediction must exactly match expected answer.

    Use when: Factual QA with deterministic answers (dates, numbers, names)
    """
    expected = example.answer.strip().lower()
    predicted = prediction.answer.strip().lower()
    return expected == predicted


def substring_match(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    """
    Check if expected answer appears in prediction.

    Use when: Answer might contain additional context but core fact is there
    """
    expected = example.answer.strip().lower()
    predicted = prediction.answer.strip().lower()
    return expected in predicted or predicted in expected


def length_constrained_match(
    example: dspy.Example,
    prediction: dspy.Prediction,
    max_length: int = 100,
) -> bool:
    """
    Combine substring match with length constraint.

    Use when: Answers should be concise (e.g., chatbot responses)
    """
    matches = substring_match(example, prediction)
    length_ok = len(prediction.answer.split()) <= max_length
    return matches and length_ok


# ============================================================================
# SEMANTIC SIMILARITY METRICS
# ============================================================================


def semantic_similarity_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    threshold: float = 0.7,
) -> float:
    """
    Use embedding-based similarity to compare answers.

    More robust than exact match - handles paraphrases.

    Args:
        threshold: Minimum similarity score to consider correct

    Returns:
        Similarity score (0.0 to 1.0)
    """
    # DSPy has built-in SemanticF1 metric
    f1_score = SemanticF1()(example, prediction)
    return float(f1_score >= threshold)


# ============================================================================
# LLM-AS-JUDGE METRICS (Most Powerful)
# ============================================================================


class LLMJudgeMetric:
    """
    Use a strong LLM to evaluate answers.

    This is the most flexible metric but also most expensive.
    Use for tasks where:
    - Answers are creative/open-ended
    - Multiple valid answers exist
    - Reasoning quality matters more than exact wording

    Critical: The judge model should be STRONGER than the task model.
    Never use GPT-4o-mini to judge GPT-4o.
    """

    def __init__(self, judge_model: Optional[dspy.LM] = None):
        """
        Args:
            judge_model: Strong model for evaluation (e.g., GPT-4o)
        """
        self.judge_model = judge_model
        self.evaluator = dspy.ChainOfThought(EvaluateAnswer)

    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        threshold: float = 0.7,
    ) -> bool:
        """
        Evaluate answer quality using LLM judge.

        Returns:
            True if score >= threshold
        """
        # Temporarily switch to judge model
        if self.judge_model:
            with dspy.context(lm=self.judge_model):
                result = self.evaluator(
                    question=example.question,
                    expected_answer=example.answer,
                    generated_answer=prediction.answer,
                )
        else:
            result = self.evaluator(
                question=example.question,
                expected_answer=example.answer,
                generated_answer=prediction.answer,
            )

        score = float(result.score)
        return score >= threshold


# ============================================================================
# RETRIEVAL-SPECIFIC METRICS
# ============================================================================


def retrieval_relevance(
    example: dspy.Example,
    prediction: dspy.Prediction,
    min_relevant_docs: int = 1,
) -> bool:
    """
    Check if retrieved contexts contain answer.

    Use for: RAG systems where retrieval quality matters
    """
    if not hasattr(prediction, "contexts"):
        return False

    answer = example.answer.lower()
    relevant_count = sum(1 for ctx in prediction.contexts if answer in ctx.lower())

    return relevant_count >= min_relevant_docs


def answer_faithfulness(
    example: dspy.Example,
    prediction: dspy.Prediction,
) -> bool:
    """
    Check if answer is grounded in retrieved context.

    Prevents hallucination - answer must be derivable from context.

    Use for: RAG systems where factuality is critical
    """
    if not hasattr(prediction, "contexts") or not hasattr(prediction, "answer"):
        return False

    # Simple heuristic: check if key terms from answer appear in context
    answer_terms = set(prediction.answer.lower().split())
    context_text = " ".join(prediction.contexts).lower()

    # At least 70% of answer terms should be in context
    grounded_terms = sum(1 for term in answer_terms if term in context_text)
    faithfulness_ratio = grounded_terms / len(answer_terms) if answer_terms else 0

    return faithfulness_ratio >= 0.7


# ============================================================================
# REASONING-SPECIFIC METRICS
# ============================================================================


def reasoning_steps_present(
    example: dspy.Example,
    prediction: dspy.Prediction,
    min_steps: int = 2,
) -> bool:
    """
    Check if multi-hop reasoning includes explicit steps.

    Use for: Complex QA where showing work is important
    """
    if not hasattr(prediction, "reasoning_steps"):
        return False

    # Count steps (naive: split by newlines or numbered points)
    steps = [
        s.strip()
        for s in prediction.reasoning_steps.split("\n")
        if s.strip() and len(s.strip()) > 10
    ]

    return len(steps) >= min_steps


# ============================================================================
# HYBRID METRICS (Recommended for Production)
# ============================================================================


def hybrid_qa_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    use_llm_judge: bool = True,
) -> float:
    """
    Combine multiple signals for robust evaluation.

    Scoring:
    1. Exact match = 1.0 (perfect)
    2. Semantic match = 0.8
    3. LLM judge pass = 0.6
    4. Otherwise = 0.0

    Returns float instead of bool for more granular optimization.
    """
    # Check exact match first (cheapest)
    if exact_match(example, prediction):
        return 1.0

    # Check semantic similarity (moderate cost)
    if semantic_similarity_metric(example, prediction, threshold=0.8):
        return 0.8

    # Use LLM judge as fallback (most expensive)
    if use_llm_judge:
        judge = LLMJudgeMetric()
        if judge(example, prediction, threshold=0.7):
            return 0.6

    return 0.0


def rag_quality_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
) -> float:
    """
    Comprehensive metric for RAG systems.

    Evaluates:
    1. Retrieval quality (are relevant docs retrieved?)
    2. Answer faithfulness (is answer grounded in context?)
    3. Answer correctness (does it match expected answer?)

    Each component weighted equally.
    """
    scores = []

    # 1. Retrieval quality
    if retrieval_relevance(example, prediction, min_relevant_docs=1):
        scores.append(1.0)
    else:
        scores.append(0.0)

    # 2. Faithfulness
    if answer_faithfulness(example, prediction):
        scores.append(1.0)
    else:
        scores.append(0.0)

    # 3. Correctness
    scores.append(hybrid_qa_metric(example, prediction, use_llm_judge=False))

    return sum(scores) / len(scores)


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================


def classification_accuracy(
    example: dspy.Example,
    prediction: dspy.Prediction,
) -> bool:
    """
    Simple accuracy for classification tasks.

    Use for: Document classification, intent detection
    """
    expected = example.category.strip().lower()
    predicted = prediction.category.strip().lower()
    return expected == predicted


def classification_with_reasoning(
    example: dspy.Example,
    prediction: dspy.Prediction,
    min_reasoning_length: int = 20,
) -> bool:
    """
    Require both correct classification AND reasoning.

    Forces model to explain decisions (better for debugging).
    """
    correct_class = classification_accuracy(example, prediction)
    has_reasoning = (
        hasattr(prediction, "reasoning")
        and len(prediction.reasoning) >= min_reasoning_length
    )

    return correct_class and has_reasoning


# ============================================================================
# UTILITY: METRIC FACTORY
# ============================================================================


def get_metric(metric_name: str, **kwargs) -> callable:
    """
    Factory function to get metric by name.

    Usage:
        metric = get_metric("hybrid_qa")
        score = metric(example, prediction)
    """
    metrics_registry = {
        "exact_match": exact_match,
        "substring_match": substring_match,
        "semantic_f1": semantic_similarity_metric,
        "llm_judge": LLMJudgeMetric(**kwargs),
        "hybrid_qa": hybrid_qa_metric,
        "rag_quality": rag_quality_metric,
        "classification": classification_accuracy,
        "classification_reasoning": classification_with_reasoning,
        "retrieval_relevance": retrieval_relevance,
        "answer_faithfulness": answer_faithfulness,
    }

    if metric_name not in metrics_registry:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available: {list(metrics_registry.keys())}"
        )

    return metrics_registry[metric_name]


# ============================================================================
# METRIC DEBUGGING UTILITIES
# ============================================================================


def debug_metric_failures(
    module: dspy.Module,
    dataset: List[dspy.Example],
    metric: callable,
    output_path: str = "metric_failures.jsonl",
):
    """
    Run module on dataset and log all metric failures.

    Critical for improving training data and understanding edge cases.

    Args:
        module: DSPy module to evaluate
        dataset: List of examples
        metric: Metric function
        output_path: Where to save failures
    """
    import json

    failures = []

    for example in dataset:
        prediction = module(**example.inputs())

        try:
            score = metric(example, prediction)
            if not score or score < 0.5:  # Failed
                failures.append(
                    {
                        "example": example.toDict(),
                        "prediction": prediction.toDict(),
                        "score": float(score) if isinstance(score, (int, float)) else 0,
                    }
                )
        except Exception as e:
            failures.append(
                {
                    "example": example.toDict(),
                    "prediction": prediction.toDict(),
                    "error": str(e),
                }
            )

    # Save failures
    with open(output_path, "w") as f:
        for failure in failures:
            f.write(json.dumps(failure) + "\n")

    print(f"✓ Saved {len(failures)} failures to {output_path}")
    return failures
