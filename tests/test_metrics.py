"""
Unit tests for metrics.

Metrics are critical for DSPy optimization - test them thoroughly!
"""

import pytest
import dspy

from src.core.metrics import (
    exact_match,
    substring_match,
    classification_accuracy,
    get_metric,
)


class TestBasicMetrics:
    """Test heuristic metrics."""

    def test_exact_match_true(self):
        """Test exact match with matching answers."""
        example = dspy.Example(question="test", answer="Paris").with_inputs("question")
        prediction = dspy.Prediction(answer="Paris")

        assert exact_match(example, prediction) is True

    def test_exact_match_false(self):
        """Test exact match with different answers."""
        example = dspy.Example(question="test", answer="Paris").with_inputs("question")
        prediction = dspy.Prediction(answer="London")

        assert exact_match(example, prediction) is False

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        example = dspy.Example(question="test", answer="PARIS").with_inputs("question")
        prediction = dspy.Prediction(answer="paris")

        assert exact_match(example, prediction) is True

    def test_substring_match_true(self):
        """Test substring match with partial match."""
        example = dspy.Example(question="test", answer="Paris").with_inputs("question")
        prediction = dspy.Prediction(answer="The capital is Paris, France")

        assert substring_match(example, prediction) is True

    def test_substring_match_false(self):
        """Test substring match with no overlap."""
        example = dspy.Example(question="test", answer="Paris").with_inputs("question")
        prediction = dspy.Prediction(answer="London")

        assert substring_match(example, prediction) is False


class TestClassificationMetrics:
    """Test classification-specific metrics."""

    def test_classification_accuracy_true(self):
        """Test correct classification."""
        example = dspy.Example(
            text="test document", category="technology"
        ).with_inputs("text")
        prediction = dspy.Prediction(category="technology")

        assert classification_accuracy(example, prediction) is True

    def test_classification_accuracy_false(self):
        """Test incorrect classification."""
        example = dspy.Example(
            text="test document", category="technology"
        ).with_inputs("text")
        prediction = dspy.Prediction(category="science")

        assert classification_accuracy(example, prediction) is False


class TestMetricFactory:
    """Test metric factory function."""

    def test_get_metric_exact_match(self):
        """Test retrieving exact_match metric."""
        metric = get_metric("exact_match")
        assert callable(metric)
        assert metric == exact_match

    def test_get_metric_unknown(self):
        """Test error on unknown metric."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("nonexistent_metric")
