"""
Unit tests for DSPy modules.

These tests ensure modules work correctly in both zero-shot and optimized modes.
"""

import pytest
from unittest.mock import Mock, patch
import dspy

from src.core.modules import SimpleRAG, DocumentClassifier, BaseModule


class TestBaseModule:
    """Test base module functionality."""

    def test_base_module_initialization(self):
        """Test BaseModule initializes correctly."""
        module = BaseModule()
        assert module.version == "unoptimized"

    def test_base_module_with_compiled_state(self, tmp_path):
        """Test loading compiled state."""
        # Create a mock compiled state file
        compiled_state_path = tmp_path / "test_module.json"
        compiled_state_path.write_text('{"test": "data"}')

        # This would normally work with a real DSPy module
        # For now, just test the path is stored
        module = BaseModule(compiled_state_path=str(compiled_state_path))
        assert module._compiled_state_path == str(compiled_state_path)


class TestSimpleRAG:
    """Test SimpleRAG module."""

    def test_simple_rag_initialization(self):
        """Test SimpleRAG initializes without errors."""
        rag = SimpleRAG()
        assert rag is not None
        assert hasattr(rag, "generate_query")
        assert hasattr(rag, "generate_answer")

    @pytest.mark.skip(reason="Requires LLM API key")
    def test_simple_rag_forward(self):
        """Test RAG forward pass (requires API key)."""
        rag = SimpleRAG()

        # Mock retriever
        def mock_retriever(query: str):
            return [
                "DSPy is a framework for programming with LLMs.",
                "It uses optimization to improve prompts.",
            ]

        # Run prediction
        result = rag.forward(
            question="What is DSPy?",
            retriever_fn=mock_retriever,
            conversation_history="",
        )

        assert hasattr(result, "answer")
        assert hasattr(result, "search_query")
        assert hasattr(result, "contexts")


class TestDocumentClassifier:
    """Test DocumentClassifier module."""

    def test_classifier_initialization(self):
        """Test classifier initializes with categories."""
        categories = ["technology", "science", "business"]
        classifier = DocumentClassifier(categories=categories)

        assert classifier is not None
        assert classifier.categories == categories

    @pytest.mark.skip(reason="Requires LLM API key")
    def test_classifier_forward(self):
        """Test classifier prediction."""
        categories = ["technology", "science", "business"]
        classifier = DocumentClassifier(categories=categories)

        result = classifier(
            document_text="This is an article about AI and machine learning."
        )

        assert hasattr(result, "category")
        assert result.category in categories
