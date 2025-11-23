"""
DSPy Modules - Business Logic with Compiled State Loading

Key Concept: Modules should work in TWO modes:
1. Zero-shot (no compiled state) - uses default prompts
2. Optimized (with compiled state) - loads few-shot examples and optimized instructions

Critical Pattern:
- Accept `compiled_state_path` in __init__
- If path provided, load using `self.load(path)` AFTER initialization
- This separates definition from optimization

Example Usage:
    # Development (zero-shot)
    rag = RAGModule()

    # Production (optimized)
    rag = RAGModule()
    rag.load("artifacts/compiled_programs/rag_v1_mipro.json")
"""

import json
from pathlib import Path
from typing import Optional, List

import dspy

from .signatures import (
    GenerateAnswer,
    GenerateSearchQuery,
    ClassifyDocument,
    MultiHopQA,
)


class BaseModule(dspy.Module):
    """
    Base class for all modules with compiled state management.

    Provides:
    - Standardized loading/saving of compiled programs
    - Version tracking
    - Logging and debugging utilities
    """

    def __init__(self, compiled_state_path: Optional[str] = None):
        super().__init__()
        self._compiled_state_path = compiled_state_path
        self._version = "unoptimized"

    def load_compiled_state(self, path: str):
        """Load optimized prompts, few-shot examples, and instructions."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Compiled state not found: {path}")

        self.load(str(path_obj))
        self._version = path_obj.stem
        print(f"✓ Loaded compiled state: {self._version}")

    def save_compiled_state(self, path: str):
        """Save optimized program to artifacts."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self.save(str(path_obj))
        print(f"✓ Saved compiled state: {path}")

    @property
    def version(self) -> str:
        """Return version identifier (filename of compiled state)."""
        return self._version


class SimpleRAG(BaseModule):
    """
    Basic RAG (Retrieval-Augmented Generation) module.

    Flow:
    1. Generate optimized search query
    2. Retrieve relevant context (external retriever)
    3. Generate answer based on context

    Optimization Points:
    - Query generation can learn to rephrase for better retrieval
    - Answer generation learns to cite sources, format properly
    """

    def __init__(self, compiled_state_path: Optional[str] = None):
        super().__init__(compiled_state_path)

        # Define sub-modules (these will be optimized during compile())
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

        # Load compiled state if provided
        if compiled_state_path:
            self.load_compiled_state(compiled_state_path)

    def forward(
        self,
        question: str,
        retriever_fn: callable,
        conversation_history: str = "",
    ):
        """
        Execute RAG pipeline.

        Args:
            question: User's question
            retriever_fn: Function that takes query string and returns List[str] contexts
            conversation_history: Previous conversation for context

        Returns:
            dspy.Prediction with answer and intermediate steps
        """
        # Step 1: Generate optimized search query
        query_result = self.generate_query(
            user_question=question,
            conversation_history=conversation_history,
        )
        search_query = query_result.search_query

        # Step 2: Retrieve context (external system)
        contexts = retriever_fn(search_query)
        combined_context = "\n\n".join(contexts)

        # Step 3: Generate answer from context
        answer_result = self.generate_answer(
            context=combined_context, question=question
        )

        return dspy.Prediction(
            answer=answer_result.answer,
            search_query=search_query,
            contexts=contexts,
            reasoning=answer_result.reasoning,  # From ChainOfThought
        )


class DocumentClassifier(BaseModule):
    """
    Multi-class document classifier with reasoning.

    Optimization learns:
    - How to extract relevant features from text
    - How to map edge cases to categories
    - When to use reasoning vs direct classification
    """

    def __init__(
        self,
        categories: List[str],
        compiled_state_path: Optional[str] = None,
    ):
        super().__init__(compiled_state_path)
        self.categories = categories
        self.classify = dspy.ChainOfThought(ClassifyDocument)

        if compiled_state_path:
            self.load_compiled_state(compiled_state_path)

    def forward(self, document_text: str):
        """Classify document into one of predefined categories."""
        result = self.classify(
            document_text=document_text,
            categories=", ".join(self.categories),
        )

        return dspy.Prediction(
            category=result.category,
            reasoning=result.reasoning,
        )


class MultiHopReasoner(BaseModule):
    """
    Complex reasoning module for multi-hop questions.

    Uses ReAct pattern:
    - Thought: What do I need to find?
    - Action: Search for information
    - Observation: What did I find?
    - Repeat until answer is complete

    This is where DSPy shines - it learns to break down complex questions.
    """

    def __init__(self, compiled_state_path: Optional[str] = None):
        super().__init__(compiled_state_path)

        # ReAct module handles iterative reasoning
        self.react = dspy.ReAct(MultiHopQA)

        if compiled_state_path:
            self.load_compiled_state(compiled_state_path)

    def forward(self, question: str, retriever_fn: callable, max_hops: int = 5):
        """
        Answer complex question through iterative reasoning.

        Args:
            question: Multi-hop question
            retriever_fn: Function to retrieve information
            max_hops: Maximum reasoning steps

        Returns:
            Prediction with reasoning steps and final answer
        """
        # Build context through iterative retrieval
        # In production, you'd implement a proper ReAct loop with tools
        initial_context = "\n".join(retriever_fn(question))

        result = self.react(question=question, context=initial_context)

        return dspy.Prediction(
            final_answer=result.final_answer,
            reasoning_steps=result.reasoning_steps,
        )


class AdaptiveModule(BaseModule):
    """
    Advanced pattern: Module that selects strategy based on input.

    Example: For simple questions, use direct QA.
             For complex questions, use multi-hop reasoning.

    The optimizer learns WHEN to use which strategy.
    """

    def __init__(self, compiled_state_path: Optional[str] = None):
        super().__init__(compiled_state_path)

        # Simple path
        self.simple_qa = dspy.ChainOfThought(GenerateAnswer)

        # Complex path
        self.complex_qa = dspy.ChainOfThought(MultiHopQA)

        # Router (learns to classify question complexity)
        self.route = dspy.Predict("question -> complexity: str")

        if compiled_state_path:
            self.load_compiled_state(compiled_state_path)

    def forward(self, question: str, context: str):
        """Route to appropriate QA strategy based on question complexity."""
        # Determine complexity
        routing = self.route(question=question)

        if "simple" in routing.complexity.lower():
            result = self.simple_qa(question=question, context=context)
            return dspy.Prediction(answer=result.answer, strategy="simple")
        else:
            result = self.complex_qa(question=question, context=context)
            return dspy.Prediction(
                final_answer=result.final_answer,
                reasoning_steps=result.reasoning_steps,
                strategy="complex",
            )


def load_module_from_artifact(artifact_path: str, module_class: type) -> BaseModule:
    """
    Factory function to load ANY module from compiled artifact.

    Usage:
        rag = load_module_from_artifact(
            "artifacts/compiled_programs/rag_v2.json",
            SimpleRAG
        )
    """
    # Create instance without compiled state
    if module_class == DocumentClassifier:
        # Special case: needs categories list
        # In production, save categories in artifact metadata
        module = module_class(categories=["tech", "business", "science"])
    else:
        module = module_class()

    # Load compiled state
    module.load_compiled_state(artifact_path)

    return module
