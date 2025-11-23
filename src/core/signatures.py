"""
DSPy Signatures - Contracts for LLM I/O

Signatures define the interface between your logic and the LLM.
They specify input/output fields and provide docstrings that DSPy uses
as part of the prompt.

Best Practices:
1. Keep docstrings clear and specific
2. Use descriptive field names
3. Add format hints in descriptions
4. Version signatures when changing behavior
"""

import dspy


class GenerateAnswer(dspy.Signature):
    """Answer the question based on the provided context."""

    context: str = dspy.InputField(desc="Relevant information to answer the question")
    question: str = dspy.InputField(desc="User's question")
    answer: str = dspy.OutputField(desc="Concise answer based on context")


class ExtractIntent(dspy.Signature):
    """Extract user intent from natural language query."""

    query: str = dspy.InputField(desc="User's natural language query")
    intent: str = dspy.OutputField(desc="Primary intent: question, command, feedback, or other")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")


class GenerateSearchQuery(dspy.Signature):
    """Generate an optimized search query for information retrieval."""

    user_question: str = dspy.InputField(desc="Original user question")
    conversation_history: str = dspy.InputField(desc="Previous conversation turns")
    search_query: str = dspy.OutputField(desc="Optimized search query for retrieval system")


class ClassifyDocument(dspy.Signature):
    """Classify document into predefined categories."""

    document_text: str = dspy.InputField(desc="Full document content")
    categories: str = dspy.InputField(desc="Comma-separated list of valid categories")
    category: str = dspy.OutputField(desc="Most relevant category from the list")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")


class SummarizeWithStyle(dspy.Signature):
    """Generate a summary with specific style and length constraints."""

    text: str = dspy.InputField(desc="Text to summarize")
    style: str = dspy.InputField(desc="Style: technical, casual, formal, or executive")
    max_words: int = dspy.InputField(desc="Maximum number of words in summary")
    summary: str = dspy.OutputField(desc="Summary matching specified style and length")


class ValidateLogicalConsistency(dspy.Signature):
    """Check if a conclusion logically follows from premises."""

    premises: str = dspy.InputField(desc="List of premises or facts")
    conclusion: str = dspy.InputField(desc="Conclusion to validate")
    is_valid: bool = dspy.OutputField(desc="True if conclusion follows from premises")
    explanation: str = dspy.OutputField(desc="Explanation of logical reasoning")


class GenerateCode(dspy.Signature):
    """Generate code snippet based on requirements."""

    requirements: str = dspy.InputField(desc="Functional requirements in natural language")
    language: str = dspy.InputField(desc="Programming language: python, javascript, rust, etc.")
    code: str = dspy.OutputField(desc="Complete, runnable code snippet")
    tests: str = dspy.OutputField(desc="Unit tests for the generated code")


class MultiHopQA(dspy.Signature):
    """Answer complex questions requiring multiple reasoning steps."""

    question: str = dspy.InputField(desc="Complex question requiring multi-hop reasoning")
    context: str = dspy.InputField(desc="Available information sources")
    reasoning_steps: str = dspy.OutputField(desc="Step-by-step reasoning process")
    final_answer: str = dspy.OutputField(desc="Final answer to the question")


class EvaluateAnswer(dspy.Signature):
    """Evaluate quality of an answer (LLM-as-Judge pattern)."""

    question: str = dspy.InputField(desc="Original question")
    expected_answer: str = dspy.InputField(desc="Ground truth or ideal answer")
    generated_answer: str = dspy.InputField(desc="Answer to evaluate")
    score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="Constructive feedback on the answer")
