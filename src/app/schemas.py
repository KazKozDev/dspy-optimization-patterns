"""
Pydantic Schemas for API Requests/Responses

These define the contract between client and server.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class QuestionRequest(BaseModel):
    """Request for question-answering."""

    question: str = Field(..., description="User's question", min_length=1)
    context: Optional[str] = Field(None, description="Optional context for answering")
    conversation_history: Optional[str] = Field(
        default="", description="Previous conversation turns"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "context": "France is a country in Europe. Paris is its capital.",
                "conversation_history": "",
            }
        }


class ClassificationRequest(BaseModel):
    """Request for document classification."""

    text: str = Field(..., description="Text to classify", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "text": "This is a groundbreaking AI research paper on transformers."
            }
        }


class RAGRequest(BaseModel):
    """Request for RAG (Retrieval-Augmented Generation)."""

    question: str = Field(..., description="User's question", min_length=1)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    conversation_history: Optional[str] = Field(
        default="", description="Previous conversation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How does DSPy optimization work?",
                "top_k": 5,
                "conversation_history": "",
            }
        }


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class QuestionResponse(BaseModel):
    """Response for question-answering."""

    answer: str = Field(..., description="Generated answer")
    reasoning: Optional[str] = Field(None, description="Chain-of-thought reasoning")
    confidence: Optional[float] = Field(None, description="Confidence score 0-1")
    model_version: str = Field(..., description="Version of compiled model used")
    prompt_used: Optional[str] = Field(None, description="The actual prompt sent to the LLM")
    execution_time_ms: Optional[float] = Field(None, description="How long the request took")
    logs: Optional[List[str]] = Field(default_factory=list, description="Execution logs")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The capital of France is Paris.",
                "reasoning": "Based on the provided context...",
                "confidence": 0.95,
                "model_version": "rag_v1_mipro",
                "prompt_used": "Answer the following question based on the context...",
                "execution_time_ms": 1234.5,
                "logs": ["Processing question...", "Generating answer..."],
            }
        }


class ClassificationResponse(BaseModel):
    """Response for classification."""

    category: str = Field(..., description="Predicted category")
    confidence: Optional[float] = Field(None, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Explanation")
    model_version: str = Field(..., description="Model version")
    prompt_used: Optional[str] = Field(None, description="The actual prompt sent to the LLM")
    execution_time_ms: Optional[float] = Field(None, description="How long the request took")
    logs: Optional[List[str]] = Field(default_factory=list, description="Execution logs")

    class Config:
        json_schema_extra = {
            "example": {
                "category": "research",
                "confidence": 0.92,
                "reasoning": "The text discusses AI research...",
                "model_version": "classifier_v2",
                "prompt_used": "Classify the following text into one of...",
                "execution_time_ms": 856.3,
                "logs": ["Classifying text...", "Computing confidence..."],
            }
        }


class RAGResponse(BaseModel):
    """Response for RAG system."""

    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Retrieved source documents")
    search_query: str = Field(..., description="Optimized search query used")
    reasoning: Optional[str] = Field(None, description="Reasoning steps")
    model_version: str = Field(..., description="Model version")
    prompt_used: Optional[str] = Field(None, description="The actual prompt sent to the LLM")
    execution_time_ms: Optional[float] = Field(None, description="How long the request took")
    retrieval_time_ms: Optional[float] = Field(None, description="Time spent retrieving documents")
    logs: Optional[List[str]] = Field(default_factory=list, description="Execution logs")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "DSPy optimization uses teacher-student...",
                "sources": ["Document 1...", "Document 2..."],
                "search_query": "dspy optimization teacher student",
                "reasoning": "First, I searched...",
                "model_version": "rag_v1_mipro",
                "prompt_used": "Given the following context, answer the question...",
                "execution_time_ms": 2145.6,
                "retrieval_time_ms": 523.2,
                "logs": ["Retrieving documents...", "Generating answer..."],
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Current model version")
    uptime_seconds: float = Field(..., description="Service uptime")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
