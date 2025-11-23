"""
Example 3: RAG with Vector Database

This example demonstrates:
1. Setting up a vector database (Chroma)
2. Ingesting documents
3. Using RAG module with real retrieval
4. Optimizing the RAG pipeline
"""

import dspy
from src.core.modules import SimpleRAG
from src.integrations.vector_db import ChromaRetriever, create_retriever
from src.core.metrics import rag_quality_metric

# 1. Setup vector database
print("Setting up vector database...")
retriever = create_retriever(
    provider="chroma",
    collection_name="dspy_docs",
    persist_directory=".chroma_demo",
)

# 2. Ingest sample documents
print("Ingesting documents...")
documents = [
    "DSPy is a framework for programming with foundation models.",
    "Optimization in DSPy uses teacher-student patterns.",
    "MIPRO is an advanced optimizer that generates instruction prompts.",
    "Metrics define what success looks like in DSPy.",
    "Compiled programs are saved as JSON artifacts.",
]

metadata = [{"source": f"doc_{i}"} for i in range(len(documents))]
retriever.upsert(documents, metadata)

# 3. Configure LLM
lm = dspy.LM(model="openai/gpt-4o-mini", temperature=0.0)
dspy.settings.configure(lm=lm)

# 4. Create RAG module
rag = SimpleRAG()

# 5. Create retriever function
def vector_db_retriever(query: str):
    """Retrieve from vector DB"""
    results = retriever.search(query, top_k=3)
    return [r.text for r in results]

# 6. Test RAG
print("\nTesting RAG...")
result = rag.forward(
    question="What is MIPRO?",
    retriever_fn=vector_db_retriever,
    conversation_history="",
)

print("Question:", "What is MIPRO?")
print("Search Query:", result.search_query)
print("Retrieved Contexts:", result.contexts)
print("Answer:", result.answer)

# 7. Optimization (optional)
print("\n--- Optimization Demo ---")
print("To optimize, prepare a dataset and run:")
print("python -m src.pipeline.optimizer \\")
print("  --module SimpleRAG \\")
print("  --data data/processed/rag_dataset.jsonl \\")
print("  --metric rag_quality \\")
print("  --optimizer mipro")
