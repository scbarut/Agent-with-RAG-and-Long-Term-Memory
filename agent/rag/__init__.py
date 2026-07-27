"""RAG (Retrieval-Augmented Generation) sub-package.

The ``rag_system`` module-level singleton is instantiated here so that all
agent nodes share a single instance of :class:`RagSystem` (avoiding duplicate
model loading).
"""

from agent.rag.system import RagSystem

# Shared singleton — import this wherever RAG operations are needed.
rag_system = RagSystem()

__all__ = ["RagSystem", "rag_system"]
