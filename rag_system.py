# rag_system.py
# ---------------------------------------------------------------------------
# Backward-compatibility shim.
# The canonical implementation has moved to agent/rag/system.py.
# This file is retained so that any existing code doing
#   `from rag_system import RagSystem`
# continues to work without modification.
# ---------------------------------------------------------------------------
from agent.rag.system import RagSystem  # noqa: F401

__all__ = ["RagSystem"]