# configuration.py
# ---------------------------------------------------------------------------
# Backward-compatibility shim.
# The canonical implementation has moved to agent/configuration.py.
# This file is retained so that any existing code doing
#   `import configuration` or `from configuration import Configuration`
# continues to work without modification.
# ---------------------------------------------------------------------------
from agent.configuration import Configuration  # noqa: F401

__all__ = ["Configuration"]