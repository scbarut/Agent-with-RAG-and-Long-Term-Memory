"""Long-term (MongoDB) and short-term (in-memory) memory store setup.

Exports
-------
long_term_memory  : MongoDBStore  — persists user profiles across threads.
short_term_memory : MemorySaver   — maintains conversation state within a thread.
get_user_memory   : helper to retrieve a user's stored profile text.
save_user_memory  : helper to persist an updated profile.
get_user_id_from_config : extract the user_id from a LangGraph RunnableConfig.
"""

import os

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

from agent.configuration import Configuration

load_dotenv()

# ---------------------------------------------------------------------------
# MongoDB connection (credentials read from environment)
# ---------------------------------------------------------------------------

_MONGO_USERNAME = os.getenv("MONGO_USERNAME")
_MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
_MONGO_URL = (
    f"mongodb+srv://{_MONGO_USERNAME}:{_MONGO_PASSWORD}"
    "@cluster0.vzjzms8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
)

_mongo_client = MongoClient(_MONGO_URL)
_db = _mongo_client["agent_rag_db"]
_collection = _db["user_memory"]

# Public store instances
long_term_memory = MongoDBStore(collection=_collection)   # across-thread persistence
short_term_memory = MemorySaver()                          # within-thread checkpointing

# ---------------------------------------------------------------------------
# Memory namespace constants
# ---------------------------------------------------------------------------

_MEMORY_NAMESPACE_PREFIX = "memory"
_MEMORY_KEY = "user_memory"
_NO_MEMORY_MSG = "No existing memory found."

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_user_memory(store: MongoDBStore, user_id: str) -> str:
    """Retrieve the stored memory for *user_id*, or a default placeholder."""
    namespace = (_MEMORY_NAMESPACE_PREFIX, user_id)
    record = store.get(namespace, _MEMORY_KEY)
    return record.value.get("memory") if record else _NO_MEMORY_MSG


def save_user_memory(store: MongoDBStore, user_id: str, memory_content: str) -> None:
    """Persist *memory_content* as the updated profile for *user_id*."""
    namespace = (_MEMORY_NAMESPACE_PREFIX, user_id)
    store.put(namespace, _MEMORY_KEY, {"memory": memory_content})


def get_user_id_from_config(config: RunnableConfig) -> str:
    """Extract ``user_id`` from a LangGraph :class:`RunnableConfig`."""
    return Configuration.from_runnable_config(config).user_id
