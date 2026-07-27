"""Shared state definition for the agent graph."""

from typing import Annotated, Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State that is passed between every node in the LangGraph workflow."""

    # Conversation history (append-only via add_messages reducer)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Routing flags
    requires_agent: bool
    requires_add_data: bool

    # Image I/O
    # input_img accepts a URL/path string on entry; decision_node converts it to PIL.Image
    input_img: Optional[Any]
    output_img: str
    requires_output_img: bool

    # Intermediate results
    rag_context: str
    research_answer: str
    final_answer: str

    # Planner queue — list of agent names still to be executed
    planned: list[str]

    # Rewritten / HyDE query sent to the RAG store
    query: str
