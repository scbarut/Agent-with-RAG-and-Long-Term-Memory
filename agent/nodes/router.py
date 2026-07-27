"""Shared routing logic for the agent graph.

The ``router`` function is used as a conditional-edge callback from multiple
nodes (``planner_node``, ``researcher_node``, ``rag_query``).  It pops the
next planned agent from ``state["planned"]`` and returns the corresponding
node name.
"""

from agent.state import AgentState

_RESEARCHER_AGENT = "Researcher Agent"
_RAG_QUERY_AGENT = "RAG Query Agent"


def router(state: AgentState) -> str:
    """Dispatch to the next agent based on the remaining plan.

    Pops the first item from ``state["planned"]`` and returns the
    corresponding LangGraph node name.  Returns ``"generate_answer"``
    when the plan is exhausted or unrecognised.
    """
    planned: list[str] = state.get("planned", [])

    if not planned:
        return "generate_answer"

    next_agent = planned.pop(0)  # mutate in-place; consumed one step of the plan

    if _RESEARCHER_AGENT in next_agent:
        return "researcher"
    if _RAG_QUERY_AGENT in next_agent:
        return "rewriting"

    return "generate_answer"
