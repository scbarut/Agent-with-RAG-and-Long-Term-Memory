"""
Graph assembly entrypoint.

This module builds and compiles the LangGraph workflow.  It is intentionally
thin — all business logic lives in the ``agent/`` package.

Referenced by langgraph.json:
    "agent": "./graph.py:graph"
"""

from functools import partial

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from agent.configuration import Configuration
from agent.memory_store import long_term_memory
from agent.nodes.answer import generate_answer
from agent.nodes.decision import decision_node, decision_router
from agent.nodes.memory import write_memory
from agent.nodes.planner import planner_node
from agent.nodes.rag_query import rag_query
from agent.nodes.researcher import researcher_node
from agent.nodes.rewriting import rewriting_node
from agent.nodes.router import router
from agent.state import AgentState

load_dotenv()

# ---------------------------------------------------------------------------
# Workflow graph
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState, config_schema=Configuration)

# --- Nodes ---
workflow.add_node("decision_node",  partial(decision_node,   store=long_term_memory))
workflow.add_node("planner_node",   planner_node)
workflow.add_node("rewriting_node", rewriting_node)
workflow.add_node("rag_query",      rag_query)
workflow.add_node("researcher_node",researcher_node)
workflow.add_node("generate_answer",partial(generate_answer, store=long_term_memory))
workflow.add_node("write_memory",   partial(write_memory,    store=long_term_memory))

# --- Edges ---
workflow.add_edge(START, "decision_node")

workflow.add_conditional_edges(
    "decision_node",
    decision_router,
    {"end": "write_memory", "planning": "planner_node"},
)

workflow.add_conditional_edges(
    "planner_node",
    router,
    {
        "researcher":      "researcher_node",
        "rewriting":       "rewriting_node",
        "generate_answer": "generate_answer",
    },
)

workflow.add_conditional_edges(
    "researcher_node",
    router,
    {"rewriting": "rewriting_node", "generate_answer": "generate_answer"},
)

workflow.add_conditional_edges(
    "rag_query",
    router,
    {"researcher": "researcher_node", "generate_answer": "generate_answer"},
)

workflow.add_edge("rewriting_node", "rag_query")
workflow.add_edge("generate_answer", "write_memory")
workflow.add_edge("write_memory", END)

# ---------------------------------------------------------------------------
# Compiled graph — exported for langgraph.json
# ---------------------------------------------------------------------------

graph = workflow.compile()
