"""Researcher node.

Performs web research via DuckDuckGo using a zero-shot ReAct agent.
"""

from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun

from agent.llm import llm
from agent.state import AgentState

_RESEARCHER_PREFIX = (
    "You are a research assistant. "
    "Ignore irrelevant or non-researchable parts of the query. "
    "Provide a clear and concise answer only for the researched part. "
    "If you need external information, use DuckDuckGo for web search."
)


def researcher_node(state: AgentState) -> dict:
    """Run a DuckDuckGo-powered research agent and return its answer."""
    user_query = state["messages"][-1].content

    agent = initialize_agent(
        tools=[DuckDuckGoSearchRun()],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        agent_kwargs={"prefix": _RESEARCHER_PREFIX},
    )

    response = agent.invoke({"input": user_query})
    return {"research_answer": response["output"]}
