"""Planner node.

Selects which sub-agents to invoke for the current query and, when requested,
adds new data to the RAG vector store.
"""

from typing import Literal

from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from agent.llm import llm
from agent.rag import rag_system
from agent.state import AgentState
from agent.utils.image import load_image

_PLANNER_SYSTEM_PROMPT = """\
You are an agency planner. Your job is to decide which agents are needed based on the user's query.

Output Format (MUST follow exactly):
{
  "planned": ["Researcher Agent" or "RAG Query Agent", ...],
  "requires_add_data": true or false
}

Available Agents:
* Researcher Agent: Gathers and synthesizes information from various sources (e.g., the web).
* RAG Query Agent: Retrieves relevant, specific information for clothing-related data (Except adding data).

EXAMPLES:
1. Query: Find data on blue jeans and research what blue pants are made of.
   -> {"planned": ["Researcher Agent", "RAG Query Agent"], "requires_add_data": false}

2. Query: **IMAGE** What season is this jacket from?
   -> {"planned": ["RAG Query Agent"], "requires_add_data": false}

3. Query: Bring me image of blue jean?
   -> {"planned": ["RAG Query Agent"], "requires_add_data": false}

4. Query: When was Google founded?
   -> {"planned": ["Researcher Agent"], "requires_add_data": false}

5. Query: Find me black Puma shoes.
   -> {"planned": ["RAG Query Agent"], "requires_add_data": false}

6. Query: Add to RAG system that white watches.
   -> {"planned": [], "requires_add_data": true}
"""

_ADDING_DATA_SYSTEM_PROMPT = """\
Your task is to determine what corrected data should be added to the RAG system and whether \
the user specified that an image will be included.

- For `img_is_available`: Set it to True only if the user explicitly mentions that an image \
  will be added; otherwise, set it to False.
- For `data_to_add`: Extract and provide only the corrected information that needs to be \
  stored in the RAG system.
- Base your answer only on the parts of the input that are relevant to adding data into the \
  RAG system, and ignore any unrelated or irrelevant text.

Data must be in English.
"""


class PlannerOutput(BaseModel):
    planned: list[Literal["Researcher Agent", "RAG Query Agent"]] = Field(
        description="Ordered list of agents to invoke for this query."
    )
    requires_add_data: bool = Field(
        description="True if the user wants to add data to the RAG system."
    )


class AddDataOutput(BaseModel):
    img_is_available: bool = Field(
        description="True if the user explicitly mentions that an image will be included."
    )
    data_to_add: str = Field(
        description="The cleaned data text to store in the RAG system."
    )


def planner_node(state: AgentState) -> dict:
    """Determine which agents to run and optionally ingest new data into the RAG store."""
    planner_llm = llm.with_structured_output(PlannerOutput)
    user_query = state["messages"][-1].content

    plan: PlannerOutput = planner_llm.invoke(
        [SystemMessage(content=_PLANNER_SYSTEM_PROMPT), user_query]
    )

    if plan.requires_add_data:
        adding_llm = llm.with_structured_output(AddDataOutput)
        add_result: AddDataOutput = adding_llm.invoke(
            [SystemMessage(content=_ADDING_DATA_SYSTEM_PROMPT), user_query]
        )
        print(f"New data to add: {add_result.data_to_add}")

        # Bug fix: was incorrectly using `plan.img_is_available` (field does not exist on PlannerOutput)
        if add_result.img_is_available:
            rag_system.add_data(add_result.data_to_add, load_image(state.get("input_img")))
        else:
            rag_system.add_data(add_result.data_to_add)

    return {"planned": plan.planned, "requires_add_data": plan.requires_add_data}
