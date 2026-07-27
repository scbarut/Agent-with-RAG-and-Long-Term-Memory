"""Decision node.

Decides whether the user query can be answered directly or requires the
agent pipeline (research, RAG retrieval, etc.).
"""

import base64
from io import BytesIO
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.mongodb import MongoDBStore
from pydantic import BaseModel, Field

from agent.configuration import Configuration
from agent.llm import llm
from agent.memory_store import get_user_memory
from agent.state import AgentState
from agent.utils.image import load_image

_DECISION_SYSTEM_PROMPT = """\
Based on the user's query, decide whether you need to use an agent (external tool, \
research, etc.) or if you can answer the question directly.

**Memory (may be empty):**
{memory}

- Answer directly only for simple conversational queries such as "How are you?" or \
"What's your name?", or when the answer is already known.

- When answering directly, use **Memory** to personalise your response:
    * Greetings and transitions
    * Guidance tailored to the user's tools and frameworks
    * Follow-up messages that continue from past context

- For all other queries, set requires_agent=True.
"""


class DecisionOutput(BaseModel):
    requires_agent: bool = Field(
        description="True if the query requires an agent (research, tool use, etc.); False for direct answers."
    )
    answer: Optional[str] = Field(
        default=None,
        description="A direct answer when requires_agent is False; None otherwise.",
    )


def decision_node(state: AgentState, config: RunnableConfig, store: MongoDBStore) -> AgentState:
    """Reset transient state, retrieve user memory, and decide whether to route to the agent pipeline."""
    # Reset per-turn transient fields
    state["output_img"] = ""
    state["requires_output_img"] = False
    state["rag_context"] = ""
    state["research_answer"] = ""
    state["planned"] = []
    state["requires_add_data"] = False

    # Retrieve user memory from the long-term store
    user_id = Configuration.from_runnable_config(config).user_id
    memory_content = get_user_memory(store, user_id)

    # Build the LLM payload
    messages = state["messages"]
    last_message = messages[-1]
    input_img = load_image(state.get("input_img"))

    system_message = SystemMessage(content=_DECISION_SYSTEM_PROMPT.format(memory=memory_content))
    llm_payload = [system_message]

    if input_img is None:
        # Text-only: just append the last user message
        llm_payload.append(last_message)
    else:
        # Multimodal: include prior history then a base64-encoded image message
        llm_payload.extend(messages[:-1])

        # Extract plain text from the last user message (handles both str and list content)
        last_text = ""
        if isinstance(last_message.content, str):
            last_text = last_message.content
        elif isinstance(last_message.content, list):
            for part in last_message.content:
                if part.get("type") == "text":
                    last_text = part.get("text", "")
                    break

        # Encode image as base64 PNG
        buffer = BytesIO()
        input_img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        llm_payload.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": last_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]
            )
        )

    # Invoke the decision LLM
    decision_llm = llm.with_structured_output(DecisionOutput)
    response: DecisionOutput = decision_llm.invoke(llm_payload)

    if response.answer:
        state["final_answer"] = response.answer
    state["requires_agent"] = response.requires_agent
    state["input_img"] = input_img  # store the loaded PIL.Image for downstream nodes

    return state


def decision_router(state: AgentState) -> str:
    """Route to the planner when the agent is needed, or straight to end."""
    return "planning" if state["requires_agent"] else "end"
