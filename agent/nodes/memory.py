"""Memory write node.

Extracts new user-profile information from the conversation and persists it
to the long-term MongoDB store so it can personalise future sessions.
"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.mongodb import MongoDBStore
from pydantic import BaseModel, Field

from agent.llm import llm
from agent.memory_store import get_user_id_from_config, get_user_memory, save_user_memory
from agent.state import AgentState

_CREATE_MEMORY_PROMPT = """\
You are collecting information about the user to personalise future responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully.
2. Identify new information about the user, such as:
   - Personal details (name, location, etc.)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with the existing memory.
4. Format the memory as a clear, bulleted list.
5. If new information conflicts with existing memory, keep the most recent version.

Rules:
- Only include factual information directly stated by the user. Do not make assumptions.
- Do NOT add summaries like "no update needed" or your own commentary.
- If no new data is present in the chat history, return the existing user information unmodified.

Update the user information based on the chat history below:
"""


class MemoryOutput(BaseModel):
    memory: str = Field(
        description="Updated user profile information extracted from the chat history."
    )


def write_memory(state: AgentState, config: RunnableConfig, store: MongoDBStore) -> None:
    """Extract new user information from the conversation and persist it to the store."""
    user_id = get_user_id_from_config(config)
    existing_memory = get_user_memory(store, user_id)

    system_msg = SystemMessage(content=_CREATE_MEMORY_PROMPT.format(memory=existing_memory))

    memory_llm = llm.with_structured_output(MemoryOutput)
    updated: MemoryOutput = memory_llm.invoke([system_msg] + list(state["messages"]))

    save_user_memory(store, user_id, updated.memory)
