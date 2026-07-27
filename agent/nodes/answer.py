"""Answer generation node.

Synthesises RAG context, web-research results, and user memory into the
final response delivered to the user.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.store.mongodb import MongoDBStore
from pydantic import BaseModel, Field

from agent.llm import llm
from agent.memory_store import get_user_id_from_config, get_user_memory
from agent.state import AgentState

_GENERATE_ANSWER_PROMPT = """\
You are an AI assistant. Generate a clear, accurate, and helpful response using the \
context and research answer provided below.

**Output image available (may be empty — no need to display it):** {requires_output_img}

**User input:**
{user_input}

**RAG context (may be empty):**
{rag_context}

**Researcher answer (may be empty):**
{researcher_answer}

**Memory (may be empty):**
{memory}

If memory is available for this user, use it to personalise your response:
    * Greetings and transitions
    * Guidance tailored to the user's tools and frameworks
    * Follow-up messages that continue from past context

The answer must be in the same language as the user input.
Generate the best possible response.
"""


def generate_answer(state: AgentState, config: RunnableConfig, store: MongoDBStore) -> dict:
    """Generate the final answer by combining RAG context, research, and user memory."""
    user_id = get_user_id_from_config(config)
    memory_content = get_user_memory(store, user_id)

    prompt_template = ChatPromptTemplate.from_template(_GENERATE_ANSWER_PROMPT)
    prompt = prompt_template.format_messages(
        user_input=state["messages"][-1].content,
        rag_context=state.get("rag_context", ""),
        researcher_answer=state.get("research_answer", ""),
        requires_output_img=state.get("requires_output_img", ""),
        memory=memory_content,
    )

    response = llm.invoke(prompt)
    return {"final_answer": response.content, "output_img": state.get("output_img", "")}
