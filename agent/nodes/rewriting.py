"""Query rewriting node.

Rewrites the raw user query into a concise retrieval query and generates a
HyDE (Hypothetical Document Embedding) document for improved RAG retrieval.
"""

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from agent.llm import llm
from agent.state import AgentState

_REWRITE_PROMPT = PromptTemplate(
    input_variables=["original_query"],
    template="""\
You are a rewriting agent for a RAG system specialised in clothing and fashion data.
Extract **only the part of the user's input relevant to this RAG system**, ignoring
unrelated requests (e.g., web search, general knowledge, non-fashion topics).

Produce three structured items:
1) query: a concise, search-friendly version of the user's RAG-relevant request (1-2 sentences).
2) requires_output_img: boolean — True only if the user explicitly asks for an image or visual.
3) hyde_document: a standalone, retrieval-friendly hypothetical document (HyDE) that fully
   answers the extracted query, containing:
   - SUMMARY: 1-2 sentence concise answer
   - KEY FACTS: 3-6 facts or claims; mark uncertain items with [VERIFY]
   - SUGGESTED RETRIEVAL QUERIES: 5-8 variant search queries or keywords
   - KEYWORDS / SYNONYMS / ALIASES: compact list
   - DOCUMENT TYPES: e.g., "catalog", "product description", "fashion article"

**IMPORTANT RULES**
- Always produce a HyDE for the extracted RAG query.
- If the user mentions multiple topics, **ignore unrelated topics**. Only include fashion/clothing parts.
- requires_output_img = True only if the user explicitly requests a visual for a clothing/fashion query.
- Output only JSON with exactly these keys: query, requires_output_img, hyde_document.

User original input (raw): {original_query}
""",
)


class RewritingOutput(BaseModel):
    query: str = Field(description="Concise, cleaned query for retrieval (1-2 sentences).")
    hyde_document: str = Field(
        description=(
            "Detailed hypothetical document (HyDE) that directly answers the query and provides "
            "retrieval-friendly keywords, suggested search queries, entity/date hints, and a short summary."
        )
    )
    requires_output_img: bool = Field(
        description="True if the user explicitly requests an image/infographic/etc., otherwise False."
    )


def rewriting_node(state: AgentState) -> dict:
    """Rewrite the latest user query and produce a HyDE document for better retrieval."""
    original_query = state["messages"][-1].content
    prompt = _REWRITE_PROMPT.format(original_query=original_query)

    rewriting_llm = llm.with_structured_output(RewritingOutput)
    response: RewritingOutput = rewriting_llm.invoke(prompt)

    # Prefer the richer HyDE document; fall back to the concise query
    retrieval_text = response.hyde_document if response and response.hyde_document else response.query

    return {"query": retrieval_text, "requires_output_img": response.requires_output_img}
