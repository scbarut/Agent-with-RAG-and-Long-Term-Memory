"""RAG query node.

Retrieves relevant documents from the FAISS vector store using text, image,
or fused text+image embeddings.
"""

from agent.rag import rag_system
from agent.state import AgentState


def rag_query(state: AgentState) -> dict:
    """Search the vector store and return context (and optionally an output image link)."""
    query = state["query"]
    input_img = state.get("input_img")

    if state["requires_output_img"]:
        if input_img:
            results = rag_system.vector_search_text_img(query, input_img, k=1)
            return {
                "rag_context": results[0].page_content,
                "output_img": results[0].metadata.get("link", ""),
            }
        else:
            results = rag_system.similarity_search_with_score(query, k=1)
            return {
                "rag_context": results[0][0].page_content,
                "output_img": results[0][0].metadata.get("link", ""),
            }
    else:
        if input_img:
            results = rag_system.vector_search_text_img(query, input_img, k=1)
            return {"rag_context": results[0].page_content}
        else:
            results = rag_system.similarity_search_with_score(query, k=1)
            return {"rag_context": results[0][0].page_content}
