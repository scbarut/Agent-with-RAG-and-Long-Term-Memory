"""Shared LLM instance used across all agent nodes."""

from langchain_google_genai import ChatGoogleGenerativeAI

# Module-level singleton — imported by every node that needs an LLM call.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
