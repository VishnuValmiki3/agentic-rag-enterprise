"""
Agent Graph — LangGraph state machine for agentic RAG.

Flow:
  retrieve → grade → (generate | requery → retrieve → grade → ... | refuse)

This is the core differentiator of the project: instead of blindly generating
from whatever the retriever returns, the agent self-corrects.
"""
import os
import time
from functools import partial

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from .nodes import (
    AgentState,
    retrieve,
    grade_documents,
    should_requery,
    rewrite_query,
    generate_answer,
    refuse_answer,
)
from ..retrieval.hybrid import HybridRetriever

load_dotenv()


def get_llm():
    """Get the configured LLM."""
    provider = os.getenv("LLM_PROVIDER", "anthropic")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            temperature=0.1,
            max_tokens=1024,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.1,
            max_tokens=1024,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def build_agent(retriever: HybridRetriever, max_retries: int = 2) -> StateGraph:
    """Build the agentic RAG graph."""
    llm = get_llm()

    # Create partial functions with dependencies injected
    retrieve_node = partial(retrieve, retriever=retriever)
    grade_node = partial(grade_documents, llm=llm)
    rewrite_node = partial(rewrite_query, llm=llm)
    generate_node = partial(generate_answer, llm=llm)

    # Define the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("refuse", refuse_answer)

    # Define edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")

    # Conditional edge after grading
    workflow.add_conditional_edges(
        "grade",
        should_requery,
        {
            "generate": "generate",
            "requery": "rewrite",
            "refuse": "refuse",
        },
    )

    # Rewrite loops back to retrieve
    workflow.add_edge("rewrite", "retrieve")

    # Terminal nodes
    workflow.add_edge("generate", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()


def query(
    agent,
    question: str,
    max_retries: int = 2,
) -> dict:
    """Run a query through the agent and return structured results."""
    start = time.time()

    initial_state = {
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "generation": "",
        "retry_count": 0,
        "max_retries": max_retries,
        "relevance_threshold": 0.6,
        "latency": {},
    }

    # Run the agent
    result = agent.invoke(initial_state)

    total_time = time.time() - start

    return {
        "question": question,
        "answer": result["generation"],
        "sources": [
            {
                "text": doc["text"][:500],
                "page": doc.get("metadata", {}).get("page_number", "?"),
                "section": doc.get("metadata", {}).get("section", ""),
                "score": doc.get("rerank_score", doc.get("rrf_score", 0)),
            }
            for doc in result.get("relevant_documents", [])
        ],
        "retries": result.get("retry_count", 0),
        "latency": {**result.get("latency", {}), "total": total_time},
    }
