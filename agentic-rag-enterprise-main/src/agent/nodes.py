"""
Agent Nodes — individual steps in the agentic RAG state machine.
Each node is a function that takes the current state and returns updated state.
"""
import json
import time
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from .prompts import RETRIEVAL_GRADER_PROMPT, ANSWER_GENERATOR_PROMPT, QUERY_REWRITER_PROMPT


class AgentState(TypedDict):
    """State that flows through the agent graph."""
    question: str
    documents: list[dict]
    relevant_documents: list[dict]
    generation: str
    retry_count: int
    max_retries: int
    relevance_threshold: float
    latency: dict


def retrieve(state: AgentState, retriever) -> AgentState:
    """Retrieve documents using hybrid search."""
    start = time.time()
    question = state["question"]
    documents = retriever.retrieve(question)
    elapsed = time.time() - start

    latency = state.get("latency", {})
    latency["retrieval"] = elapsed

    return {
        **state,
        "documents": documents,
        "latency": latency,
    }


def grade_documents(state: AgentState, llm) -> AgentState:
    """Grade each retrieved document for relevance."""
    start = time.time()
    question = state["question"]
    documents = state["documents"]

    relevant = []
    for doc in documents:
        prompt = RETRIEVAL_GRADER_PROMPT.format(
            question=question,
            document=doc["text"][:1000],  # truncate for grading
        )

        response = llm.invoke([HumanMessage(content=prompt)])

        try:
            result = json.loads(response.content)
            if result.get("score") == "relevant":
                doc["relevance_reason"] = result.get("reason", "")
                relevant.append(doc)
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, include the document (err on side of inclusion)
            relevant.append(doc)

    elapsed = time.time() - start
    latency = state.get("latency", {})
    latency["grading"] = elapsed

    return {
        **state,
        "relevant_documents": relevant,
        "latency": latency,
    }


def should_requery(state: AgentState) -> str:
    """Decision node: should we rewrite the query and try again?"""
    relevant = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    # If we have enough relevant docs, generate answer
    if len(relevant) >= 2:
        return "generate"

    # If we've exhausted retries, generate with what we have (or refuse)
    if retry_count >= max_retries:
        if len(relevant) == 0:
            return "refuse"
        return "generate"

    # Otherwise, rewrite and retry
    return "requery"


def rewrite_query(state: AgentState, llm) -> AgentState:
    """Rewrite the query to improve retrieval."""
    start = time.time()

    prompt = QUERY_REWRITER_PROMPT.format(
        question=state["question"],
        num_relevant=len(state.get("relevant_documents", [])),
        num_total=len(state.get("documents", [])),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    new_question = response.content.strip()

    elapsed = time.time() - start
    latency = state.get("latency", {})
    latency[f"rewrite_{state.get('retry_count', 0)}"] = elapsed

    return {
        **state,
        "question": new_question,
        "retry_count": state.get("retry_count", 0) + 1,
        "latency": latency,
    }


def generate_answer(state: AgentState, llm) -> AgentState:
    """Generate a cited answer from relevant documents."""
    start = time.time()

    relevant_docs = state.get("relevant_documents", [])

    # Format context with page numbers
    context_parts = []
    for i, doc in enumerate(relevant_docs):
        page = doc.get("metadata", {}).get("page_number", "?")
        section = doc.get("metadata", {}).get("section", "")
        context_parts.append(
            f"[Chunk {i+1} | Page {page} | Section: {section}]\n{doc['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = ANSWER_GENERATOR_PROMPT.format(
        context=context,
        question=state["question"],
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    elapsed = time.time() - start
    latency = state.get("latency", {})
    latency["generation"] = elapsed

    return {
        **state,
        "generation": response.content,
        "latency": latency,
    }


def refuse_answer(state: AgentState) -> AgentState:
    """Gracefully refuse when no relevant information is found."""
    return {
        **state,
        "generation": (
            "I could not find sufficient information in the available documents "
            "to answer this question. The retrieval system was unable to locate "
            "relevant content even after query reformulation. Please try rephrasing "
            "your question or check if the relevant documents have been ingested."
        ),
    }
