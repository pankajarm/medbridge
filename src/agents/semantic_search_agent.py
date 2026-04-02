"""Semantic search agent: cross-lingual trial retrieval via Harrier embeddings."""

from src.agents.state import MedBridgeState


def semantic_search_node(state: MedBridgeState, embedder, vector_store) -> dict:
    """Search for relevant trials using Harrier multilingual embeddings."""
    query = state["query"]
    query_type = state.get("query_type", "literature_search")

    # Choose task-specific embedding prompt
    task_map = {
        "literature_search": "trial_retrieval",
        "drug_interaction": "trial_retrieval",
        "adverse_event": "adverse_event",
        "cross_cultural_analysis": "trial_retrieval",
        "trial_comparison": "trial_similarity",
    }
    task = task_map.get(query_type, "trial_retrieval")

    # Embed query with Harrier (cross-lingual!)
    query_vector = embedder.embed_query(query, task=task)

    # Search Qdrant — results span ALL languages automatically
    results = vector_store.search(query_vector=query_vector, limit=15)

    trace_entry = {
        "agent": "semantic_search",
        "action": "vector_search",
        "task_prompt": task,
        "results_count": len(results),
        "languages_found": list({r.get("language", "?") for r in results}),
    }

    return {
        "search_results": results,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }
