"""Analysis agent: cross-cultural comparison, drug interactions, evidence synthesis."""

from src.agents.state import MedBridgeState


def analysis_node(state: MedBridgeState, llm, embedder) -> dict:
    """Analyze search and graph results for cross-cultural insights."""
    query = state["query"]
    query_type = state.get("query_type", "cross_cultural_analysis")
    search_results = state.get("search_results", [])
    graph_results = state.get("graph_results", [])

    if query_type == "cross_cultural_analysis" or query_type == "adverse_event":
        report = _cross_cultural_analysis(search_results, graph_results, query, llm)
    elif query_type == "trial_comparison":
        report = _trial_comparison(search_results, query, llm, embedder)
    else:
        report = _general_analysis(search_results, graph_results, query, llm)

    trace_entry = {
        "agent": "analysis",
        "action": query_type,
        "report_sections": list(report.keys()),
    }

    return {
        "analysis_report": report,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }


def _cross_cultural_analysis(search_results, graph_results, query, llm) -> dict:
    """Analyze adverse events across populations."""
    # Group results by population/country
    by_population = {}
    for r in search_results:
        pop = r.get("country", "Unknown")
        lang = r.get("language", "?")
        key = f"{pop} ({lang})"
        by_population.setdefault(key, []).append(r)

    # Build context for LLM analysis
    context = "Clinical trial results by population:\n\n"
    for pop, trials in by_population.items():
        context += f"## {pop}\n"
        for t in trials[:3]:
            context += f"- {t.get('title', 'Untitled')} (score: {t.get('score', 0):.3f})\n"
            context += f"  Abstract: {t.get('abstract', '')[:200]}...\n"
        context += "\n"

    if graph_results:
        context += "\nGraph data (adverse events):\n"
        for g in graph_results[:10]:
            context += f"- {g}\n"

    analysis = llm.generate(
        f"Query: {query}\n\n{context}\n\nProvide a cross-cultural analysis focusing on:\n1. Population-specific differences in adverse events\n2. Efficacy variations across populations\n3. Key safety signals\n4. Recommendations",
        system_prompt="You are a clinical research analyst specializing in cross-cultural pharmacovigilance.",
        max_tokens=800,
    )

    return {
        "type": "cross_cultural",
        "populations": list(by_population.keys()),
        "num_trials": len(search_results),
        "analysis": analysis,
        "data": by_population,
    }


def _trial_comparison(search_results, query, llm, embedder) -> dict:
    """Compare trials using embedding similarity matrix."""
    if len(search_results) < 2:
        return {"type": "comparison", "error": "Need at least 2 trials to compare"}

    # Compute pairwise similarity for top results
    top_results = search_results[:6]
    abstracts = [r.get("abstract", "") for r in top_results]
    embeddings = embedder.embed_documents(abstracts)

    similarity_matrix = []
    for i, emb_i in enumerate(embeddings):
        row = []
        for j, emb_j in enumerate(embeddings):
            sim = sum(a * b for a, b in zip(emb_i, emb_j))
            row.append(round(sim, 3))
        similarity_matrix.append(row)

    labels = [
        f"{r.get('trial_id', '?')} ({r.get('language', '?')})"
        for r in top_results
    ]

    return {
        "type": "comparison",
        "labels": labels,
        "similarity_matrix": similarity_matrix,
        "num_compared": len(top_results),
    }


def _general_analysis(search_results, graph_results, query, llm) -> dict:
    """General analysis and synthesis."""
    context = f"Search results ({len(search_results)} trials):\n"
    for r in search_results[:5]:
        context += f"- [{r.get('language', '?')}] {r.get('title', 'Untitled')} (score: {r.get('score', 0):.3f})\n"

    if graph_results:
        context += f"\nGraph results ({len(graph_results)} entries):\n"
        for g in graph_results[:5]:
            context += f"- {g}\n"

    synthesis = llm.generate(
        f"Query: {query}\n\n{context}\n\nSynthesize the findings.",
        system_prompt="You are a medical research analyst.",
        max_tokens=600,
    )

    return {
        "type": "general",
        "synthesis": synthesis,
    }
