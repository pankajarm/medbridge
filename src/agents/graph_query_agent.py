"""Graph query agent: NL-to-Cypher for structural knowledge queries."""

from src.agents.state import MedBridgeState


def graph_query_node(state: MedBridgeState, llm, graph_store) -> dict:
    """Generate and execute Cypher queries against FalkorDBLite."""
    query = state["query"]
    query_type = state.get("query_type", "drug_interaction")

    # For drug interactions, use a pre-built query pattern
    if query_type == "drug_interaction":
        # Extract drug name from query (simple approach)
        results = _drug_interaction_query(query, graph_store, llm)
    else:
        # Generate Cypher from natural language
        schema = graph_store.get_schema()
        cypher = llm.generate_cypher(query, schema)
        try:
            results = graph_store.query(cypher)
        except Exception as e:
            results = [{"error": str(e), "cypher": cypher}]

    trace_entry = {
        "agent": "graph_query",
        "action": "cypher_query",
        "results_count": len(results),
    }

    return {
        "graph_results": results,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }


def _drug_interaction_query(query: str, graph_store, llm) -> list[dict]:
    """Handle drug interaction queries with targeted Cypher."""
    # Use LLM to extract drug name
    drug_name = llm.generate(
        f"Extract the main drug name from this query. Return ONLY the drug name in English, nothing else: {query}",
        system_prompt="You extract drug names from text. Return only the drug name.",
        max_tokens=30,
        temperature=0.0,
    ).strip()

    results = graph_store.get_drug_interactions(drug_name)
    if not results:
        # Try broader query
        results = graph_store.query(
            """MATCH (d:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
            RETURN d.name AS drug1, d2.name AS drug2, r.type AS type, r.severity AS severity
            LIMIT 20"""
        )
    return results
