"""LangGraph StateGraph workflow for MedBridge multi-agent system."""

from functools import partial

from langgraph.graph import END, StateGraph

from src.agents.state import MedBridgeState
from src.agents.supervisor import supervisor_node
from src.agents.semantic_search_agent import semantic_search_node
from src.agents.graph_query_agent import graph_query_node
from src.agents.analysis_agent import analysis_node


def _synthesize(state: MedBridgeState, llm) -> dict:
    """Synthesize final response from all agent outputs."""
    query = state["query"]
    search_results = state.get("search_results", [])
    graph_results = state.get("graph_results", [])
    analysis_report = state.get("analysis_report", {})

    # Build context
    parts = [f"User query: {query}\n"]

    if search_results:
        parts.append(f"Found {len(search_results)} relevant trials across languages:")
        langs = {}
        for r in search_results[:10]:
            lang = r.get("language", "?")
            langs[lang] = langs.get(lang, 0) + 1
            parts.append(f"  [{lang}] {r.get('title', 'Untitled')} (similarity: {r.get('score', 0):.3f})")
        parts.append(f"Languages represented: {dict(langs)}\n")

    if graph_results:
        parts.append(f"Knowledge graph results ({len(graph_results)} entries):")
        for g in graph_results[:8]:
            parts.append(f"  {g}")
        parts.append("")

    if analysis_report:
        report_type = analysis_report.get("type", "general")
        if report_type == "cross_cultural":
            parts.append(f"Cross-cultural analysis ({analysis_report.get('num_trials', 0)} trials):")
            parts.append(analysis_report.get("analysis", ""))
        elif report_type == "comparison":
            parts.append(f"Trial comparison ({analysis_report.get('num_compared', 0)} trials compared)")
            if "similarity_matrix" in analysis_report:
                parts.append(f"Labels: {analysis_report.get('labels', [])}")
        elif "synthesis" in analysis_report:
            parts.append(analysis_report["synthesis"])

    context = "\n".join(parts)

    response = llm.generate(
        f"Based on the following multi-agent analysis results, provide a clear, comprehensive response to the user.\n\n{context}",
        system_prompt="You are MedBridge, a multilingual clinical trial intelligence system. Provide clear, evidence-based responses. Highlight cross-lingual findings and population-specific insights.",
        max_tokens=1000,
    )

    trace_entry = {
        "agent": "synthesizer",
        "action": "synthesize",
        "input_sources": [],
    }
    if search_results:
        trace_entry["input_sources"].append(f"vector_search ({len(search_results)} results)")
    if graph_results:
        trace_entry["input_sources"].append(f"graph_query ({len(graph_results)} results)")
    if analysis_report:
        trace_entry["input_sources"].append(f"analysis ({analysis_report.get('type', 'general')})")

    return {
        "final_response": response,
        "agent_trace": state.get("agent_trace", []) + [trace_entry],
    }


def _route_by_query_type(state: MedBridgeState) -> str:
    """Route from supervisor to the appropriate specialist agent."""
    qt = state.get("query_type", "literature_search")
    routing = {
        "literature_search": "semantic_search",
        "trial_comparison": "semantic_search",
        "cross_cultural_analysis": "semantic_search",
        "adverse_event": "semantic_search",
        "drug_interaction": "graph_query",
    }
    return routing.get(qt, "semantic_search")


def _post_search_route(state: MedBridgeState) -> str:
    """After semantic search, decide next step."""
    qt = state.get("query_type", "literature_search")
    if qt in ("cross_cultural_analysis", "adverse_event"):
        return "graph_query"
    if qt == "trial_comparison":
        return "analysis"
    return "synthesizer"


def _post_graph_route(state: MedBridgeState) -> str:
    """After graph query, decide next step."""
    qt = state.get("query_type", "drug_interaction")
    if qt in ("cross_cultural_analysis", "adverse_event"):
        return "analysis"
    return "synthesizer"


def build_workflow(llm, embedder, vector_store, graph_store):
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(MedBridgeState)

    # Add nodes with bound dependencies
    workflow.add_node("supervisor", partial(supervisor_node, llm=llm))
    workflow.add_node("semantic_search", partial(semantic_search_node, embedder=embedder, vector_store=vector_store))
    workflow.add_node("graph_query", partial(graph_query_node, llm=llm, graph_store=graph_store))
    workflow.add_node("analysis", partial(analysis_node, llm=llm, embedder=embedder))
    workflow.add_node("synthesizer", partial(_synthesize, llm=llm))

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routes based on query type
    workflow.add_conditional_edges("supervisor", _route_by_query_type, {
        "semantic_search": "semantic_search",
        "graph_query": "graph_query",
    })

    # After search: analyze, enrich with graph, or synthesize
    workflow.add_conditional_edges("semantic_search", _post_search_route, {
        "graph_query": "graph_query",
        "analysis": "analysis",
        "synthesizer": "synthesizer",
    })

    # After graph query: analyze or synthesize
    workflow.add_conditional_edges("graph_query", _post_graph_route, {
        "analysis": "analysis",
        "synthesizer": "synthesizer",
    })

    # Analysis always goes to synthesizer
    workflow.add_edge("analysis", "synthesizer")

    # Synthesizer ends
    workflow.add_edge("synthesizer", END)

    return workflow.compile()
