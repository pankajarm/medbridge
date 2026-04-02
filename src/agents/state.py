"""LangGraph shared state for MedBridge multi-agent system."""

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class MedBridgeState(TypedDict):
    """Shared state across all agents in the MedBridge workflow."""
    messages: Annotated[list, add_messages]
    query: str
    query_type: Literal[
        "literature_search",
        "drug_interaction",
        "adverse_event",
        "cross_cultural_analysis",
        "trial_comparison",
        "ingestion",
    ]
    search_results: list[dict]
    graph_results: list[dict]
    analysis_report: dict
    agent_trace: list[dict]
    final_response: str
