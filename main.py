"""MedBridge: Multilingual Clinical Trial Intelligence System.

Usage:
    uv run python main.py          # CLI mode
    uv run streamlit run src/ui/app.py  # Web UI mode
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings.harrier_embedder import HarrierEmbedder
from src.llm.gemma_llm import GemmaLLM
from src.storage.vector_store import VectorStore
from src.storage.graph_store import GraphStore
from src.graph.workflow import build_workflow


def main():
    print("=" * 60)
    print("  MedBridge: Multilingual Clinical Trial Intelligence")
    print("  Powered by Microsoft Harrier-OSS-v1 Embeddings")
    print("=" * 60)

    # Initialize components
    print("\nLoading models...")
    embedder = HarrierEmbedder()
    llm = GemmaLLM()

    print("Connecting to databases...")
    vector_store = VectorStore(embedding_dim=embedder.dim)
    graph_store = GraphStore()

    print(f"  Qdrant: {vector_store.count()} vectors")
    print(f"  FalkorDB: {graph_store.node_count()} nodes, {graph_store.relationship_count()} relationships")

    if vector_store.count() == 0:
        print("\nNo data found. Run setup first:")
        print("  uv run python scripts/setup_databases.py")
        return

    # Build workflow
    print("\nBuilding LangGraph workflow...")
    workflow = build_workflow(llm, embedder, vector_store, graph_store)
    print("Ready!\n")

    # Interactive CLI
    print("Enter your query (or 'quit' to exit):")
    print("Example: Find studies on metformin cardiovascular outcomes\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nRunning multi-agent pipeline...")
        result = workflow.invoke({
            "query": query,
            "messages": [],
            "query_type": "literature_search",
            "search_results": [],
            "graph_results": [],
            "analysis_report": {},
            "agent_trace": [],
            "final_response": "",
        })

        # Print response
        print("\n" + "=" * 60)
        print(result.get("final_response", "No response generated."))
        print("=" * 60)

        # Print trace
        trace = result.get("agent_trace", [])
        if trace:
            print(f"\nAgent trace ({len(trace)} steps):")
            for i, step in enumerate(trace):
                print(f"  {i+1}. {step.get('agent', '?')} -> {step.get('action', '?')}")

        # Print search results summary
        results = result.get("search_results", [])
        if results:
            langs = {}
            for r in results:
                lang = r.get("language", "?")
                langs[lang] = langs.get(lang, 0) + 1
            print(f"\nResults: {len(results)} trials across languages: {dict(langs)}")

        print()


if __name__ == "__main__":
    main()
