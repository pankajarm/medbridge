"""MedBridge Streamlit UI — Multilingual Clinical Trial Intelligence."""

import json
import time
from urllib.parse import quote_plus

import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MedBridge",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_system():
    """Load all models and databases (cached across reruns)."""
    from src.embeddings.harrier_embedder import HarrierEmbedder
    from src.llm.gemma_llm import GemmaLLM
    from src.storage.vector_store import VectorStore
    from src.storage.graph_store import GraphStore
    from src.graph.workflow import build_workflow

    with st.spinner("Loading Harrier embeddings (MPS GPU)..."):
        embedder = HarrierEmbedder()
    with st.spinner("Loading Gemma LLM (Metal GPU)..."):
        llm = GemmaLLM()
    vector_store = VectorStore(embedding_dim=embedder.dim)
    graph_store = GraphStore()
    workflow = build_workflow(llm, embedder, vector_store, graph_store)

    return {
        "embedder": embedder,
        "llm": llm,
        "vector_store": vector_store,
        "graph_store": graph_store,
        "workflow": workflow,
    }


def render_sidebar(system):
    """Render the sidebar with system stats and suggested queries."""
    st.sidebar.title("MedBridge")
    st.sidebar.caption("Multilingual Clinical Trial Intelligence")

    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    vs = system["vector_store"]
    gs = system["graph_store"]
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Trials", vs.count())
    col2.metric("Graph Nodes", gs.node_count())

    st.sidebar.divider()
    st.sidebar.subheader("Suggested Queries")
    suggestions = [
        "Find studies on metformin cardiovascular outcomes",
        "Drug interactions for diabetes medications",
        "Adverse events for empagliflozin in Asian vs Western populations",
        "Compare statin trials across languages",
        "Show all trials studying heart failure",
        "What are the side effects of pembrolizumab in East Asian patients?",
    ]
    for s in suggestions:
        if st.sidebar.button(s, key=f"sug_{hash(s)}", use_container_width=True):
            st.session_state["query_input"] = s
            st.session_state["auto_run"] = True
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Tech Stack")
    st.sidebar.markdown("""
    - **Embeddings**: Harrier-OSS-v1-0.6B (MPS)
    - **LLM**: Gemma 4 E2B (llama-server)
    - **Vector DB**: Qdrant (embedded)
    - **Graph DB**: FalkorDBLite (embedded)
    - **Agents**: LangGraph
    """)


def render_results(results):
    """Render search results as cards."""
    if not results:
        st.info("No results found.")
        return

    # Language flag mapping
    flags = {"en": "🇺🇸", "zh": "🇨🇳", "ja": "🇯🇵", "de": "🇩🇪", "fr": "🇫🇷", "es": "🇪🇸", "ko": "🇰🇷"}

    for r in results:
        lang = r.get("language", "?")
        flag = flags.get(lang, "🌐")
        score = r.get("score", 0)
        phase = r.get("phase", "?")

        with st.container(border=True):
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            title = r.get("title", "Untitled")
            drugs = r.get("drugs", [])
            pubmed_query = quote_plus(f"{title} {' '.join(drugs)}".strip())
            pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={pubmed_query}"
            with col1:
                st.markdown(f"**{flag} [{title}]({pubmed_url})**")
            with col2:
                st.metric("Similarity", f"{score:.3f}")
            with col3:
                st.caption(f"Phase {phase} | {r.get('country', '?')}")

            abstract = r.get("abstract", "")
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            st.caption(abstract)

            drugs = r.get("drugs", [])
            if drugs:
                st.markdown(" ".join([f"`{d}`" for d in drugs]))


def render_graph_results(results):
    """Render graph query results."""
    if not results:
        return

    st.subheader("Knowledge Graph Results")
    for r in results:
        if "error" in r:
            st.error(f"Query error: {r['error']}")
            if "cypher" in r:
                st.code(r["cypher"], language="cypher")
        else:
            st.json(r)


def render_analysis(report):
    """Render analysis report."""
    if not report:
        return

    report_type = report.get("type", "general")

    if report_type == "cross_cultural":
        st.subheader("Cross-Cultural Analysis")
        st.markdown(report.get("analysis", ""))
        if "populations" in report:
            st.caption(f"Populations analyzed: {', '.join(report['populations'])}")

    elif report_type == "comparison":
        st.subheader("Trial Comparison")
        if "similarity_matrix" in report and "labels" in report:
            _render_heatmap(report["similarity_matrix"], report["labels"])
        if "error" in report:
            st.warning(report["error"])

    elif report_type == "general":
        st.subheader("Analysis")
        st.markdown(report.get("synthesis", ""))


def _render_heatmap(matrix, labels):
    """Render a similarity matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        text=matrix,
        texttemplate="%{text:.2f}",
    ))
    fig.update_layout(
        title="Cross-Lingual Similarity Matrix",
        height=400,
        xaxis_title="Trial",
        yaxis_title="Trial",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_agent_trace(trace):
    """Render the agent execution trace."""
    if not trace:
        return

    with st.expander("Agent Trace", expanded=False):
        for i, entry in enumerate(trace):
            agent = entry.get("agent", "?")
            action = entry.get("action", "?")
            icon = {"supervisor": "🧠", "semantic_search": "🔍",
                    "graph_query": "📊", "analysis": "📈", "synthesizer": "✍️"}.get(agent, "⚙️")

            st.markdown(f"**{icon} Step {i+1}: {agent}** → `{action}`")
            details = {k: v for k, v in entry.items() if k not in ("agent", "action")}
            if details:
                st.json(details)


def render_drug_graph(graph_store):
    """Render the drug interaction network."""
    try:
        from streamlit_agraph import agraph, Node, Edge, Config

        results = graph_store.query(
            """MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
            RETURN d1.name AS drug1, d2.name AS drug2, r.type AS type, r.severity AS severity"""
        )

        if not results:
            st.info("No drug interactions in graph yet.")
            return

        nodes_set = set()
        nodes = []
        edges = []

        colors = {"low": "#4CAF50", "moderate": "#FF9800", "high": "#F44336"}

        for r in results:
            d1, d2 = r["drug1"], r["drug2"]
            if d1 not in nodes_set:
                nodes.append(Node(id=d1, label=d1, size=25, color="#2196F3"))
                nodes_set.add(d1)
            if d2 not in nodes_set:
                nodes.append(Node(id=d2, label=d2, size=25, color="#2196F3"))
                nodes_set.add(d2)
            edges.append(Edge(
                source=d1, target=d2,
                label=r.get("type", ""),
                color=colors.get(r.get("severity", ""), "#999"),
            ))

        config = Config(
            width=700, height=400, directed=True,
            physics=True, hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
        )
        agraph(nodes=nodes, edges=edges, config=config)

    except ImportError:
        st.warning("Install streamlit-agraph for interactive graph visualization.")
    except Exception as e:
        st.error(f"Graph rendering error: {e}")


def main():
    # Load system
    system = load_system()
    render_sidebar(system)

    # Main content
    st.title("MedBridge")
    st.caption("Multilingual Clinical Trial Intelligence powered by Microsoft Harrier Embeddings, Google Gemma 4, Qdrant Vector DB, FalkorDBLite Graph DB, and LangGraph Agents")

    # Tabs
    tab_search, tab_graph, tab_dashboard = st.tabs(["Search & Analysis", "Drug Interaction Graph", "Dashboard"])

    with tab_search:
        # Query input
        query = st.text_input(
            "Ask about clinical trials in any language...",
            value=st.session_state.get("query_input", ""),
            key="query_box",
            placeholder="e.g., Find studies on metformin cardiovascular outcomes",
        )

        auto_run = st.session_state.pop("auto_run", False)
        if (st.button("Search", type="primary", use_container_width=True) or auto_run) and query:
            st.session_state["query_input"] = ""

            with st.status("Running multi-agent pipeline...", expanded=True) as status:
                start = time.time()

                st.write("Classifying query intent...")
                result = system["workflow"].invoke({
                    "query": query,
                    "messages": [],
                    "query_type": "literature_search",
                    "search_results": [],
                    "graph_results": [],
                    "analysis_report": {},
                    "agent_trace": [],
                    "final_response": "",
                })

                elapsed = time.time() - start
                status.update(label=f"Complete in {elapsed:.1f}s", state="complete")

            # Store results in session
            st.session_state["last_result"] = result

        # Display results
        result = st.session_state.get("last_result")
        if result:
            # Final response
            st.markdown("---")
            st.markdown(result.get("final_response", ""))

            # Search results
            if result.get("search_results"):
                st.markdown("---")
                st.subheader(f"Retrieved Trials ({len(result['search_results'])} results)")
                render_results(result["search_results"])

            # Graph results
            if result.get("graph_results"):
                render_graph_results(result["graph_results"])

            # Analysis
            if result.get("analysis_report"):
                st.markdown("---")
                render_analysis(result["analysis_report"])

            # Agent trace
            if result.get("agent_trace"):
                st.markdown("---")
                render_agent_trace(result["agent_trace"])

    with tab_graph:
        st.subheader("Drug Interaction Network")
        st.caption("Relationships extracted from clinical trial data across languages")
        render_drug_graph(system["graph_store"])

    with tab_dashboard:
        st.subheader("Research Dashboard")

        col1, col2, col3, col4 = st.columns(4)
        vs = system["vector_store"]
        gs = system["graph_store"]

        col1.metric("Total Trials", vs.count())
        col2.metric("Graph Nodes", gs.node_count())
        col3.metric("Relationships", gs.relationship_count())
        col4.metric("Languages", len(["en", "zh", "ja", "de", "fr", "es", "ko"]))

        # Language distribution
        st.subheader("Trials by Language")
        lang_data = gs.query(
            """MATCH (t:ClinicalTrial)
            RETURN t.language AS language, count(t) AS count
            ORDER BY count DESC"""
        )
        if lang_data:
            flags = {"en": "English 🇺🇸", "zh": "Chinese 🇨🇳", "ja": "Japanese 🇯🇵",
                     "de": "German 🇩🇪", "fr": "French 🇫🇷", "es": "Spanish 🇪🇸", "ko": "Korean 🇰🇷"}
            fig = go.Figure(data=[go.Bar(
                x=[flags.get(d["language"], d["language"]) for d in lang_data],
                y=[d["count"] for d in lang_data],
                marker_color=["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01", "#46BDC6", "#7B1FA2"],
            )])
            fig.update_layout(height=300, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        # Cross-lingual similarities
        st.subheader("Cross-Lingual Trial Similarities")
        sim_data = gs.query(
            """MATCH (t1:ClinicalTrial)-[s:SIMILAR_TO]->(t2:ClinicalTrial)
            WHERE s.cross_lingual = true
            RETURN t1.id AS trial1, t1.language AS lang1,
                   t2.id AS trial2, t2.language AS lang2,
                   s.score AS score
            ORDER BY s.score DESC
            LIMIT 10"""
        )
        if sim_data:
            for s in sim_data:
                flags_sm = {"en": "🇺🇸", "zh": "🇨🇳", "ja": "🇯🇵", "de": "🇩🇪", "fr": "🇫🇷", "es": "🇪🇸", "ko": "🇰🇷"}
                f1 = flags_sm.get(s.get("lang1", ""), "🌐")
                f2 = flags_sm.get(s.get("lang2", ""), "🌐")
                score = s.get("score", 0)
                st.markdown(
                    f"{f1} `{s.get('trial1', '?')}` ↔ {f2} `{s.get('trial2', '?')}` — "
                    f"similarity: **{score:.3f}**"
                )
        else:
            st.info("No cross-lingual similarities computed yet.")


if __name__ == "__main__":
    main()
