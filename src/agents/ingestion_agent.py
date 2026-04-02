"""Ingestion agent: processes clinical trial data into vector + graph stores."""

import json
from pathlib import Path

from src.config import DATA_DIR


def ingest_trials(embedder, vector_store, graph_store) -> dict:
    """Ingest all sample trials into Qdrant and FalkorDBLite."""
    trials_dir = DATA_DIR / "sample_trials"
    aliases_path = trials_dir / "_drug_aliases.json"
    interactions_path = trials_dir / "_drug_interactions.json"

    # Load trial files
    trial_files = sorted(trials_dir.glob("NCT-*.json"))
    if not trial_files:
        return {"status": "error", "message": "No trial files found. Run: uv run python scripts/generate_sample_trials.py"}

    print(f"Ingesting {len(trial_files)} trials...")
    trials = []
    for fp in trial_files:
        with open(fp, encoding="utf-8") as f:
            trials.append(json.load(f))

    # 1. Embed all abstracts in batch
    print("Embedding abstracts with Harrier...")
    abstracts = [t["abstract"] for t in trials]
    embeddings = embedder.embed_documents(abstracts)

    # 2. Upsert to Qdrant
    print("Upserting to Qdrant...")
    items = []
    for trial, embedding in zip(trials, embeddings):
        items.append({
            "id": trial["id"],
            "vector": embedding,
            "metadata": {
                "title": trial["title"],
                "abstract": trial["abstract"],
                "language": trial["language"],
                "phase": trial["phase"],
                "country": trial["country"],
                "drugs": trial.get("drugs", []),
                "enrollment": trial.get("enrollment", 0),
                "year": trial.get("year", 2023),
            },
        })
    vector_store.upsert_batch(items)

    # 3. Build graph in FalkorDBLite
    print("Building knowledge graph...")
    all_drugs = set()
    all_diseases = set()

    for trial in trials:
        # Add trial node
        graph_store.add_trial({
            "id": trial["id"],
            "title": trial["title"],
            "abstract": trial["abstract"][:500],
            "language": trial["language"],
            "phase": trial["phase"],
            "country": trial["country"],
            "enrollment": trial.get("enrollment", 0),
            "year": trial.get("year", 2023),
        })

        # Add drug nodes and relationships
        for drug in trial.get("drugs", []):
            all_drugs.add(drug)
            graph_store.add_drug(drug)
            graph_store.link_trial_drug(trial["id"], drug)

        # Add disease nodes and relationships
        for disease in trial.get("diseases", []):
            all_diseases.add(disease)
            graph_store.add_disease(disease)
            graph_store.link_trial_disease(trial["id"], disease)

        # Add adverse events
        for ae in trial.get("adverse_events", []):
            graph_store.add_adverse_event(ae["name"], frequency=str(ae.get("rate", "")))
            graph_store.link_trial_adverse_event(
                trial["id"], ae["name"],
                population=ae.get("population", ""),
                rate=ae.get("rate", 0.0),
            )

    # 4. Add drug aliases (cross-lingual entity resolution)
    if aliases_path.exists():
        with open(aliases_path, encoding="utf-8") as f:
            aliases = json.load(f)
        for canonical, local_names in aliases.items():
            for local_name in local_names:
                if local_name in all_drugs:
                    graph_store.link_drug_alias(canonical, local_name)

    # 5. Add drug interactions
    if interactions_path.exists():
        with open(interactions_path, encoding="utf-8") as f:
            interactions = json.load(f)
        for interaction in interactions:
            graph_store.link_drug_interaction(
                interaction["drug1"], interaction["drug2"],
                interaction.get("type", ""),
                interaction.get("severity", ""),
            )

    # 6. Compute cross-lingual similarity edges
    print("Computing cross-lingual similarity edges...")
    _compute_similarity_edges(trials, embeddings, graph_store)

    stats = {
        "status": "success",
        "trials_ingested": len(trials),
        "vectors_count": vector_store.count(),
        "graph_nodes": graph_store.node_count(),
        "graph_relationships": graph_store.relationship_count(),
    }
    print(f"Ingestion complete: {stats}")
    return stats


def _compute_similarity_edges(trials, embeddings, graph_store, threshold=0.75):
    """Create SIMILAR_TO edges for trials with high cross-lingual similarity."""
    for i in range(len(trials)):
        for j in range(i + 1, len(trials)):
            # Compute cosine similarity (embeddings are already L2-normalized)
            sim = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            if sim >= threshold:
                cross_lingual = trials[i]["language"] != trials[j]["language"]
                graph_store.link_similar_trials(
                    trials[i]["id"], trials[j]["id"],
                    score=round(sim, 4),
                    cross_lingual=cross_lingual,
                )
