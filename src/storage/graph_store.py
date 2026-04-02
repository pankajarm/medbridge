"""FalkorDBLite embedded graph store for clinical trial knowledge graph."""

from redislite import FalkorDB

from src.config import FALKORDB_PATH


GRAPH_SCHEMA = """
Node types:
- ClinicalTrial: id, title, abstract, language, phase, country, enrollment, year
- Drug: id, name, canonical_name, drug_class
- Disease: id, name, canonical_name, category
- AdverseEvent: id, name, severity, frequency
- Institution: id, name, country

Relationships:
- (ClinicalTrial)-[:STUDIES]->(Disease)
- (ClinicalTrial)-[:USES_DRUG {dosage, duration}]->(Drug)
- (ClinicalTrial)-[:REPORTS {population, rate}]->(AdverseEvent)
- (ClinicalTrial)-[:CONDUCTED_BY]->(Institution)
- (Drug)-[:INTERACTS_WITH {type, severity}]->(Drug)
- (Drug)-[:TREATS]->(Disease)
- (ClinicalTrial)-[:SIMILAR_TO {score, cross_lingual}]->(ClinicalTrial)
- (Drug)-[:ALSO_KNOWN_AS]->(Drug)
"""


class GraphStore:
    """FalkorDBLite embedded graph store — no server needed."""

    def __init__(self, path: str | None = None):
        import os
        db_dir = path or FALKORDB_PATH
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, "medbridge.db")
        self.db = FalkorDB(db_file)
        self.graph = self.db.select_graph("medbridge")
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create indexes for fast lookups."""
        try:
            self.graph.query("CREATE INDEX FOR (t:ClinicalTrial) ON (t.id)")
            self.graph.query("CREATE INDEX FOR (d:Drug) ON (d.name)")
            self.graph.query("CREATE INDEX FOR (d:Disease) ON (d.name)")
            self.graph.query("CREATE INDEX FOR (a:AdverseEvent) ON (a.name)")
        except Exception:
            pass  # Indexes may already exist

    def add_trial(self, trial: dict):
        """Add a clinical trial node."""
        self.graph.query(
            """CREATE (t:ClinicalTrial {
                id: $id, title: $title, abstract: $abstract,
                language: $language, phase: $phase, country: $country,
                enrollment: $enrollment, year: $year
            })""",
            params=trial,
        )

    def add_drug(self, name: str, canonical_name: str = "", drug_class: str = ""):
        """Add a drug node (skip if exists)."""
        self.graph.query(
            """MERGE (d:Drug {name: $name})
            ON CREATE SET d.canonical_name = $canonical, d.drug_class = $drug_class""",
            params={"name": name, "canonical": canonical_name or name, "drug_class": drug_class},
        )

    def add_disease(self, name: str, canonical_name: str = "", category: str = ""):
        """Add a disease node (skip if exists)."""
        self.graph.query(
            """MERGE (d:Disease {name: $name})
            ON CREATE SET d.canonical_name = $canonical, d.category = $category""",
            params={"name": name, "canonical": canonical_name or name, "category": category},
        )

    def add_adverse_event(self, name: str, severity: str = "", frequency: str = ""):
        """Add an adverse event node."""
        self.graph.query(
            """MERGE (a:AdverseEvent {name: $name})
            ON CREATE SET a.severity = $severity, a.frequency = $frequency""",
            params={"name": name, "severity": severity, "frequency": frequency},
        )

    def link_trial_drug(self, trial_id: str, drug_name: str):
        """Create USES_DRUG relationship."""
        self.graph.query(
            """MATCH (t:ClinicalTrial {id: $tid}), (d:Drug {name: $drug})
            MERGE (t)-[:USES_DRUG]->(d)""",
            params={"tid": trial_id, "drug": drug_name},
        )

    def link_trial_disease(self, trial_id: str, disease_name: str):
        """Create STUDIES relationship."""
        self.graph.query(
            """MATCH (t:ClinicalTrial {id: $tid}), (d:Disease {name: $name})
            MERGE (t)-[:STUDIES]->(d)""",
            params={"tid": trial_id, "name": disease_name},
        )

    def link_trial_adverse_event(self, trial_id: str, ae_name: str, population: str = "", rate: float = 0.0):
        """Create REPORTS relationship."""
        self.graph.query(
            """MATCH (t:ClinicalTrial {id: $tid}), (a:AdverseEvent {name: $ae})
            MERGE (t)-[:REPORTS {population: $pop, rate: $rate}]->(a)""",
            params={"tid": trial_id, "ae": ae_name, "pop": population, "rate": rate},
        )

    def link_drug_interaction(self, drug1: str, drug2: str, interaction_type: str = "", severity: str = ""):
        """Create INTERACTS_WITH relationship between two drugs."""
        self.graph.query(
            """MATCH (d1:Drug {name: $d1}), (d2:Drug {name: $d2})
            MERGE (d1)-[:INTERACTS_WITH {type: $type, severity: $severity}]->(d2)""",
            params={"d1": drug1, "d2": drug2, "type": interaction_type, "severity": severity},
        )

    def link_drug_alias(self, drug1: str, drug2: str):
        """Create ALSO_KNOWN_AS for cross-lingual entity resolution."""
        self.graph.query(
            """MATCH (d1:Drug {name: $d1}), (d2:Drug {name: $d2})
            MERGE (d1)-[:ALSO_KNOWN_AS]->(d2)""",
            params={"d1": drug1, "d2": drug2},
        )

    def link_similar_trials(self, trial1_id: str, trial2_id: str, score: float, cross_lingual: bool = False):
        """Create SIMILAR_TO relationship between trials."""
        self.graph.query(
            """MATCH (t1:ClinicalTrial {id: $t1}), (t2:ClinicalTrial {id: $t2})
            MERGE (t1)-[:SIMILAR_TO {score: $score, cross_lingual: $xl}]->(t2)""",
            params={"t1": trial1_id, "t2": trial2_id, "score": score, "xl": cross_lingual},
        )

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return results as list of dicts."""
        result = self.graph.query(cypher, params=params or {})
        if not result.result_set:
            return []
        headers = result.header
        return [
            {h[1]: row[i] for i, h in enumerate(headers)}
            for row in result.result_set
        ]

    def get_drug_interactions(self, drug_name: str) -> list[dict]:
        """Get all drug interactions for a given drug."""
        return self.query(
            """MATCH (d:Drug {name: $name})-[r:INTERACTS_WITH]-(d2:Drug)
            RETURN d.name AS drug1, d2.name AS drug2, r.type AS type, r.severity AS severity""",
            params={"name": drug_name},
        )

    def get_trial_graph(self, trial_id: str) -> list[dict]:
        """Get the full subgraph for a trial."""
        return self.query(
            """MATCH (t:ClinicalTrial {id: $id})-[r]->(n)
            RETURN type(r) AS relationship, labels(n)[0] AS node_type,
                   n.name AS name, properties(r) AS props""",
            params={"id": trial_id},
        )

    def get_adverse_events_by_population(self, drug_name: str) -> list[dict]:
        """Get adverse events grouped by population for a drug."""
        return self.query(
            """MATCH (t:ClinicalTrial)-[:USES_DRUG]->(d:Drug {name: $drug}),
                  (t)-[r:REPORTS]->(ae:AdverseEvent)
            RETURN ae.name AS adverse_event, r.population AS population,
                   r.rate AS rate, t.country AS country, t.language AS language
            ORDER BY ae.name, r.population""",
            params={"drug": drug_name},
        )

    def get_schema(self) -> str:
        """Return the graph schema description for LLM prompts."""
        return GRAPH_SCHEMA

    def node_count(self) -> int:
        """Get total node count."""
        result = self.query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0

    def relationship_count(self) -> int:
        """Get total relationship count."""
        result = self.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0]["cnt"] if result else 0
