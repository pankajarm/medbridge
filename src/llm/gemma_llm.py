"""Gemma LLM wrapper using OpenAI-compatible API (llama-server).

Connects to a local llama-server running Gemma 4 (or any compatible model)
via the OpenAI chat completions API at http://localhost:8080/v1.

Start the server with:
    llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M

Falls back to a keyword-based mock when the server is not running.
"""

import re

from src.config import LLM_API_BASE, LLM_MODEL_NAME, LLM_TEMPERATURE

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class GemmaLLM:
    """Wrapper for Gemma via llama-server OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or LLM_API_BASE
        self.model = model or LLM_MODEL_NAME
        self.client = None

        if HAS_OPENAI:
            try:
                self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
                # Quick connectivity check
                self.client.models.list()
                print(f"Connected to LLM server at {self.base_url}")
            except Exception as e:
                print(f"[MockLLM] Could not connect to LLM server at {self.base_url}: {e}")
                print("[MockLLM] Start the server with: llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M")
                self.client = None
        else:
            print("[MockLLM] openai package not installed. Using mock.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful medical research assistant.",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        """Generate a response from the model."""
        if self.client is None:
            return self._mock_generate(prompt, system_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            )
            return response.choices[0].message.content
        except Exception:
            return self._mock_generate(prompt, system_prompt)

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: str = "You are a helpful medical research assistant.",
        max_tokens: int = 1024,
    ) -> dict:
        """Generate a response with tool/function calling support."""
        if self.client is None:
            return {"content": self._mock_generate(prompt, system_prompt)}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tools=tools,
                max_tokens=max_tokens,
                temperature=LLM_TEMPERATURE,
            )
            msg = response.choices[0].message
            return {"content": msg.content, "tool_calls": getattr(msg, "tool_calls", None)}
        except Exception:
            return {"content": self._mock_generate(prompt, system_prompt)}

    def classify_intent(self, query: str) -> str:
        """Classify a user query into a query type."""
        if self.client is None:
            return self._mock_classify(query)

        system = """Classify the user's medical research query into exactly one category.
Reply with ONLY the category name, nothing else.

Categories:
- literature_search: Finding clinical trials or research papers
- drug_interaction: Questions about drug-drug interactions
- adverse_event: Questions about side effects or adverse events
- cross_cultural_analysis: Comparing outcomes across populations/countries
- trial_comparison: Comparing specific trials or studies"""

        result = self.generate(query, system_prompt=system, max_tokens=512, temperature=0.1)
        result = result.strip().lower().replace(" ", "_")
        valid = {"literature_search", "drug_interaction", "adverse_event",
                 "cross_cultural_analysis", "trial_comparison"}
        return result if result in valid else "literature_search"

    def extract_entities(self, text: str, language: str = "en") -> dict:
        """Extract medical entities from clinical trial text."""
        if self.client is None:
            return {"drugs": [], "diseases": [], "adverse_events": [],
                    "biomarkers": [], "population": ""}

        system = """Extract medical entities from the clinical trial text.
Return a JSON object with these keys:
- drugs: list of drug names mentioned
- diseases: list of disease/condition names
- adverse_events: list of adverse events/side effects
- biomarkers: list of biomarkers measured
- population: description of study population

Return ONLY valid JSON, no explanation."""

        result = self.generate(
            f"Language: {language}\n\nText: {text}",
            system_prompt=system,
            max_tokens=2048,
            temperature=0.1,
        )
        import json
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"drugs": [], "diseases": [], "adverse_events": [],
                    "biomarkers": [], "population": ""}

    def generate_cypher(self, question: str, schema: str) -> str:
        """Generate a Cypher query from natural language."""
        if self.client is None:
            return self._mock_cypher(question)

        system = f"""You are a graph database expert. Generate a Cypher query for FalkorDB
to answer the user's question about clinical trials.

Graph schema:
{schema}

Return ONLY the Cypher query, no explanation. Use MATCH, WHERE, RETURN clauses."""

        result = self.generate(question, system_prompt=system, max_tokens=1024, temperature=0.1)
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return result.strip()

    # --- Mock methods for testing without LLM ---

    def _mock_classify(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["interaction", "interact", "combine"]):
            return "drug_interaction"
        if any(w in q for w in ["adverse", "side effect", "safety", "toxicity"]):
            return "adverse_event"
        if any(w in q for w in ["compare", "comparison", "versus", "vs"]):
            return "trial_comparison"
        if any(w in q for w in ["population", "asian", "western", "cross-cultural", "cultural"]):
            return "cross_cultural_analysis"
        return "literature_search"

    def _mock_generate(self, prompt: str, system_prompt: str) -> str:
        return (
            "[MockLLM Response] Based on the available clinical trial data, "
            "the analysis shows cross-lingual results from multiple populations. "
            "The Harrier embedding model successfully retrieved relevant trials "
            "across languages without translation. Start llama-server for full "
            "LLM-powered analysis: llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M"
        )

    def _mock_cypher(self, question: str) -> str:
        q = question.lower()
        if "interaction" in q:
            return (
                "MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug) "
                "RETURN d1.name AS drug1, d2.name AS drug2, r.type AS type, r.severity AS severity"
            )
        if "adverse" in q or "side effect" in q:
            return (
                "MATCH (t:ClinicalTrial)-[r:REPORTS]->(ae:AdverseEvent) "
                "RETURN t.id AS trial, ae.name AS adverse_event, r.population AS population, r.rate AS rate "
                "ORDER BY r.rate DESC LIMIT 20"
            )
        return (
            "MATCH (t:ClinicalTrial) "
            "RETURN t.id AS id, t.title AS title, t.language AS language, t.country AS country "
            "LIMIT 20"
        )
