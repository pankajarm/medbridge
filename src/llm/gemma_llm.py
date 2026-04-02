"""Gemma 4 E2B LLM wrapper using llama-cpp-python with Metal GPU acceleration.

Falls back to a keyword-based mock when llama-cpp-python is not installed,
allowing the rest of the system (Harrier embeddings, Qdrant, FalkorDB, LangGraph)
to be tested independently.
"""

import re
from pathlib import Path

from src.config import GEMMA_MODEL_PATH, LLM_N_GPU_LAYERS, LLM_N_CTX, LLM_TEMPERATURE

try:
    from llama_cpp import Llama
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


class GemmaLLM:
    """Wrapper for Gemma 4 E2B GGUF model via llama-cpp-python."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or GEMMA_MODEL_PATH
        self.llm = None

        if HAS_LLAMA_CPP and Path(self.model_path).exists():
            print(f"Loading Gemma model from {self.model_path}...")
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=LLM_N_GPU_LAYERS,
                n_ctx=LLM_N_CTX,
                verbose=False,
            )
            print("Gemma model loaded.")
        else:
            reason = "llama-cpp-python not installed" if not HAS_LLAMA_CPP else f"Model not found at {self.model_path}"
            print(f"[MockLLM] {reason}. Using keyword-based mock for testing.")
            print("[MockLLM] Install llama-cpp-python and download models for full LLM support.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful medical research assistant.",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> str:
        """Generate a response from the model."""
        if self.llm is None:
            return self._mock_generate(prompt, system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature or LLM_TEMPERATURE,
        )
        return response["choices"][0]["message"]["content"]

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        system_prompt: str = "You are a helpful medical research assistant.",
        max_tokens: int = 1024,
    ) -> dict:
        """Generate a response with tool/function calling support."""
        if self.llm is None:
            return {"content": self._mock_generate(prompt, system_prompt)}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.llm.create_chat_completion(
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=LLM_TEMPERATURE,
        )
        return response["choices"][0]["message"]

    def classify_intent(self, query: str) -> str:
        """Classify a user query into a query type."""
        if self.llm is None:
            return self._mock_classify(query)

        system = """Classify the user's medical research query into exactly one category.
Reply with ONLY the category name, nothing else.

Categories:
- literature_search: Finding clinical trials or research papers
- drug_interaction: Questions about drug-drug interactions
- adverse_event: Questions about side effects or adverse events
- cross_cultural_analysis: Comparing outcomes across populations/countries
- trial_comparison: Comparing specific trials or studies"""

        result = self.generate(query, system_prompt=system, max_tokens=50, temperature=0.1)
        result = result.strip().lower().replace(" ", "_")
        valid = {"literature_search", "drug_interaction", "adverse_event",
                 "cross_cultural_analysis", "trial_comparison"}
        return result if result in valid else "literature_search"

    def extract_entities(self, text: str, language: str = "en") -> dict:
        """Extract medical entities from clinical trial text."""
        if self.llm is None:
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
            max_tokens=512,
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
        if self.llm is None:
            return self._mock_cypher(question)

        system = f"""You are a graph database expert. Generate a Cypher query for FalkorDB
to answer the user's question about clinical trials.

Graph schema:
{schema}

Return ONLY the Cypher query, no explanation. Use MATCH, WHERE, RETURN clauses."""

        result = self.generate(question, system_prompt=system, max_tokens=256, temperature=0.1)
        result = result.strip()
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return result.strip()

    # --- Mock methods for testing without LLM ---

    def _mock_classify(self, query: str) -> str:
        """Keyword-based intent classification for testing."""
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
        """Simple mock response for testing."""
        return (
            "[MockLLM Response] Based on the available clinical trial data, "
            "the analysis shows cross-lingual results from multiple populations. "
            "The Harrier embedding model successfully retrieved relevant trials "
            "across languages without translation. Further analysis with a full "
            "LLM would provide deeper insights into population-specific outcomes."
        )

    def _mock_cypher(self, question: str) -> str:
        """Generate simple Cypher queries based on keywords."""
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
        # Default: return all trials
        return (
            "MATCH (t:ClinicalTrial) "
            "RETURN t.id AS id, t.title AS title, t.language AS language, t.country AS country "
            "LIMIT 20"
        )
