"""Harrier-OSS-v1-0.6B embedding wrapper with MPS support and medical task prompts.

Falls back to a smaller compatible model when Harrier can't load (e.g., on older
torch/transformers versions running under Rosetta on x86_64).
"""

import torch
from sentence_transformers import SentenceTransformer

from src.config import HARRIER_MODEL_NAME, EMBEDDING_DEVICE, EMBEDDING_BATCH_SIZE, HARRIER_PROMPTS

# Fallback model that works with older transformers
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_DIM = 384


class HarrierEmbedder:
    """Wrapper for Microsoft Harrier-OSS-v1-0.6B multilingual embeddings."""

    def __init__(
        self,
        model_name: str = HARRIER_MODEL_NAME,
        device: str | None = None,
    ):
        self.device = device or self._detect_device()
        self.is_fallback = False

        try:
            print(f"Loading Harrier model on {self.device}...")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.dim = 1024
            print(f"Harrier model loaded: {model_name}")
        except (ValueError, OSError, ImportError) as e:
            print(f"[Warning] Could not load Harrier model: {e}")
            print(f"[Fallback] Loading {FALLBACK_MODEL} for testing...")
            self.model = SentenceTransformer(FALLBACK_MODEL, device=self.device)
            self.dim = FALLBACK_DIM
            self.is_fallback = True
            print(f"[Fallback] Model loaded. Cross-lingual features will be limited.")

    def _detect_device(self) -> str:
        if EMBEDDING_DEVICE == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def embed_query(self, query: str, task: str = "trial_retrieval") -> list[float]:
        """Embed a query with task-specific instruction prompt."""
        if self.is_fallback:
            # Fallback model doesn't support instruction prompts
            embedding = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        else:
            prompt = HARRIER_PROMPTS.get(task, HARRIER_PROMPTS["trial_retrieval"])
            instruction = f"Instruct: {prompt}\nQuery: "
            embedding = self.model.encode(
                query, prompt=instruction, normalize_embeddings=True, show_progress_bar=False,
            )
        return embedding.tolist()

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed documents (no instruction prefix needed per Harrier spec)."""
        embeddings = self.model.encode(
            documents, normalize_embeddings=True,
            batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=len(documents) > 10,
        )
        return embeddings.tolist()

    def embed_entity(self, entity_name: str) -> list[float]:
        """Embed an entity name for cross-lingual entity resolution."""
        return self.embed_query(entity_name, task="entity_matching")

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.model.encode(text1, normalize_embeddings=True)
        emb2 = self.model.encode(text2, normalize_embeddings=True)
        return float(emb1 @ emb2)
