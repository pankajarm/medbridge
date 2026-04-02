"""MedBridge configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

# Model settings
HARRIER_MODEL_NAME = os.getenv("HARRIER_MODEL_NAME", "microsoft/harrier-oss-v1-0.6b")
GEMMA_MODEL_PATH = os.getenv("GEMMA_MODEL_PATH", str(DATA_DIR / "models" / "google_gemma-3-4b-it-Q4_K_M.gguf"))

# Database paths
QDRANT_PATH = os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant"))
FALKORDB_PATH = os.getenv("FALKORDB_PATH", str(DATA_DIR / "falkordb"))

# LLM settings
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "8192"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Embedding settings
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "mps")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DIM = 1024

# Qdrant collection
QDRANT_COLLECTION = "clinical_trials"

# Medical task prompts for Harrier
HARRIER_PROMPTS = {
    "trial_retrieval": "Given a medical research query, retrieve relevant clinical trial abstracts",
    "trial_similarity": "Identify semantically similar clinical trial studies",
    "entity_matching": "Match medical entity names that refer to the same drug or disease",
    "adverse_event": "Find clinical studies reporting similar adverse events or safety signals",
}

# Sample trial languages
SUPPORTED_LANGUAGES = ["en", "zh", "ja", "de", "fr", "es", "ko"]
