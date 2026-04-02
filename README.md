# MedBridge: Multilingual Clinical Trial Intelligence

A fully local, multi-agentic system for cross-lingual clinical trial discovery and analysis, powered by [Microsoft Harrier-OSS-v1](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) multilingual embeddings.

**Search "metformin cardiovascular outcomes" in English, get results from Chinese, Japanese, and German studies — with zero translation.**

## What It Does

MedBridge demonstrates how multilingual embedding models enable cross-lingual medical research intelligence:

- **Cross-lingual semantic search** — Query in any language, find trials in all 94 supported languages
- **Drug interaction knowledge graph** — Visualize drug relationships extracted from global studies
- **Cross-cultural adverse event analysis** — Compare safety signals across populations (e.g., Asian vs Western)
- **Multi-agent orchestration** — 5 LangGraph agents collaborate with visible trace

## Tech Stack (100% Local, No Cloud Required)

| Component | Technology |
|-----------|-----------|
| Embeddings | [Harrier-OSS-v1-0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) via sentence-transformers (MPS GPU) |
| LLM | [Gemma 4 E2B](https://huggingface.co/lmstudio-community/gemma-4-E2B-it-GGUF) via llama-cpp-python (Metal GPU) |
| Vector DB | Qdrant (embedded mode, no server) |
| Graph DB | FalkorDBLite (embedded, Cypher queries) |
| Agents | LangGraph (StateGraph with conditional routing) |
| UI | Streamlit |
| Package Manager | uv |

**Runs entirely on a MacBook Pro M3 with 8GB RAM.**

## Quick Start

**Requirements**: macOS with Apple Silicon (M1/M2/M3), Python 3.12, [uv](https://docs.astral.sh/uv/), [Homebrew](https://brew.sh/)

```bash
# 1. Clone
git clone https://github.com/youruser/harrier-demo.git
cd harrier-demo

# 2. Install system dependency (FalkorDBLite needs OpenMP)
brew install libomp

# 3. Install all Python dependencies (includes Metal-accelerated llama-cpp-python)
uv sync

# 4. Download models (~2.5GB total)
uv run python scripts/download_models.py

# 5. Generate sample data + build databases
uv run python scripts/setup_databases.py

# 6. Launch web UI
uv run streamlit run src/ui/app.py

# Or use CLI mode:
uv run python main.py
```

> **Note**: The `uv.lock` file is pre-generated for `macosx_arm64`. On Apple Silicon Macs,
> `uv sync` will install Metal-accelerated `llama-cpp-python` and `torch` with MPS support automatically.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Streamlit Web UI             │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │     LangGraph Supervisor Agent       │
                    └──┬───────┬───────┬──────┬──────────┘
                       │       │       │      │
          ┌────────────▼─┐ ┌──▼─────┐ ┌▼─────▼──┐ ┌─────────────┐
          │  Ingestion   │ │Semantic│ │ Graph    │ │  Analysis   │
          │  Agent       │ │Search  │ │ Query    │ │  Agent      │
          └──────┬───────┘ └──┬─────┘ └──┬──────┘ └──────┬──────┘
                 │            │          │               │
    ┌────────────▼────────────▼──┐  ┌────▼──────────────▼────┐
    │   Qdrant (embedded)       │  │  FalkorDBLite (embedded)│
    └───────────────────────────┘  └─────────────────────────┘
                 │                          │
    ┌────────────▼──────────────────────────▼────┐
    │      Harrier-OSS-v1-0.6B (MPS GPU)        │
    └───────────────────────────────────────────┘
```

## Demo Queries

Try these in the UI:

1. **Cross-lingual search**: "Find studies on metformin cardiovascular outcomes"
2. **Drug interactions**: "Drug interactions for diabetes medications"
3. **Cross-cultural analysis**: "Adverse events for empagliflozin in Asian vs Western populations"
4. **Trial comparison**: "Compare statin trials across languages"
5. **Oncology**: "What are the side effects of pembrolizumab in East Asian patients?"

## How Harrier Makes This Possible

Harrier-OSS-v1 maps text from 94 languages into a **single shared 1024-dimensional vector space**. This means:

- "Metformin" (English), "二甲双胍" (Chinese), "メトホルミン" (Japanese) are geometrically close
- Cross-lingual search is just cosine similarity — no translation pipeline needed
- Entity resolution across languages works via embedding proximity

## Sample Data

The demo includes 20 synthetic clinical trial abstracts in 7 languages (EN, ZH, JA, DE, FR, ES, KO) covering:

- Type 2 Diabetes (Metformin, Sitagliptin, Empagliflozin)
- Cardiovascular (Atorvastatin, Rosuvastatin, Aspirin)
- Hypertension (Amlodipine, Losartan)
- Oncology (Pembrolizumab)
- Mental Health (Sertraline, Escitalopram)

With deliberate cross-population differences in adverse event rates for demo purposes.

## License

MIT
