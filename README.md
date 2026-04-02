# MedBridge: Multilingual Clinical Trial Intelligence

**Everything runs locally. No cloud. No API keys. No data leaves your machine.**

A fully local, multi-agentic system for cross-lingual clinical trial discovery and analysis -- powered by [Microsoft Harrier](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) multilingual embeddings, [Google Gemma 4 E2B](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) LLM (via [llama.cpp](https://github.com/ggml-org/llama.cpp)), [Qdrant](https://qdrant.tech/) vector search, [FalkorDB](https://www.falkordb.com/) knowledge graph, and [LangGraph](https://github.com/langchain-ai/langgraph) multi-agent orchestration.

**Search "metformin cardiovascular outcomes" in English, get results from Chinese, Japanese, and German studies -- with zero translation.**

## Screenshots

### LLM-Powered Cross-Lingual Analysis
Gemma 4 E2B analyzes retrieved trials across languages and synthesizes findings with cross-lingual insights -- all running locally on Metal GPU via llama-server.

![LLM Analysis](docs/images/llm_analysis.png)

### Multilingual Trial Retrieval
Harrier embeddings retrieve relevant trials across languages (English, German, Japanese, etc.) ranked by semantic similarity -- no translation needed.

![Search Results](docs/images/search_results.png)

### Hardware Footprint (Apple M3, 8GB RAM)
Both models (Harrier + Gemma 4) run comfortably on an 8GB Apple Silicon Mac -- 5.6GB RAM, 14% GPU, under 1W average power draw at idle after inference.

![Hardware Monitor](docs/images/asitop_hardware.png)

## Why Local Matters

Clinical trial data is sensitive. MedBridge proves you can build a production-grade multilingual research intelligence system that runs entirely on a laptop -- no cloud dependencies, no data exfiltration risk, no API costs. Every component is open-source and runs on consumer hardware.

## What It Does

- **Cross-lingual semantic search** -- Query in any language, find trials in all 94 supported languages via Harrier embeddings
- **Drug interaction knowledge graph** -- Visualize drug relationships in FalkorDBLite with Cypher queries
- **Cross-cultural adverse event analysis** -- Compare safety signals across populations using Gemma 4 LLM analysis
- **Multi-agent orchestration** -- 5 LangGraph agents (supervisor, search, graph, analysis, ingestion) collaborate with visible trace
- **Vector + graph hybrid retrieval** -- Qdrant semantic search combined with FalkorDB graph traversal

## Tech Stack (100% Local, No Cloud Required)

| Component | Technology | Role |
|-----------|-----------|------|
| Embeddings | [Harrier-OSS-v1-0.6B](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) (MPS GPU) | 94-language multilingual embeddings, 1024-dim, 32K context |
| LLM | [Gemma 4 E2B-it](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) Q4_K_M via [llama-server](https://github.com/ggml-org/llama.cpp) (Metal GPU) | Intent classification, Cypher generation, analysis synthesis |
| Vector DB | [Qdrant](https://qdrant.tech/) (embedded, no server) | Semantic similarity search over trial embeddings |
| Graph DB | [FalkorDBLite](https://github.com/FalkorDB/FalkorDB) (embedded) | Drug interactions, trial relationships via Cypher |
| Agents | [LangGraph](https://github.com/langchain-ai/langgraph) (StateGraph) | Multi-agent orchestration with conditional routing |
| UI | [Streamlit](https://streamlit.io/) | Interactive web dashboard with search, graph viz, analytics |

**Runs entirely on a MacBook with Apple Silicon (M1/M2/M3/M4) and 8GB RAM.**

## Quick Start

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.12 (must be arm64 -- see Troubleshooting if unsure)
- Xcode Command Line Tools: `xcode-select --install`
- [Homebrew](https://brew.sh/): needed to install llama.cpp
- ~8GB free disk space (dependencies ~2GB + Harrier model ~1.5GB + Gemma 4 model ~3.2GB)

### Setup (~10 minutes)

```bash
# 1. Clone (~1s)
git clone https://github.com/pankajarm/medbridge.git
cd medbridge

# 2. Install llama.cpp for the LLM server (~30s)
#    Gemma 4 requires build b8636+. Use --HEAD if stable is too old.
brew install llama.cpp --HEAD

# 3. Create a native arm64 virtual environment (~2s)
#    IMPORTANT: The venv MUST use arm64 Python, not x86_64/Rosetta.
#    PyTorch >= 2.4 only ships macOS arm64 wheels.
python3.12 -m venv .venv
source .venv/bin/activate

# Verify architecture (must print arm64, NOT x86_64):
python -c "import platform; print(platform.machine())"

# 4. Install Python dependencies (~2 min)
pip install -e .

# 5. Copy environment config
cp .env.example .env

# 6. Download Harrier embedding model (~2 min, ~1.5GB)
python scripts/download_models.py

# 7. Start Gemma 4 LLM server in a separate terminal (~3 min first run)
#    Auto-downloads the 3.2GB model on first run, then starts serving.
#    Keep this running in the background.
llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M

# 8. Generate sample data + build databases (~1 min)
#    (Back in your original terminal with the venv activated)
python scripts/setup_databases.py

# 9. Launch web UI
streamlit run src/ui/app.py
```

Open http://localhost:8501 in your browser.

### Setup Time Breakdown

| Step | Command | Time | Notes |
|------|---------|------|-------|
| 1. Clone | `git clone` | ~1s | |
| 2. Install llama.cpp | `brew install llama.cpp --HEAD` | ~30s | Needs Xcode CLT for HEAD build |
| 3. Create venv | `python3.12 -m venv .venv` | ~2s | Must be arm64 Python |
| 4. Install Python deps | `pip install -e .` | ~2 min | PyTorch, Transformers, Streamlit, etc. |
| 5. Config | `cp .env.example .env` | instant | |
| 6. Download Harrier | `python scripts/download_models.py` | ~2 min | Harrier 1.5GB embedding model |
| 7. Start LLM server | `llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M` | ~3 min | Auto-downloads 3.2GB on first run |
| 8. Setup databases | `python scripts/setup_databases.py` | ~1 min | 20 trials, 116 graph nodes |
| 9. Launch | `streamlit run src/ui/app.py` | ~10s | First load warms up models |
| **Total** | | **~10 min** | **Tested on M3 8GB** |

### CLI Mode

```bash
python main.py
```

## Memory Usage (8GB Mac)

| Component | RAM |
|-----------|-----|
| Harrier embeddings (MPS) | ~1.5 GB |
| Gemma 4 E2B Q4 via llama-server (Metal) | ~3.2 GB |
| Qdrant + FalkorDB | ~100 MB |
| Streamlit + Python | ~200 MB |
| **Total** | **~5.0 GB** |

Some swap usage is expected on 8GB machines when both models are loaded. Performance remains good thanks to unified memory and Metal GPU acceleration.

## Architecture

```
                    +-------------------------------------+
                    |         Streamlit Web UI             |
                    +--------------+----------------------+
                                   |
                    +--------------v----------------------+
                    |     LangGraph Supervisor Agent       |
                    +--+-------+-------+------+----------+
                       |       |       |      |
          +------------v-+ +--v-----+ +v-----v--+ +-------------+
          |  Ingestion   | |Semantic| | Graph    | |  Analysis   |
          |  Agent       | |Search  | | Query    | |  Agent      |
          +------+-------+ +--+-----+ +--+------+ +------+------+
                 |            |          |               |
    +------------v------------v--+  +----v----+  +------v--------+
    |   Qdrant (embedded)       |  | FalkorDB|  | llama-server   |
    +---------------------------+  | (embed) |  | Gemma 4 (Metal)|
                 |                 +----+-----+  +------+---------+
    +------------v----------------------v---------------+
    |        Harrier-OSS-v1-0.6B (MPS GPU)              |
    +---------------------------------------------------+
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
- Cross-lingual search is just cosine similarity -- no translation pipeline needed
- Entity resolution across languages works via embedding proximity

## Sample Data

The demo includes 20 synthetic clinical trial abstracts in 7 languages (EN, ZH, JA, DE, FR, ES, KO) covering:

- Type 2 Diabetes (Metformin, Sitagliptin, Empagliflozin)
- Cardiovascular (Atorvastatin, Rosuvastatin, Aspirin)
- Hypertension (Amlodipine, Losartan)
- Oncology (Pembrolizumab)
- Mental Health (Sertraline, Escitalopram)

With deliberate cross-population differences in adverse event rates for demo purposes.

## Troubleshooting

### `llama-server` not found

Install llama.cpp via Homebrew:
```bash
brew install llama.cpp --HEAD
```

The `--HEAD` flag builds from the latest source (requires Xcode Command Line Tools). If that fails, try the stable bottle first:
```bash
brew install llama.cpp
```

If the stable version gives `unknown model architecture: 'gemma4'`, the bottle is too old. Either retry with `--HEAD` or download a pre-built binary (build b8636+) from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases):
Download the `llama-*-bin-macos-arm64.tar.gz` asset from the [latest release](https://github.com/ggml-org/llama.cpp/releases), extract it, and run `llama-server` from the extracted directory.

### LLM analysis shows mock responses

The LLM server isn't running. Start it in a separate terminal:
```bash
llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M
```

Wait until you see `HTTP server listening` before using the app. The app gracefully falls back to keyword-based mock responses when the server is unavailable.

### Python reports `x86_64` instead of `arm64`

Your venv was created with an x86_64 Python (Rosetta). Recreate it:

```bash
rm -rf .venv
# Use a universal or arm64 Python binary:
arch -arm64 python3.12 -m venv .venv
source .venv/bin/activate
```

## License

MIT
