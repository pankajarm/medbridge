"""Download Harrier and Gemma models for local inference."""

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def download_harrier():
    """Download Harrier-OSS-v1-0.6B for sentence-transformers."""
    print("Downloading Harrier-OSS-v1-0.6B...")
    snapshot_download(
        repo_id="microsoft/harrier-oss-v1-0.6b",
        local_dir=None,  # Uses default HF cache
    )
    print("Harrier model downloaded to HuggingFace cache.")


def download_gemma():
    """Download Gemma 3 4B GGUF Q4_K_M for llama-cpp-python."""
    models_dir = Path(__file__).parent.parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / "google_gemma-3-4b-it-Q4_K_M.gguf"

    if output_path.exists():
        print(f"Gemma model already exists at {output_path}")
        return

    print("Downloading Gemma 3 4B GGUF Q4_K_M...")
    hf_hub_download(
        repo_id="bartowski/google_gemma-3-4b-it-GGUF",
        filename="google_gemma-3-4b-it-Q4_K_M.gguf",
        local_dir=str(models_dir),
    )
    print(f"Gemma model downloaded to {output_path}")


def main():
    print("=" * 60)
    print("MedBridge Model Download")
    print("=" * 60)
    download_harrier()
    print()
    download_gemma()
    print()
    print("All models downloaded successfully!")
    print("You can now run: uv run python scripts/setup_databases.py")


if __name__ == "__main__":
    main()
