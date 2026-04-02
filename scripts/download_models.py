"""Download Harrier embedding model for local inference."""

from huggingface_hub import snapshot_download


def download_harrier():
    """Download Harrier-OSS-v1-0.6B for sentence-transformers."""
    print("Downloading Harrier-OSS-v1-0.6B...")
    snapshot_download(
        repo_id="microsoft/harrier-oss-v1-0.6b",
        local_dir=None,  # Uses default HF cache
    )
    print("Harrier model downloaded to HuggingFace cache.")


def main():
    print("=" * 60)
    print("MedBridge Model Download")
    print("=" * 60)
    download_harrier()
    print()
    print("Harrier embedding model downloaded!")
    print()
    print("For the LLM, start llama-server (auto-downloads the model):")
    print("  llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_K_M")


if __name__ == "__main__":
    main()
