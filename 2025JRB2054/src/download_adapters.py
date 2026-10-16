"""Download all Assignment 1 PEFT adapters from the public Hugging Face repo.

Usage (from the repository root):
    python src/download_adapters.py

This script can also run from a different working directory. It always stores
models at <repository_root>/adapters/, not <repository_root>/src/adapters/.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "uthith/ELL8299-Assignment1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_NAMES = ("best_sanitizer", "victim_raw", "victim_san")


def download_adapters() -> Path:
    """Download public adapters while preserving the remote folder structure."""
    print(f"Downloading adapters from https://huggingface.co/{HF_REPO_ID}")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="model",
        allow_patterns=["adapters/**"],
        local_dir=str(PROJECT_ROOT),
    )

    adapters_dir = PROJECT_ROOT / "adapters"
    missing = [
        name
        for name in ADAPTER_NAMES
        if not (adapters_dir / name).is_dir()
        or not any(p.is_file() for p in (adapters_dir / name).rglob("*"))
    ]
    if missing:
        raise RuntimeError(
            "Adapter folders missing or empty: "
            + ", ".join(missing)
            + ". Verify the Hugging Face upload and internet access."
        )

    for name in ADAPTER_NAMES:
        print(f"Ready: {adapters_dir / name}")
    return adapters_dir


if __name__ == "__main__":
    download_adapters()
