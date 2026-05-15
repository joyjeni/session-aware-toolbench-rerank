"""Download ToolBench (HuggingFace mirror) into data/toolbench_hf/."""
from __future__ import annotations
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "tuandunghcmut/toolbench-v1"
ROOT = Path(__file__).resolve().parents[3] / "data" / "toolbench_hf"


def main() -> str:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(ROOT),
        allow_patterns=["benchmark/*.parquet", "data/*.parquet", "README.md"],
    )
    print(f"[download] dataset at {path}")
    return path


if __name__ == "__main__":
    main()
