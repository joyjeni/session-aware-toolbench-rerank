"""Download ToolBench (HuggingFace mirror) into data/toolbench_hf/.

FIX v18: added --out argparse argument so the notebook (or CLI) can
ovverride the destination directory. Without this, passing --out caused
'unrecognized arguments' crash.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "tuandunghcmut/toolbench-v1"
_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "toolbench_hf"


def main() -> str:
    ap = argparse.ArgumentParser(description="Download ToolBench from HuggingFace.")
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Destination directory (default: <repo_root>/data/toolbench_hf). "
            "The dataset will be placed directly inside this directory."
        ),
    )
    # parse_known_args so callers that pass extra flags (e.g. legacy --out path)
    # don't crash the process.
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[download] ignoring unknown args: {unknown}")

    dest = Path(args.out) if args.out else _DEFAULT_ROOT
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[download] fetching {REPO_ID!r} → {dest}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["benchmark/*.parquet", "data/*.parquet", "README.md"],
    )
    print(f"[download] dataset at {path}")
    return path


if __name__ == "__main__":
    main()
