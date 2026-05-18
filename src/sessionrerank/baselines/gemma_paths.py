"""Discover & validate Gemma model paths.

Also defines :class:`GemmaModelUnavailable`, raised by Gemma baselines
when no usable checkpoint is reachable so the caller can mark the method
skipped instead of crashing.
"""
from __future__ import annotations


class GemmaModelUnavailable(RuntimeError):
    """Raised when a Gemma model id/path can't be loaded.

    ``run_all`` catches this and marks the affected method as skipped in
    the metrics output instead of aborting the entire evaluation.
    """

import glob
import os
import re
from pathlib import Path

DEFAULT_LOCAL_ROOTS = ("/kaggle/input", "/content", os.path.expanduser("~/.cache/huggingface"))

# Known-public Gemma model IDs we'll accept as safe fallbacks.
# Order matters: the FIRST entry is the preferred default if nothing else
# is available. These have all been validated on huggingface.co.
KNOWN_PUBLIC_BIENCODER_IDS = (
    "google/gemma-2-2b",       # public, no gate beyond HF login
    "google/gemma-3-270m",     # public Gemma 3 small
)
KNOWN_PUBLIC_RERANKER_IDS = (
    "google/gemma-2-2b-it",    # instruct, public
    "google/gemma-3-270m-it",  # instruct Gemma 3
)

# IDs known to be invalid / unpublished (we'll refuse them with a clear msg
# instead of letting transformers raise a 404 inside heavy stack frames).
KNOWN_INVALID_IDS = frozenset({
    "google/gemma-4-270m",
    "google/gemma-4-270m-it",
})


def _is_local_model_dir(path: str | os.PathLike) -> bool:
    """A directory is a usable HF model dir iff it has config.json and a
    tokenizer file (tokenizer.json, tokenizer.model, or tokenizer_config.json).
    """
    if not path:
        return False
    p = Path(path)
    if not p.is_dir():
        return False
    if not (p / "config.json").is_file():
        return False
    tok_candidates = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    return any((p / t).is_file() for t in tok_candidates)


def find_local_gemma_dirs(roots: tuple[str, ...] = DEFAULT_LOCAL_ROOTS) -> list[str]:
    """Return all directories under ``roots`` that look like a Gemma model.

    A directory is considered a Gemma checkpoint if:
      * it contains ``config.json`` and a tokenizer file, and
      * either the directory name OR ``config.json`` mentions 'gemma'.
    """
    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for cfg in glob.iglob(os.path.join(root, "**", "config.json"), recursive=True):
            d = os.path.dirname(cfg)
            if d in seen:
                continue
            if not _is_local_model_dir(d):
                continue
            name = d.lower()
            looks_gemma = "gemma" in name
            if not looks_gemma:
                try:
                    with open(cfg, "r") as f:
                        head = f.read(2048).lower()
                    looks_gemma = "gemma" in head
                except Exception:
                    looks_gemma = False
            if looks_gemma:
                found.append(d)
                seen.add(d)
    return sorted(found)


def _classify_local_dir(path: str) -> str:
    """'reranker' if the path/name implies instruct (it/instruct/chat),
    otherwise 'biencoder'."""
    n = path.lower()
    if re.search(r"(?:^|[-_/])it(?:$|[-_/])", n) or "instruct" in n or "chat" in n:
        return "reranker"
    return "biencoder"


def discover_gemma_paths(
    roots: tuple[str, ...] = DEFAULT_LOCAL_ROOTS,
) -> dict[str, list[str]]:
    """Group local Gemma dirs by best-guess role."""
    out: dict[str, list[str]] = {"biencoder": [], "reranker": []}
    for d in find_local_gemma_dirs(roots):
        out[_classify_local_dir(d)].append(d)
    return out


def _hf_id_exists(model_id: str, token: str | None = None) -> bool:
    """Return True if the given HF model id resolves on the Hub.

    Uses ``huggingface_hub.HfApi.model_info`` if available. If
    ``huggingface_hub`` isn't installed, we conservatively return True so
    the caller can attempt the load and surface a real error from
    transformers.
    """
    if not model_id or "/" not in model_id:
        return False
    if model_id in KNOWN_INVALID_IDS:
        return False
    try:
        from huggingface_hub import HfApi
        try:
            HfApi().model_info(model_id, token=token)
            return True
        except Exception:
            return False
    except ImportError:
        return True


def resolve_gemma_model(
    role: str,
    override: str | None = None,
    *,
    env_var: str | None = None,
    roots: tuple[str, ...] = DEFAULT_LOCAL_ROOTS,
    allow_public_fallback: bool = True,
    hf_token: str | None = None,
    verbose: bool = True,
) -> str | None:
    """Resolve a usable Gemma model path/id for ``role``.

    Parameters
    ----------
    role : 'biencoder' | 'reranker'
    override : explicit caller value (highest priority)
    env_var : env-var name to consult if override is None
        (defaults: GEMMA_BIENCODER / GEMMA_RERANKER)

    Returns
    -------
    str or None — path/id, or None if nothing usable was found.
    """
    if role not in ("biencoder", "reranker"):
        raise ValueError(f"role must be biencoder|reranker, got {role!r}")
    if env_var is None:
        env_var = "GEMMA_BIENCODER" if role == "biencoder" else "GEMMA_RERANKER"
    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    def _say(msg: str) -> None:
        if verbose:
            print(f"[gemma_paths] {msg}")

    candidates: list[tuple[str, str]] = []  # (source, value)
    if override:
        candidates.append(("override", override))
    env_val = os.environ.get(env_var)
    if env_val and env_val != override:
        candidates.append((f"env:{env_var}", env_val))

    for source, val in candidates:
        if val in KNOWN_INVALID_IDS:
            _say(f"{source}={val!r} is a known-invalid public id; ignoring")
            continue
        if _is_local_model_dir(val):
            _say(f"{role}: using {source}={val} (local dir OK)")
            return val
        if os.path.isabs(val) or val.startswith(("./", "../")) or os.sep in val:
            _say(f"{source}={val!r} looks like a path but is not a valid model dir; ignoring")
            continue
        if _hf_id_exists(val, token=hf_token):
            _say(f"{role}: using {source}={val} (HF id resolves)")
            return val
        _say(f"{source}={val!r} does not resolve on HF; ignoring")

    grouped = discover_gemma_paths(roots)
    for path in grouped.get(role, []):
        _say(f"{role}: discovered local checkpoint {path}")
        return path
    other_role = "reranker" if role == "biencoder" else "biencoder"
    for path in grouped.get(other_role, []):
        _say(f"{role}: falling back to {other_role} checkpoint {path} "
             "(no role-specific dir found)")
        return path

    if allow_public_fallback:
        ids = KNOWN_PUBLIC_BIENCODER_IDS if role == "biencoder" else KNOWN_PUBLIC_RERANKER_IDS
        for hf_id in ids:
            if _hf_id_exists(hf_id, token=hf_token):
                _say(f"{role}: falling back to public HF id {hf_id}")
                return hf_id

    _say(f"{role}: no usable model found; Gemma methods will be skipped")
    return None


def print_discovery_report(roots: tuple[str, ...] = DEFAULT_LOCAL_ROOTS) -> None:
    """Pretty-print all local Gemma candidates. Useful from notebooks."""
    grouped = discover_gemma_paths(roots)
    print("Local Gemma checkpoint discovery:")
    print(f"  roots searched: {list(roots)}")
    for role in ("biencoder", "reranker"):
        paths = grouped.get(role, [])
        print(f"  {role}: {len(paths)} candidate(s)")
        for p in paths:
            print(f"    - {p}")
    if not any(grouped.values()):
        print("  (no local Gemma checkpoints found; will fall back to HF or skip)")


def validate_or_resolve(
    biencoder: str | None,
    reranker: str | None,
    *,
    allow_public_fallback: bool = True,
    verbose: bool = True,
) -> tuple[str | None, str | None]:
    """Convenience: resolve both roles in one call."""
    b = resolve_gemma_model(
        "biencoder", override=biencoder,
        allow_public_fallback=allow_public_fallback, verbose=verbose,
    )
    r = resolve_gemma_model(
        "reranker", override=reranker,
        allow_public_fallback=allow_public_fallback, verbose=verbose,
    )
    return b, r
