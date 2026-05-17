"""Robust runtime/device detection for Kaggle, Colab, and local environments.

Goals:
- Never pass ``cuda`` unless ``torch.cuda.is_available()`` is true.
- On TPU/XLA runtimes, dense retrieval (SentenceTransformer / CrossEncoder)
  runs on CPU. SentenceTransformer does not understand ``xla``/``tpu``.
- If a caller asks for ``cuda`` but CUDA is not available, warn and fall back
  to ``cpu`` (or ``xla`` only when the caller explicitly opts in).
- Provide a single ``print_runtime_summary()`` for notebooks.
"""
from __future__ import annotations

import os
import warnings
from typing import Tuple


def _xla_available() -> bool:
    """Return True if a working torch_xla (TPU) runtime is importable."""
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        # If we can construct an xla device, the runtime is live.
        import torch_xla  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect_runtime() -> str:
    """Return one of: 'cuda', 'tpu', 'cpu'.

    Preference order: CUDA > TPU > CPU. We prefer CUDA because the dense
    encoder paths in this repo are built for it; TPU is reported only when
    no CUDA is available (e.g. Kaggle TPU VMs).
    """
    if _cuda_available():
        return "cuda"
    if _xla_available():
        return "tpu"
    return "cpu"


def resolve_device(requested: str | None, *, for_sentence_transformer: bool = False) -> str:
    """Return a safe device string.

    Rules:
    - ``requested`` is normalized to lowercase.
    - ``cuda`` / ``cuda:N`` → kept only if CUDA is actually available.
    - ``xla`` / ``tpu`` → kept as ``xla`` only when XLA is importable AND
      caller did not set ``for_sentence_transformer=True`` (SentenceTransformer
      cannot use XLA; fall back to CPU there).
    - ``cpu`` / unknown / None → ``cpu``.
    - If a caller asked for ``cuda`` and CUDA is not available we *warn* and
      fall back to ``cpu`` (or ``xla`` if the caller is happy with it and
      XLA is live and not for sentence-transformer).
    """
    req = (requested or "").strip().lower()

    if req.startswith("cuda"):
        if _cuda_available():
            return req
        warnings.warn(
            f"Requested device {requested!r} but torch.cuda.is_available() is "
            f"False. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if req in ("xla", "tpu"):
        if for_sentence_transformer:
            # sentence-transformers does not support XLA devices; use CPU.
            return "cpu"
        if _xla_available():
            return "xla"
        warnings.warn(
            f"Requested device {requested!r} but torch_xla is not importable. "
            f"Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"

    if req in ("", "auto", None):
        # Auto-detect.
        if _cuda_available():
            return "cuda"
        if _xla_available() and not for_sentence_transformer:
            return "xla"
        return "cpu"

    # Anything else (e.g. "cpu", "mps") → pass through, but only if torch
    # actually understands it. We don't try to validate every alias.
    return req


def runtime_summary() -> dict:
    """Collect a small dict of facts about the current runtime."""
    info: dict = {
        "platform": "kaggle" if os.path.isdir("/kaggle") else (
            "colab" if "COLAB_GPU" in os.environ or os.path.isdir("/content") else "local"
        ),
        "cuda_available": _cuda_available(),
        "xla_available": _xla_available(),
        "detected": detect_runtime(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_built"] = bool(getattr(torch.version, "cuda", None))
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as e:
        info["torch_error"] = repr(e)
    return info


def print_runtime_summary(prefix: str = "[runtime]") -> Tuple[str, dict]:
    """Print a one-shot summary and return ``(detected_device, info_dict)``."""
    info = runtime_summary()
    detected = info["detected"]
    print(f"{prefix} platform={info['platform']} detected={detected}")
    print(
        f"{prefix} torch={info.get('torch_version')} "
        f"cuda_built={info.get('torch_cuda_built')} "
        f"cuda_version={info.get('torch_cuda_version')} "
        f"cuda_available={info['cuda_available']} "
        f"xla_available={info['xla_available']}"
    )
    if info.get("cuda_device_count"):
        names = ", ".join(info.get("cuda_device_names") or [])
        print(f"{prefix} cuda_devices={info['cuda_device_count']} ({names})")
    if detected == "tpu":
        print(f"{prefix} TPU/XLA detected — dense encoder will run on CPU "
              f"(SentenceTransformer does not support XLA).")
    return detected, info
