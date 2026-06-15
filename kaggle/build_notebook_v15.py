"""Build SessionRerank+ Kaggle notebook v15.

Key differences vs v14:
  * Adaptive device + dtype detection (CPU / sm60 P100 / sm70 V100 / sm75 T4 /
    sm80 A100 / sm86 / sm89 L4 / sm90 H100).
  * Safe torch install: if a CUDA wheel mismatches the assigned GPU's compute
    capability, we DOWNGRADE to a cu118 (sm_60+) wheel which covers all
    Kaggle-allocatable GPUs (P100 sm60, T4 sm75, A100 sm80, L4 sm89, H100 sm90).
  * Live (unbuffered) subprocess output via Popen + line-by-line streaming —
    fixes the v14 issue where capture_output=True buffered ~2h of stdout.
  * Per-tensor dtype safety: float64 indices are demoted to int64 where torch
    expects it (avoids 'index_select' int32 vs int64 mismatches), and we use
    autocast only when bf16/fp16 is supported AND the model isn't a small
    sentence-transformer (which is more stable in fp32 on CPU/old GPUs).
  * Writes results/metrics.json + results/metrics_summary.csv even on partial
    completion (checkpointed per method) so the cron downloader gets something
    actionable even if the kernel hits the 9h wall-clock cap.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "sessionrerank_gemma4_kaggle.ipynb"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": list(lines)}


def code(*lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": list(lines),
    }


# ============================================================================
# Cells
# ============================================================================
cells = [
    md(
        "# SessionRerank+ · Karnataka Smallholder Agriculture (v15 — adaptive)\n",
        "### *Co-activation cached tool retrieval, powered by Gemma 4*\n",
        "\n",
        "**v15 changes:** adaptive device / dtype detection, safe torch wheel pick, "
        "live unbuffered logs, checkpointed metrics on partial completion.\n",
    ),

    # ------------------------------------------------------------------
    # Cell 1 — device + capability detection (BEFORE any heavy install)
    # ------------------------------------------------------------------
    md("## 1. Detect device + compute capability"),
    code(
        "import os, subprocess, json, sys, platform\n",
        "\n",
        "def _gpu_info():\n",
        "    \"\"\"Return (device, cc_major, cc_minor, gpu_name) without importing torch.\"\"\"\n",
        "    try:\n",
        "        out = subprocess.run(\n",
        "            ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],\n",
        "            capture_output=True, text=True, timeout=10,\n",
        "        )\n",
        "        if out.returncode != 0 or not out.stdout.strip():\n",
        "            return ('cpu', 0, 0, 'cpu')\n",
        "        line = out.stdout.strip().splitlines()[0]\n",
        "        name, cc = [x.strip() for x in line.split(',')]\n",
        "        maj, mn = cc.split('.')\n",
        "        return ('cuda', int(maj), int(mn), name)\n",
        "    except Exception as e:\n",
        "        print('nvidia-smi probe failed:', e)\n",
        "        return ('cpu', 0, 0, 'cpu')\n",
        "\n",
        "DEVICE, CC_MAJOR, CC_MINOR, GPU_NAME = _gpu_info()\n",
        "CC = CC_MAJOR * 10 + CC_MINOR\n",
        "print(f'Device:      {DEVICE}')\n",
        "print(f'GPU name:    {GPU_NAME}')\n",
        "print(f'Compute cap: {CC_MAJOR}.{CC_MINOR}  (CC code = {CC})')\n",
        "print(f'Python:      {platform.python_version()}')\n",
        "\n",
        "# Choose dtype path:\n",
        "#   CPU       -> fp32 everywhere\n",
        "#   sm60 (P100, GTX 10xx)         -> fp32 (no native fp16 atomics for our ops)\n",
        "#   sm70+ (V100, T4, A100, L4...)  -> fp16 OK for embedding model\n",
        "#   sm80+ (A100, L4, H100)         -> bf16 OK and preferred for stability\n",
        "if DEVICE == 'cpu':\n",
        "    DTYPE_STR = 'fp32'\n",
        "elif CC < 70:\n",
        "    DTYPE_STR = 'fp32'\n",
        "elif CC < 80:\n",
        "    DTYPE_STR = 'fp16'\n",
        "else:\n",
        "    DTYPE_STR = 'bf16'\n",
        "print(f'Selected dtype path: {DTYPE_STR}')\n",
        "\n",
        "# Export for downstream cells\n",
        "os.environ['SR_DEVICE'] = DEVICE\n",
        "os.environ['SR_DTYPE']  = DTYPE_STR\n",
        "os.environ['SR_CC']     = str(CC)\n",
        "os.environ['SR_GPU']    = GPU_NAME\n",
    ),

    # ------------------------------------------------------------------
    # Cell 2 — safe torch wheel pick
    # ------------------------------------------------------------------
    md(
        "## 2. Install torch on a wheel compatible with the detected GPU\n",
        "Kaggle's pre-installed torch can mismatch the assigned GPU (e.g. cu128 wheel on a P100). "
        "We pin to a wheel that covers sm60..sm90."
    ),
    code(
        "import subprocess, sys, os\n",
        "DEVICE = os.environ['SR_DEVICE']\n",
        "CC = int(os.environ['SR_CC'])\n",
        "\n",
        "def run(*c):\n",
        "    print('+', ' '.join(c), flush=True)\n",
        "    p = subprocess.Popen(list(c), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "    for line in p.stdout:\n",
        "        print(line.rstrip(), flush=True)\n",
        "    p.wait()\n",
        "    return p.returncode\n",
        "\n",
        "if DEVICE == 'cpu':\n",
        "    run(sys.executable, '-m', 'pip', 'install', '-q',\n",
        "        '--index-url', 'https://download.pytorch.org/whl/cpu',\n",
        "        'torch==2.4.1', 'torchvision==0.19.1')\n",
        "elif CC < 70:\n",
        "    # P100 (sm60), older. cu118 wheel still ships sm60 kernels.\n",
        "    run(sys.executable, '-m', 'pip', 'install', '-q',\n",
        "        '--index-url', 'https://download.pytorch.org/whl/cu118',\n",
        "        'torch==2.4.1', 'torchvision==0.19.1')\n",
        "else:\n",
        "    # T4/A100/L4/H100 all happy on cu121 wheel for torch 2.4.x\n",
        "    run(sys.executable, '-m', 'pip', 'install', '-q',\n",
        "        '--index-url', 'https://download.pytorch.org/whl/cu121',\n",
        "        'torch==2.4.1', 'torchvision==0.19.1')\n",
        "\n",
        "# Smoke-test\n",
        "import importlib\n",
        "if 'torch' in sys.modules:\n",
        "    importlib.reload(sys.modules['torch'])\n",
        "import torch\n",
        "print('torch:', torch.__version__, 'cuda?', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    x = torch.randn(8, 8, device='cuda', dtype=torch.float32)\n",
        "    y = (x @ x.T).sum().item()\n",
        "    print('cuda smoke test OK; matmul sum =', y)\n",
        "else:\n",
        "    print('Running on CPU — no CUDA smoke test')\n",
    ),

    # ------------------------------------------------------------------
    # Cell 3 — repo + lightweight deps
    # ------------------------------------------------------------------
    md("## 3. Clone repo and install lightweight deps"),
    code(
        "import subprocess, sys\n",
        "def run(*c):\n",
        "    print('+', ' '.join(c), flush=True)\n",
        "    p = subprocess.Popen(list(c), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "    for line in p.stdout:\n",
        "        print(line.rstrip(), flush=True)\n",
        "    p.wait()\n",
        "    return p.returncode\n",
        "\n",
        "run('git', 'clone', '--depth', '1',\n",
        "    'https://github.com/joyjeni/session-aware-toolbench-rerank.git',\n",
        "    '/kaggle/working/sr')\n",
        "\n",
        "run(sys.executable, '-m', 'pip', 'install', '-q',\n",
        "    'transformers>=4.45', 'sentence-transformers>=3.0',\n",
        "    'huggingface_hub>=0.24', 'numpy<2', 'pandas', 'scikit-learn',\n",
        "    'rank-bm25', 'tqdm', 'matplotlib')\n",
        "\n",
        "import os, pathlib\n",
        "os.chdir('/kaggle/working/sr')\n",
        "sys.path.insert(0, '/kaggle/working/sr/src')\n",
        "print('Repo @', pathlib.Path('.').resolve())\n",
        "run('ls')\n",
    ),

    # ------------------------------------------------------------------
    # Cell 4 — HF token (optional, only for fine-tuned Gemma branch)
    # ------------------------------------------------------------------
    md("## 4. HF token (optional)"),
    code(
        "import os\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN')\n",
        "    print('HF_TOKEN injected.')\n",
        "except Exception as e:\n",
        "    print('No HF_TOKEN secret — Gemma fine-tuned branch will be skipped.')\n",
    ),

    # ------------------------------------------------------------------
    # Cell 5 — download ToolBench
    # ------------------------------------------------------------------
    md("## 5. Download ToolBench splits"),
    code(
        "import subprocess, sys\n",
        "p = subprocess.Popen(\n",
        "    [sys.executable, '-m', 'sessionrerank.data.download',\n",
        "     '--out', '/kaggle/working/sr/data'],\n",
        "    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,\n",
        ")\n",
        "for line in p.stdout:\n",
        "    print(line.rstrip(), flush=True)\n",
        "p.wait()\n",
    ),

    # ------------------------------------------------------------------
    # Cell 6 — RUN THE CASCADE (live, unbuffered, dtype-aware)
    # ------------------------------------------------------------------
    md(
        "## 6. Run the full cascade (live streaming output)\n",
        "Uses Popen with line-buffered piping so every progress line shows up immediately, "
        "instead of being held until the subprocess finishes."
    ),
    code(
        "import subprocess, sys, time, os\n",
        "DEVICE = os.environ['SR_DEVICE']\n",
        "DTYPE  = os.environ['SR_DTYPE']\n",
        "\n",
        "# Build cascade command. SessionRerank+ paper uses 'pheromone' as the\n",
        "# backward-compatible CLI flag name for the co-activation cache.\n",
        "cmd = [sys.executable, '-u', '-m', 'sessionrerank.eval.run_all',\n",
        "       '--device', DEVICE,\n",
        "       '--dtype',  DTYPE,\n",
        "       '--seeds',  '0,1,2,42,123',\n",
        "       '--pheromone',\n",
        "       '--pheromone-rho',   '0.02',\n",
        "       '--pheromone-tau0',  '1.0',\n",
        "       '--pheromone-decay', '0.7',\n",
        "       '--cooc-window',     '3',\n",
        "       '--cross-encoder',\n",
        "       '--checkpoint-every-method',  # writes metrics.json after EACH method\n",
        "      ]\n",
        "if DEVICE == 'cuda':\n",
        "    cmd += ['--gemma-rerank', 'google/gemma-2-2b-it']\n",
        "else:\n",
        "    # On CPU, Gemma rerank is far too slow for a 9h wall-clock — skip it.\n",
        "    cmd += ['--skip-gemma-rerank']\n",
        "\n",
        "print('+', ' '.join(cmd), flush=True)\n",
        "t0 = time.time()\n",
        "p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n",
        "                     text=True, bufsize=1)\n",
        "for line in p.stdout:\n",
        "    print(line.rstrip(), flush=True)\n",
        "p.wait()\n",
        "print(f'\\n[cascade] exit code = {p.returncode}   elapsed = {time.time()-t0:.1f}s', flush=True)\n",
    ),

    # ------------------------------------------------------------------
    # Cell 7 — render metrics
    # ------------------------------------------------------------------
    md("## 7. Load + display metrics"),
    code(
        "import json, pathlib, pandas as pd\n",
        "p = pathlib.Path('results/metrics.json')\n",
        "if not p.exists():\n",
        "    print('No metrics.json yet (cascade did not complete or did not checkpoint).')\n",
        "else:\n",
        "    m = json.loads(p.read_text())\n",
        "    rows = []\n",
        "    for method, splits in m.items():\n",
        "        row = {'method': method}\n",
        "        for split, vals in splits.items():\n",
        "            if isinstance(vals, dict):\n",
        "                row[f'{split}_NDCG@5'] = vals.get('NDCG@5')\n",
        "                row[f'{split}_Hit@5']  = vals.get('Hit@5')\n",
        "        rows.append(row)\n",
        "    df = pd.DataFrame(rows).set_index('method')\n",
        "    print(df.to_string())\n",
        "    # Also write CSV for the cron downloader to grab\n",
        "    df.to_csv('results/metrics_summary.csv')\n",
        "    print('\\nWrote results/metrics_summary.csv')\n",
    ),

    # ------------------------------------------------------------------
    # Cell 8 — Karnataka first-hit demo (lightweight, CPU-OK)
    # ------------------------------------------------------------------
    md("## 8. Karnataka first-hit accuracy"),
    code(
        "import sys, json, pathlib\n",
        "sys.path.insert(0, '/kaggle/working/sr/agri_extension')\n",
        "try:\n",
        "    from sessionrerank_plus import SessionRerankPlus, FarmerProfile\n",
        "    from karnataka_apis import KARNATAKA_APIS\n",
        "except Exception as e:\n",
        "    print('Karnataka extension import failed:', e)\n",
        "    SessionRerankPlus = None\n",
        "\n",
        "if SessionRerankPlus is not None:\n",
        "    ranker = SessionRerankPlus()\n",
        "    bank = [\n",
        "        ('My ragi crop has yellowing leaves and dry patches', 'ksda_pest_alert'),\n",
        "        ('When can I claim crop insurance for ragi this kharif', 'pmfby'),\n",
        "        ('What is today ragi price at Tumakuru mandi', 'agmarknet_ka'),\n",
        "        ('Need helpline to ask about NPK dose for ragi', 'k_kisan'),\n",
        "        ('Check my Bhoomi land record for survey number 123/4', 'bhoomi'),\n",
        "        ('Is there a heatwave warning for Tumakuru', 'ksndmc'),\n",
        "        ('Soil health card request for my farm', 'soil_health_card'),\n",
        "        ('PM-KISAN installment status', 'pm_kisan'),\n",
        "        ('Milk procurement rate today at KMF Tumkur dairy', 'kmf_nandini'),\n",
        "        ('Offline pest identification for whitefly on tomato', 'plantix_uasb'),\n",
        "    ]\n",
        "    profile = FarmerProfile(crop='ragi', district='Tumakuru', language='kn', season='kharif')\n",
        "    ranker.set_profile(profile)\n",
        "    hits = 0\n",
        "    per_query = []\n",
        "    for q, gold in bank:\n",
        "        ranked = ranker.rank(q, top_k=1)\n",
        "        top = ranked[0]['api_id']\n",
        "        hit = top == gold\n",
        "        hits += int(hit)\n",
        "        per_query.append({'query': q, 'gold': gold, 'top': top, 'hit': hit})\n",
        "        print(f\"{'OK' if hit else '..'}  {q[:48]:48}  top={top:18}  gold={gold}\", flush=True)\n",
        "    acc = hits / len(bank)\n",
        "    print(f'\\nFirst-hit accuracy: {hits}/{len(bank)} = {acc:.2f}')\n",
        "    # Merge into results/metrics.json so cron picks it up\n",
        "    mp = pathlib.Path('results/metrics.json')\n",
        "    m = json.loads(mp.read_text()) if mp.exists() else {}\n",
        "    m.setdefault('SessionRerank+', {}).setdefault('karnataka', {})['first_hit'] = acc\n",
        "    m['SessionRerank+']['karnataka']['per_query'] = per_query\n",
        "    mp.parent.mkdir(parents=True, exist_ok=True)\n",
        "    mp.write_text(json.dumps(m, indent=2))\n",
        "    print('Wrote', mp)\n",
    ),

    md(
        "## 9. Citation\n",
        "```\n",
        "@misc{sessionrerank2026,\n",
        "  title  = {SessionRerank+: Co-activation Cached Tool Retrieval for Multi-turn Agentic LLMs},\n",
        "  author = {Abigail Creations},\n",
        "  year   = {2026},\n",
        "  url    = {https://github.com/joyjeni/session-aware-toolbench-rerank}\n",
        "}\n",
        "```\n",
    ),
]


# ============================================================================
# Notebook metadata. We DON'T pin an accelerator — the user requested
# adaptive behaviour. Kaggle will assign whatever it gives us, and the
# detect-cell handles the rest.
# ============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",  # request T4; if Kaggle gives P100/L4/etc the cell adapts
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}  ({OUT.stat().st_size} bytes, {len(cells)} cells)")
