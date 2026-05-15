"""LoRA fine-tune of Gemma 4 on ToolBench session triples with Unsloth.

Training data
-------------
Sessions are built from G2/G3 training conversations (see
`data.loader.iter_train_sessions`). Each example is a triple

    (history, query, positive_api)  + N hard negatives

We format it as an instruction-tuning sample:

    <bos>You are a tool-selection assistant.
    Previously used APIs: <a₁, a₂, …>
    Current query: <q>
    Choose the most relevant API from: <candidates>
    Answer: <gold tool/api><eos>

Unsloth handles 2x faster training + 70% less VRAM. On free T4 with
``gemma-4-270m``, one epoch over 5K triples completes in ~30 min.

The resulting LoRA adapter is saved to ``runs/gemma_sessionrerank/``.
At eval time we load it back inside `GemmaPromptedReranker` (same
class, just point ``model_name`` at the merged adapter path).

This script is designed to be called from Colab. It runs unchanged on
A100 / L4 / T4 by passing the appropriate batch size.
"""
from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..data.loader import (
    load_eval_split, build_corpus, iter_train_sessions, _norm,
)


SYSTEM = "You are a tool-selection assistant for an API-calling agent."

TEMPLATE = (
    "{system}\n"
    "Previously used APIs in this session: {history}\n"
    "Current user query: {query}\n"
    "Candidate APIs:\n{candidates}\n"
    "Pick the SINGLE most relevant API. Answer with just `tool / api`.\n"
    "Answer: {answer}"
)


@dataclass
class TrainExample:
    history: str
    query: str
    candidates: str
    answer: str

    def render(self) -> str:
        return TEMPLATE.format(
            system=SYSTEM,
            history=self.history or "(none)",
            query=self.query,
            candidates=self.candidates,
            answer=self.answer,
        )


def build_dataset(
    data_root: str,
    n_examples: int = 5000,
    n_negatives: int = 7,
    seed: int = 0,
) -> List[TrainExample]:
    """Mine training triples from ToolBench G2/G3 multi-turn sessions."""
    random.seed(seed)
    corpus = build_corpus(data_root)
    by_uid = {a.uid: a for a in corpus}
    by_cat = {}
    for a in corpus:
        by_cat.setdefault(a.category, []).append(a)

    examples: List[TrainExample] = []
    for sess in iter_train_sessions(data_root, splits=("G2", "G3")):
        history_uids: List[str] = []
        for turn in sess.turns:
            if not turn.gold_apis:
                continue
            gold = by_uid.get(turn.gold_apis[0])
            if gold is None:
                continue

            # Hard negatives: random APIs from the same category
            pool = [a for a in by_cat.get(gold.category, []) if a.uid != gold.uid]
            if len(pool) < n_negatives:
                pool = [a for a in corpus if a.uid != gold.uid]
            negs = random.sample(pool, min(n_negatives, len(pool)))

            cand_list = [gold] + negs
            random.shuffle(cand_list)
            cand_str = "\n".join(f"- {c.tool} / {c.api}: {(c.description or '')[:140]}" for c in cand_list)

            hist_str = ", ".join(
                f"{by_uid[u].tool}/{by_uid[u].api}" for u in history_uids[-5:]
                if u in by_uid
            )

            examples.append(TrainExample(
                history=hist_str,
                query=turn.query,
                candidates=cand_str,
                answer=f"{gold.tool} / {gold.api}",
            ))
            history_uids.append(gold.uid)
            if len(examples) >= n_examples:
                return examples
    return examples


def train(
    data_root: str = "data/toolbench_hf",
    base_model: str = "google/gemma-4-270m-it",
    output_dir: str = "runs/gemma_sessionrerank",
    n_examples: int = 5000,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    epochs: int = 1,
    max_seq_len: int = 1024,
    lora_r: int = 16,
):
    """Run Unsloth LoRA fine-tune. Requires unsloth + trl installed in Colab."""
    # Imports are local so the rest of the package works on CPU sandboxes
    # without unsloth installed.
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print(f"[finetune] building dataset (n={n_examples}) …")
    examples = build_dataset(data_root, n_examples=n_examples)
    print(f"[finetune] built {len(examples)} training examples")

    ds = Dataset.from_list([{"text": e.render()} for e in examples])

    # Patch Gemma 4 tokenizer_config if it ships the v5-style list format.
    from ..baselines.gemma_embed import _maybe_patch_tokenizer_config
    base_model = _maybe_patch_tokenizer_config(base_model)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=None,         # auto
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    cfg = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        warmup_steps=20,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        max_seq_length=max_seq_len,
        report_to="none",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=cfg)
    print("[finetune] starting training …")
    trainer.train()

    print(f"[finetune] saving merged model to {output_dir}/merged")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(f"{output_dir}/merged", tokenizer, save_method="merged_16bit")
    with open(f"{output_dir}/training_meta.json", "w") as f:
        json.dump({
            "base_model": base_model,
            "n_examples": len(examples),
            "epochs": epochs,
            "lr": lr,
            "lora_r": lora_r,
        }, f, indent=2)
    print("[finetune] done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/toolbench_hf")
    ap.add_argument("--base-model", default="google/gemma-4-270m-it")
    ap.add_argument("--output-dir", default="runs/gemma_sessionrerank")
    ap.add_argument("--n-examples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=16)
    args = ap.parse_args()
    train(
        data_root=args.data_root,
        base_model=args.base_model,
        output_dir=args.output_dir,
        n_examples=args.n_examples,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        lora_r=args.lora_r,
    )


if __name__ == "__main__":
    main()
