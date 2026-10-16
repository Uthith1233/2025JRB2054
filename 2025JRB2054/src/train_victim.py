from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--corpus", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class CausalCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for x in features:
            seq = list(x["input_ids"])
            pad = max_len - len(seq)
            ids.append(seq + [self.pad_id] * pad)
            masks.append([1] * len(seq) + [0] * pad)
            labels.append(seq + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def tokenize_rows(rows, tokenizer, max_length):
    encoded = []
    for row in rows:
        text = row["text"]
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1,
        )
        if tokenizer.eos_token_id is not None:
            ids = ids + [tokenizer.eos_token_id]
        if len(ids) < 2:
            continue
        encoded.append(ids)
    return encoded


@torch.inference_mode()
def token_weighted_ppl(model, token_lists, collator, batch_size, device):
    model.eval()
    loader = DataLoader(
        [{"input_ids": x} for x in token_lists],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    total_nll = 0.0
    total_tokens = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].contiguous()
        targets = labels[:, 1:].contiguous()

        loss_sum = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = int((targets != -100).sum().item())

        total_nll += float(loss_sum.item())
        total_tokens += n_tokens

    mean_nll = total_nll / max(total_tokens, 1)
    ppl = math.exp(min(mean_nll, 50.0))
    return {
        "token_weighted_ppl": ppl,
        "mean_token_nll": mean_nll,
        "n_scored_tokens": total_tokens,
    }


def main():
    args = parse_args()
    if not (1 <= args.epochs <= 10):
        raise ValueError("--epochs must be in [1,10].")
    seed_everything(args.seed)

    train_rows = load_jsonl(args.corpus)
    heldout_rows = load_jsonl(args.heldout)

    if len(train_rows) != 8007:
        print(f"WARNING: assignment corpus is expected to contain 8007 rows; found {len(train_rows)}.")
    if len(heldout_rows) != 500:
        print(f"WARNING: assignment heldout is expected to contain 500 rows; found {len(heldout_rows)}.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ids = tokenize_rows(train_rows, tokenizer, args.max_length)
    heldout_ids = tokenize_rows(heldout_rows, tokenizer, args.max_length)

    train_ds = Dataset.from_dict({"input_ids": train_ids})
    collator = CausalCollator(tokenizer.pad_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    base.eval()

    print("\nComputing untouched base held-out perplexity...")
    base_ppl = token_weighted_ppl(
        base, heldout_ids, collator, args.eval_batch_size, device
    )
    print("Base held-out PPL:", base_ppl["token_weighted_ppl"])

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    targs = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_bf16,
        fp16=(device.type == "cuda" and not use_bf16),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\nStarting victim training...")
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0

    peak_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    model.config.use_cache = True
    model.eval()

    print("\nComputing trained victim held-out perplexity...")
    trained_ppl = token_weighted_ppl(
        model, heldout_ids, collator, args.eval_batch_size, device
    )
    print("Trained held-out PPL:", trained_ppl["token_weighted_ppl"])

    metrics = {
        "model": args.model,
        "corpus": args.corpus,
        "heldout": args.heldout,
        "n_train_rows": len(train_rows),
        "n_heldout_rows": len(heldout_rows),
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "train_wallclock_s": elapsed,
        "peak_gpu_mem_gb": peak_gb,
        "trainer_train_loss": float(result.training_loss),
        "base_heldout": base_ppl,
        "trained_heldout": trained_ppl,
    }

    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== VICTIM TRAINING COMPLETE =====")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
