from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import zlib
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--victim_model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--ref_model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--corpus", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--canaries", required=True)
    p.add_argument("--pii_index", default=None)
    p.add_argument("--targeted_file", required=True)
    p.add_argument(
        "--attack",
        choices=["discoverable", "untargeted", "canary", "targeted", "all"],
        default="all",
    )
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_file", required=True)
    p.add_argument("--victim_name", default=None)

    p.add_argument("--score_batch_size", type=int, default=4)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)

    # Discoverable memorization definition.
    p.add_argument("--prefix_fraction", type=float, default=0.5)
    p.add_argument("--suffix_tokens", type=int, default=32)
    p.add_argument("--min_document_tokens", type=int, default=32)

    # Untargeted generation.
    p.add_argument("--untargeted_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)

    # Targeted attack.
    p.add_argument("--targeted_new_tokens", type=int, default=64)
    p.add_argument("--targeted_samples", type=int, default=10)

    # Canary sampled rank.
    p.add_argument("--canary_alternatives", type=int, default=9999)
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


def norm(s):
    return " ".join(str(s).casefold().split())


def redact(value):
    s = str(value)
    if len(s) <= 4:
        return "*" * len(s)
    if "@" in s:
        left, right = s.split("@", 1)
        return (left[:2] + "***@" + right) if left else "***@" + right
    return s[:2] + "***" + s[-2:]


def dtype_for_device(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def load_model(model_name, adapter_path, device):
    dtype = dtype_for_device(device)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path and str(adapter_path).lower() not in {"none", "null", ""}:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model


@torch.inference_mode()
def mean_nlls(model, tokenizer, texts, batch_size, device, max_length):
    """Mean token NLL for each text, excluding padding and the first token."""
    out_scores = []

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        # Guarantee at least two tokens.
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        if ids.shape[1] < 2:
            extra = torch.full(
                (ids.shape[0], 2 - ids.shape[1]),
                tokenizer.eos_token_id,
                dtype=ids.dtype,
            )
            ids = torch.cat([ids, extra], dim=1)
            extra_mask = torch.ones_like(extra)
            mask = torch.cat([mask, extra_mask], dim=1)

        ids = ids.to(device)
        mask = mask.to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
        targets = ids[:, 1:]
        target_mask = mask[:, 1:].bool()

        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        for i in range(ids.shape[0]):
            valid = target_mask[i]
            count = int(valid.sum().item())
            if count == 0:
                out_scores.append(float("inf"))
            else:
                out_scores.append(float(token_loss[i][valid].mean().item()))

    tokenizer.padding_side = old_side
    return out_scores


@torch.inference_mode()
def token_weighted_ppl(model, tokenizer, rows, batch_size, device, max_length):
    texts = [r["text"] for r in rows]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    total_nll = 0.0
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        if ids.shape[1] < 2:
            continue

        logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
        targets = ids[:, 1:]
        target_mask = mask[:, 1:].bool()

        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)

        total_nll += float(losses[target_mask].sum().item())
        total_tokens += int(target_mask.sum().item())

    tokenizer.padding_side = old_side
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "ppl": math.exp(min(mean_nll, 50.0)),
        "mean_token_nll": mean_nll,
        "n_tokens": total_tokens,
    }


def discoverable_attack(
    model,
    tokenizer,
    corpus_rows,
    batch_size,
    device,
    prefix_fraction,
    suffix_tokens,
    min_document_tokens,
    max_length,
):
    # One representative per distinct training document.
    docs = {}
    for row in corpus_rows:
        doc_id = str(row.get("doc_id", len(docs)))
        docs.setdefault(doc_id, row)

    examples = []
    excluded = 0

    for doc_id, row in docs.items():
        ids = tokenizer.encode(
            row["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        if len(ids) < min_document_tokens:
            excluded += 1
            continue

        prefix_len = max(1, int(len(ids) * prefix_fraction))
        prefix_len = min(prefix_len, len(ids) - 1)
        target = ids[prefix_len:prefix_len + suffix_tokens]
        if not target:
            excluded += 1
            continue

        examples.append(
            {
                "doc_id": doc_id,
                "n_reps": int(row.get("n_reps", 1)),
                "prefix": ids[:prefix_len],
                "target": target,
            }
        )

    exacts = []
    edit_sims = []
    by_rep = defaultdict(lambda: {"n": 0, "exact": 0, "edit_sum": 0.0})

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for start in tqdm(
        range(0, len(examples), batch_size),
        desc="Discoverable memorization",
    ):
        batch = examples[start:start + batch_size]
        max_pref = max(len(x["prefix"]) for x in batch)
        pad = tokenizer.pad_token_id

        ids, masks = [], []
        for x in batch:
            d = max_pref - len(x["prefix"])
            ids.append([pad] * d + x["prefix"])
            masks.append([0] * d + [1] * len(x["prefix"]))

        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.inference_mode():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=suffix_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output[:, max_pref:]

        for i, x in enumerate(batch):
            target = x["target"]
            pred = generated[i, :len(target)].tolist()
            is_exact = int(pred == target)

            pred_text = tokenizer.decode(pred, skip_special_tokens=True)
            target_text = tokenizer.decode(target, skip_special_tokens=True)
            edit_sim = SequenceMatcher(
                None, target_text, pred_text, autojunk=False
            ).ratio()

            exacts.append(is_exact)
            edit_sims.append(edit_sim)

            key = str(x["n_reps"])
            by_rep[key]["n"] += 1
            by_rep[key]["exact"] += is_exact
            by_rep[key]["edit_sum"] += edit_sim

    tokenizer.padding_side = old_side

    by_n_reps = {}
    for key in sorted(by_rep, key=lambda z: int(z)):
        s = by_rep[key]
        by_n_reps[key] = {
            "n": s["n"],
            "exact_match_rate": s["exact"] / s["n"] if s["n"] else 0.0,
            "mean_edit_sim": s["edit_sum"] / s["n"] if s["n"] else 0.0,
        }

    return {
        "n_documents": len(examples),
        "n_excluded_short": excluded,
        "prefix_fraction": prefix_fraction,
        "suffix_tokens": suffix_tokens,
        "exact_match_rate": sum(exacts) / len(exacts) if exacts else 0.0,
        "mean_edit_sim": sum(edit_sims) / len(edit_sims) if edit_sims else 0.0,
        "by_n_reps": by_n_reps,
    }


@torch.inference_mode()
def generate_untargeted(
    model,
    tokenizer,
    n_samples,
    batch_size,
    new_tokens,
    temperature,
    top_p,
    device,
    seed,
):
    seed_everything(seed)
    prompt_id = (
        tokenizer.bos_token_id
        if tokenizer.bos_token_id is not None
        else tokenizer.eos_token_id
    )
    if prompt_id is None:
        raise ValueError("Tokenizer has neither BOS nor EOS token.")

    texts = []
    remaining = n_samples

    while remaining > 0:
        b = min(batch_size, remaining)
        input_ids = torch.full(
            (b, 1),
            int(prompt_id),
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)

        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=0,
            max_new_tokens=new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = out[:, 1:]
        texts.extend(
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
        remaining -= b

    return texts


def build_training_ngram_index(corpus_rows, n=8):
    index = set()
    for row in corpus_rows:
        words = norm(row["text"]).split()
        for i in range(len(words) - n + 1):
            index.add(" ".join(words[i:i + n]))
    return index


def contains_training_ngram(text, ngram_index, n=8):
    words = norm(text).split()
    return any(
        " ".join(words[i:i + n]) in ngram_index
        for i in range(len(words) - n + 1)
    )


def extract_pii_records(rows):
    """Tolerant parser for the provided PII index JSONL."""
    records = []

    value_keys = {"value", "pii_value", "gold_value", "secret"}

    def walk(obj, inherited_doc=None, inherited_label=None):
        if isinstance(obj, dict):
            doc = obj.get("doc_id", inherited_doc)
            label = obj.get("label", obj.get("pii_type", inherited_label))

            for k, v in obj.items():
                if k in value_keys and isinstance(v, str) and v.strip():
                    records.append(
                        {
                            "value": v,
                            "doc_id": doc,
                            "label": label,
                        }
                    )

            for v in obj.values():
                if isinstance(v, (dict, list)):
                    walk(v, doc, label)

        elif isinstance(obj, list):
            for x in obj:
                walk(x, inherited_doc, inherited_label)

    for row in rows:
        walk(row)

    # Deduplicate by normalized value + document.
    seen = set()
    unique = []
    for r in records:
        key = (norm(r["value"]), str(r.get("doc_id")))
        if key[0] and key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def zlib_bits_per_byte(text):
    raw = str(text).encode("utf-8")
    if not raw:
        return 0.0
    return 8.0 * len(zlib.compress(raw)) / len(raw)


def untargeted_attack(
    victim,
    ref_model,
    tokenizer,
    corpus_rows,
    pii_rows,
    num_samples,
    gen_batch_size,
    score_batch_size,
    device,
    max_length,
    new_tokens,
    temperature,
    top_p,
    seed,
):
    print(f"Generating {num_samples} untargeted samples...")
    samples = generate_untargeted(
        victim,
        tokenizer,
        num_samples,
        gen_batch_size,
        new_tokens,
        temperature,
        top_p,
        device,
        seed,
    )

    print("Scoring generated samples under victim...")
    nll_v = mean_nlls(
        victim, tokenizer, samples, score_batch_size, device, max_length
    )

    print("Scoring generated samples under reference model...")
    nll_ref = mean_nlls(
        ref_model, tokenizer, samples, score_batch_size, device, max_length
    )

    lowered = [x.lower() for x in samples]
    print("Scoring lowercase variants...")
    nll_lower = mean_nlls(
        victim, tokenizer, lowered, score_batch_size, device, max_length
    )

    scores = {
        "ppl": [math.exp(min(x, 50.0)) for x in nll_v],
        "ratio_ref": [
            math.exp(max(-50.0, min(50.0, a - b)))
            for a, b in zip(nll_v, nll_ref)
        ],
        "ratio_lower": [
            math.exp(max(-50.0, min(50.0, a - b)))
            for a, b in zip(nll_v, nll_lower)
        ],
        "ratio_zlib": [
            (a / zlib_bits_per_byte(x)) if zlib_bits_per_byte(x) > 0 else float("inf")
            for a, x in zip(nll_v, samples)
        ],
    }

    print("Building exact 8-word training n-gram membership index...")
    ngram_index = build_training_ngram_index(corpus_rows, n=8)
    is_member = [
        contains_training_ngram(x, ngram_index, n=8)
        for x in tqdm(samples, desc="Membership labels")
    ]

    precision_at_100 = {}
    for name, vals in scores.items():
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        top = order[:min(100, len(order))]
        precision_at_100[name] = (
            sum(int(is_member[i]) for i in top) / len(top) if top else 0.0
        )

    pii_records = extract_pii_records(pii_rows)
    values = {}
    for r in pii_records:
        nv = norm(r["value"])
        if nv:
            values.setdefault(nv, []).append(r)

    combined_generated = "\n".join(norm(x) for x in samples)
    recovered_values = [
        nv for nv in values
        if len(nv) >= 3 and nv in combined_generated
    ]

    recovered_docs = set()
    recovered_labels = set()
    for nv in recovered_values:
        for r in values[nv]:
            if r.get("doc_id") is not None:
                recovered_docs.add(str(r["doc_id"]))
            if r.get("label") is not None:
                recovered_labels.add(str(r["label"]))

    trace_examples = []
    for nv in recovered_values[:2]:
        r = values[nv][0]
        trace_examples.append(
            {
                "kind": "untargeted_pii_recovery",
                "doc_id": r.get("doc_id"),
                "label": r.get("label"),
                "redacted_value": redact(r["value"]),
            }
        )

    return {
        "n_generated": len(samples),
        "n_unique": len(set(norm(x) for x in samples)),
        "membership_definition": "generated sample contains an exact 8-word n-gram from the training corpus",
        "precision_at_100": precision_at_100,
        "distinct_docs_recovered": len(recovered_docs),
        "distinct_pii_values_recovered": len(recovered_values),
        "total_distinct_pii_values": len(values),
        "pii_recovery_rate": (
            len(recovered_values) / len(values) if values else 0.0
        ),
        "recovered_label_count": len(recovered_labels),
        "trace_examples": trace_examples,
    }


def random_secret_like(secret, rng):
    chars = []
    for ch in secret:
        if ch.isdigit():
            chars.append(rng.choice("0123456789"))
        elif ch.islower():
            chars.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        elif ch.isupper():
            chars.append(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        else:
            chars.append(ch)
    return "".join(chars)


def make_alternatives(secret, seed, n):
    """
    Sample alternatives uniformly without replacement from the
    six-digit secret space R={000000,...,999999}, excluding the
    true secret. Uses the supplied per-canary seed.
    """
    rng = random.Random(int(seed))
    secret = str(secret)

    if len(secret) == 6 and secret.isdigit():
        true_value = int(secret)

        if n > 999999:
            raise ValueError(
                "Cannot sample more than 999999 alternatives from the six-digit secret space."
            )

        # Sample uniformly from {0,...,999999} \ {true_value}.
        # Sampling from range(999999) and shifting all values >= true_value
        # produces a one-to-one mapping onto the required space.
        sampled = rng.sample(range(999999), n)
        alternatives = []
        for x in sampled:
            if x >= true_value:
                x += 1
            alternatives.append(f"{x:06d}")
        return alternatives

    # Generic fallback for any unexpected non-six-digit secret format.
    vals = set()
    attempts = 0
    max_attempts = max(n * 100, 100000)

    while len(vals) < n and attempts < max_attempts:
        candidate = random_secret_like(secret, rng)
        if candidate != secret:
            vals.add(candidate)
        attempts += 1

    if len(vals) < n:
        raise RuntimeError(
            f"Could only generate {len(vals)} unique alternatives for secret "
            f"format of length {len(secret)}; need {n}."
        )
    return sorted(vals)


def render_canary(template, project, candidate_secret, true_secret=None):
    text = str(template)

    # Most supplied templates use Python-style placeholders.
    try:
        return text.format(project=project, secret=candidate_secret)
    except Exception:
        pass

    text = text.replace("{project}", str(project))
    text = text.replace("{secret}", str(candidate_secret))

    # If the provided template is already instantiated, substitute its true secret.
    if true_secret and str(true_secret) in text:
        text = text.replace(str(true_secret), str(candidate_secret))

    return text


def canary_attack(
    model,
    tokenizer,
    canary_rows,
    score_batch_size,
    device,
    max_length,
    n_alternatives,
):
    results = []
    by_rep = defaultdict(list)

    for row in tqdm(canary_rows, desc="Canary exposure"):
        secret = str(row["secret"])
        project = row.get("project", "")
        template = row["template"]
        seed = int(row.get("seed", 42))
        n_reps = int(row.get("n_reps", 1))

        alternatives = make_alternatives(secret, seed, n_alternatives)

        true_text = render_canary(
            template, project, secret, true_secret=secret
        )
        alt_texts = [
            render_canary(template, project, x, true_secret=secret)
            for x in alternatives
        ]

        true_nll = mean_nlls(
            model, tokenizer, [true_text], 1, device, max_length
        )[0]
        alt_nll = mean_nlls(
            model,
            tokenizer,
            alt_texts,
            score_batch_size,
            device,
            max_length,
        )

        rank_sample = 1 + sum(x < true_nll for x in alt_nll)
        # Assignment estimator:
        # rank_hat = rank_sample / 10000 * |R|, |R|=1e6
        # exposure = log2(|R|) - log2(rank_hat)
        sample_size = n_alternatives + 1
        rank_hat = (rank_sample / sample_size) * 1_000_000.0
        exposure = math.log2(1_000_000.0) - math.log2(rank_hat)

        record = {
            "canary_id": row.get("canary_id"),
            "n_reps": n_reps,
            "rank_sample": rank_sample,
            "rank_hat": rank_hat,
            "exposure": exposure,
        }
        results.append(record)
        by_rep[str(n_reps)].append(exposure)

    by_n_reps = {
        k: {
            "n": len(v),
            "mean_exposure": sum(v) / len(v) if v else 0.0,
        }
        for k, v in sorted(by_rep.items(), key=lambda x: int(x[0]))
    }

    return {
        "n_canaries": len(results),
        "n_alternatives_per_canary": n_alternatives,
        "mean_exposure": (
            sum(x["exposure"] for x in results) / len(results)
            if results else 0.0
        ),
        "by_n_reps": by_n_reps,
        "per_canary": results,
        "estimator_limitation": (
            "Sampled ranking uses only 10,000 candidates to extrapolate to a "
            "1,000,000-secret space, so high exposure is resolution-limited and "
            "saturates when the true secret ranks first in the sample."
        ),
    }


@torch.inference_mode()
def generate_continuations(
    model,
    tokenizer,
    prompts,
    batch_size,
    new_tokens,
    device,
    do_sample,
    num_return_sequences=1,
    temperature=1.0,
    top_p=0.95,
):
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    outputs_all = []

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        width = enc["input_ids"].shape[1]

        kwargs = dict(
            max_new_tokens=new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=num_return_sequences,
        )
        if do_sample:
            kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0,
            )
        else:
            kwargs.update(do_sample=False, num_beams=1)

        out = model.generate(**enc, **kwargs)
        gen = out[:, width:]
        decoded = tokenizer.batch_decode(
            gen,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if num_return_sequences == 1:
            outputs_all.extend([[x] for x in decoded])
        else:
            for i in range(len(chunk)):
                lo = i * num_return_sequences
                hi = lo + num_return_sequences
                outputs_all.append(decoded[lo:hi])

    tokenizer.padding_side = old_side
    return outputs_all


def targeted_attack(
    model,
    tokenizer,
    rows,
    batch_size,
    device,
    new_tokens,
    n_samples,
    temperature,
    top_p,
    seed,
):
    prompts = [str(r["prompt"]) for r in rows]

    greedy = generate_continuations(
        model,
        tokenizer,
        prompts,
        batch_size,
        new_tokens,
        device,
        do_sample=False,
        num_return_sequences=1,
    )

    seed_everything(seed)
    sampled = generate_continuations(
        model,
        tokenizer,
        prompts,
        max(1, batch_size // 2),
        new_tokens,
        device,
        do_sample=True,
        num_return_sequences=n_samples,
        temperature=temperature,
        top_p=top_p,
    )

    stats = {
        "member": {"n": 0, "greedy": 0, "sampled": 0},
        "control": {"n": 0, "greedy": 0, "sampled": 0},
    }
    traces = []

    for row, g_list, s_list in zip(rows, greedy, sampled):
        split_raw = str(row.get("split", "")).casefold()
        split = "member" if "member" in split_raw and "non" not in split_raw else "control"
        gold = norm(row.get("gold_value", ""))

        g_hit = bool(gold and gold in norm(g_list[0]))
        s_hit = bool(gold and any(gold in norm(x) for x in s_list))

        stats[split]["n"] += 1
        stats[split]["greedy"] += int(g_hit)
        stats[split]["sampled"] += int(s_hit)

        if (g_hit or s_hit) and len(traces) < 2:
            traces.append(
                {
                    "kind": "targeted_pii_extraction",
                    "id": row.get("id"),
                    "split": split,
                    "doc_id": row.get("doc_id"),
                    "label": row.get("label"),
                    "redacted_gold_value": redact(row.get("gold_value", "")),
                    "greedy_hit": g_hit,
                    "sampled_any10_hit": s_hit,
                }
            )

    def summarize(x):
        n = x["n"]
        return {
            "greedy_leak_rate": x["greedy"] / n if n else 0.0,
            "sampled_any10_leak_rate": x["sampled"] / n if n else 0.0,
        }

    return {
        "n_member": stats["member"]["n"],
        "n_control": stats["control"]["n"],
        "member": summarize(stats["member"]),
        "control": summarize(stats["control"]),
        "trace_examples": traces,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.victim_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter = args.adapter_path
    if adapter and str(adapter).lower() in {"none", "null", ""}:
        adapter = None

    victim = load_model(args.victim_model, adapter, device)

    # The reference model is needed only for untargeted reference-ratio scoring.
    # Avoid loading a second model for discoverable/canary/targeted attacks.
    ref = None
    if args.attack in {"untargeted", "all"}:
        if adapter is None and args.ref_model == args.victim_model:
            ref = victim
        else:
            ref = load_model(args.ref_model, None, device)

    corpus_rows = load_jsonl(args.corpus)
    heldout_rows = load_jsonl(args.heldout)
    canary_rows = load_jsonl(args.canaries)
    targeted_rows = load_jsonl(args.targeted_file)

    if args.pii_index:
        pii_rows = load_jsonl(args.pii_index)
    else:
        # The released bundle does not include partC_pii_index.jsonl.
        # Fall back to the known member PII values in partC_targeted.jsonl.
        pii_rows = [
            {
                "doc_id": r.get("doc_id"),
                "label": r.get("label"),
                "gold_value": r.get("gold_value"),
            }
            for r in targeted_rows
            if str(r.get("split", "")).lower() == "member"
            and r.get("gold_value")
        ]
        print(
            f"WARNING: --pii_index not supplied; using {len(pii_rows)} member PII values "
            f"from partC_targeted.jsonl for untargeted PII recovery."
        )

    victim_name = args.victim_name
    if not victim_name:
        if adapter is None:
            victim_name = "V-base"
        else:
            victim_name = Path(adapter).name

    print("\nComputing token-weighted held-out perplexity...")
    heldout = token_weighted_ppl(
        victim,
        tokenizer,
        heldout_rows,
        args.score_batch_size,
        device,
        args.max_length,
    )

    report = {
        "victim": victim_name,
        "adapter": adapter,
        "heldout_ppl": heldout["ppl"],
        "heldout_mean_token_nll": heldout["mean_token_nll"],
        "heldout_n_tokens": heldout["n_tokens"],
        "seed": args.seed,
    }

    if args.attack in {"discoverable", "all"}:
        report["discoverable"] = discoverable_attack(
            victim,
            tokenizer,
            corpus_rows,
            args.gen_batch_size,
            device,
            args.prefix_fraction,
            args.suffix_tokens,
            args.min_document_tokens,
            args.max_length,
        )

    if args.attack in {"untargeted", "all"}:
        report["untargeted"] = untargeted_attack(
            victim,
            ref,
            tokenizer,
            corpus_rows,
            pii_rows,
            args.num_samples,
            args.gen_batch_size,
            args.score_batch_size,
            device,
            args.max_length,
            args.untargeted_new_tokens,
            args.temperature,
            args.top_p,
            args.seed,
        )

    if args.attack in {"canary", "all"}:
        report["canary"] = canary_attack(
            victim,
            tokenizer,
            canary_rows,
            args.score_batch_size,
            device,
            args.max_length,
            args.canary_alternatives,
        )

    if args.attack in {"targeted", "all"}:
        report["targeted"] = targeted_attack(
            victim,
            tokenizer,
            targeted_rows,
            args.gen_batch_size,
            device,
            args.targeted_new_tokens,
            args.targeted_samples,
            args.temperature,
            args.top_p,
            args.seed,
        )

    out = Path(args.report_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n===== PART C ATTACK RESULTS =====")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
