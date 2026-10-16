from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


SYSTEM_PROMPT = (
    "You are a PII sanitization system. You will be given a piece of text, which may be "
    "plain prose or a structured document (e.g. JSON). Rewrite it by replacing every span "
    "of personally identifiable information (PII) with a placeholder in the exact form "
    "[LABEL], where LABEL is the PII category (e.g. [EMAIL], [TIME], [LASTNAME1]). "
    "Preserve everything else exactly as-is: all non-PII text, structure, formatting, "
    "whitespace, and any bracketed placeholder tokens already present in the input "
    "(these are not PII and must be left unchanged). Output only the rewritten text, "
    "nothing else — no explanations, no additional formatting."
)

PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")

REGEX_RULES = [
    ("EMAIL", re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)),
    ("URL", re.compile(r"\b(?:https?://|www\.)[^\s<>\]\)]+", re.I)),
    ("IP", re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")),
    ("CREDITCARDNUMBER", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("TIME", re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?[APap][Mm])?\b|\b(?:1[0-2]|0?[1-9])\s?[APap][Mm]\b")),
    ("DATE", re.compile(
        r"\b(?:"
        r"(?:0?[1-9]|[12]\d|3[01])[/-](?:0?[1-9]|1[0-2])[/-](?:19|20)\d{2}"
        r"|(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])"
        r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)(?:19|20)\d{2}"
        r")\b", re.I
    )),
    ("TELEPHONENUM", re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]\d{3,4}(?!\d)")),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--test_file", required=True)
    p.add_argument(
        "--system",
        required=True,
        choices=["finetuned", "regex", "presidio", "zeroshot"],
    )
    p.add_argument("--output_predictions", required=True)
    p.add_argument("--report_file", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_input_tokens", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug_rows", type=int, default=None)
    args = p.parse_args()

    if args.system == "finetuned" and not args.adapter_path:
        p.error("--adapter_path is required when --system finetuned")
    if args.batch_size < 1:
        p.error("--batch_size must be >= 1")
    return args


def load_jsonl(path, max_rows=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_rows is not None and len(rows) >= max_rows:
                    break
    return rows


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_everything(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(tokenizer, source_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_generation_model(base_model, adapter_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    model = model.to(device)
    model.eval()
    model.config.use_cache = True

    # Explicitly remove sampling defaults: Part B must be deterministic.
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )

    return model, tokenizer, device, generation_config


@torch.inference_mode()
def generate_model_predictions(
    rows,
    base_model,
    adapter_path,
    batch_size,
    max_input_tokens,
    max_new_tokens,
):
    model, tokenizer, device, gen_cfg = load_generation_model(
        base_model, adapter_path
    )
    gen_cfg.max_new_tokens = max_new_tokens

    predictions = []
    latencies_ms = []
    truncation_count = 0

    for start in tqdm(range(0, len(rows), batch_size), desc="Generating"):
        batch = rows[start:start + batch_size]
        prompts = [build_prompt(tokenizer, row["source_text"]) for row in batch]

        # Count prompts that will be truncated.
        lengths = tokenizer(
            prompts,
            add_special_tokens=False,
            return_length=True,
            truncation=False,
        )["length"]
        truncation_count += sum(x > max_input_tokens for x in lengths)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model.generate(
            **inputs,
            generation_config=gen_cfg,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        prompt_width = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, prompt_width:]
        texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        predictions.extend(texts)
        latencies_ms.extend([elapsed_ms / len(batch)] * len(batch))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions, latencies_ms, truncation_count


def apply_regex_baseline(text):
    candidates = []
    for priority, (label, pattern) in enumerate(REGEX_RULES):
        for m in pattern.finditer(text):
            if m.end() <= m.start():
                continue
            candidates.append((m.start(), m.end(), priority, label))

    # Prefer earlier spans, then longer spans, then rule priority.
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[2]))

    selected = []
    for start, end, priority, label in candidates:
        if any(not (end <= s or start >= e) for s, e, _, _ in selected):
            continue
        selected.append((start, end, priority, label))

    out = text
    for start, end, _, label in sorted(selected, reverse=True):
        out = out[:start] + f"[{label}]" + out[end:]
    return out


def generate_regex_predictions(rows):
    preds, latencies = [], []
    for row in tqdm(rows, desc="Regex"):
        t0 = time.perf_counter()
        pred = apply_regex_baseline(row["source_text"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(pred)
    return preds, latencies


PRESIDIO_LABEL_MAP = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "TELEPHONENUM",
    "IP_ADDRESS": "IP",
    "URL": "URL",
    "CREDIT_CARD": "CREDITCARDNUMBER",
    "DATE_TIME": "DATE",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "IBAN_CODE": "IBAN",
    "US_SSN": "SSN",
    "US_PASSPORT": "PASSPORT",
}


def generate_presidio_predictions(rows):
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as e:
        raise RuntimeError(
            "Presidio is not installed. Run: pip install presidio-analyzer presidio-anonymizer"
        ) from e

    analyzer = AnalyzerEngine()
    preds, latencies = [], []

    for row in tqdm(rows, desc="Presidio"):
        text = row["source_text"]
        t0 = time.perf_counter()
        results = analyzer.analyze(text=text, language="en")

        spans = []
        for r in results:
            label = PRESIDIO_LABEL_MAP.get(r.entity_type, r.entity_type)
            spans.append((int(r.start), int(r.end), float(r.score), label))

        # Keep highest-confidence non-overlapping spans.
        spans.sort(key=lambda x: (-x[2], x[0], -(x[1] - x[0])))
        selected = []
        for start, end, score, label in spans:
            if any(not (end <= s or start >= e) for s, e, _, _ in selected):
                continue
            selected.append((start, end, score, label))

        out = text
        for start, end, _, label in sorted(selected, reverse=True):
            out = out[:start] + f"[{label}]" + out[end:]

        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(out)

    return preds, latencies


def output_pieces(text):
    pieces = []
    current = 0
    for match in PLACEHOLDER_RE.finditer(text):
        if match.start() > current:
            pieces.append(
                ("text", text[current:match.start()], None, (current, match.start()))
            )
        pieces.append(
            ("placeholder", match.group(0), match.group(1), (match.start(), match.end()))
        )
        current = match.end()
    if current < len(text):
        pieces.append(("text", text[current:], None, (current, len(text))))
    return pieces


def recover_predicted_spans(source_text, predicted_text):
    """
    Left-to-right anchor alignment.

    Literal output text is aligned back to the source. Newly generated [LABEL]
    placeholders consume the source substring between the previous aligned
    literal and the next literal anchor. A placeholder already present verbatim
    in the source at the current position is preserved, not counted as PII.

    failure_count records places where literal anchors could not be matched
    exactly at the expected source position.
    """
    pieces = output_pieces(predicted_text)
    predicted_spans = []
    inserted_ranges = []
    source_pos = 0
    failures = 0

    def next_nonempty_literal(index):
        for j in range(index + 1, len(pieces)):
            if pieces[j][0] == "text" and pieces[j][1]:
                return pieces[j][1]
        return None

    for i, (kind, raw, label, output_range) in enumerate(pieces):
        if kind == "text":
            if source_text.startswith(raw, source_pos):
                source_pos += len(raw)
            else:
                pos = source_text.find(raw, source_pos)
                if pos >= 0:
                    failures += 1
                    source_pos = pos + len(raw)
                else:
                    # Do not invent a new source position if the model rewrote
                    # literal text; keep going and record the alignment failure.
                    failures += 1
            continue

        # Existing placeholder in source: preserve it, do not call it a prediction.
        if source_text.startswith(raw, source_pos):
            source_pos += len(raw)
            continue

        anchor = next_nonempty_literal(i)
        if anchor is None:
            end = len(source_text)
        else:
            end = source_text.find(anchor, source_pos)
            if end < 0:
                failures += 1
                end = len(source_text)

        if end >= source_pos:
            predicted_spans.append(
                {"start": int(source_pos), "end": int(end), "label": str(label)}
            )
            inserted_ranges.append(output_range)
            source_pos = end

    return predicted_spans, inserted_ranges, failures


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def span_iou(a, b):
    intersection = max(
        0,
        min(a["end"], b["end"]) - max(a["start"], b["start"]),
    )
    if intersection <= 0:
        return 0.0
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return intersection / union if union else 0.0


def exact_tp(predicted, gold, label):
    pred_counter = Counter(
        (x["start"], x["end"], x["label"])
        for x in predicted
        if x["label"] == label
    )
    gold_counter = Counter(
        (x["start"], x["end"], x["label"])
        for x in gold
        if x["label"] == label
    )
    return sum((pred_counter & gold_counter).values())


def maximum_iou_tp(predicted, gold, label, threshold=0.5):
    """
    One-to-one maximum-IoU matching for one label.

    We maximize total IoU over valid pairs (IoU >= threshold). If two
    assignments have the same total IoU, prefer the one with more matches.
    """
    p = [x for x in predicted if x["label"] == label]
    g = [x for x in gold if x["label"] == label]

    if not p or not g:
        return 0

    # Keep the bitmask on the smaller side.
    if len(g) <= len(p):
        left, right = p, g
    else:
        left, right = g, p

    # Same-label entity counts are normally tiny. Fallback keeps runtime bounded
    # on pathological documents.
    if len(right) > 20:
        edges = []
        for i, a in enumerate(left):
            for j, b in enumerate(right):
                score = span_iou(a, b)
                if score >= threshold:
                    edges.append((score, i, j))
        edges.sort(reverse=True)
        used_l, used_r = set(), set()
        count = 0
        for score, i, j in edges:
            if i not in used_l and j not in used_r:
                used_l.add(i)
                used_r.add(j)
                count += 1
        return count

    @lru_cache(maxsize=None)
    def dp(i, used_mask):
        if i == len(left):
            return (0.0, 0)

        best = dp(i + 1, used_mask)  # skip this entity

        for j, b in enumerate(right):
            if used_mask & (1 << j):
                continue
            score = span_iou(left[i], b)
            if score < threshold:
                continue

            rest_score, rest_count = dp(i + 1, used_mask | (1 << j))
            candidate = (rest_score + score, rest_count + 1)

            if candidate[0] > best[0] + 1e-12:
                best = candidate
            elif abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] > best[1]:
                best = candidate

        return best

    return dp(0, 0)[1]


def calculate_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def protected_prediction(predicted_text, inserted_ranges):
    chars = list(predicted_text)
    for start, end in inserted_ranges:
        for i in range(start, min(end, len(chars))):
            chars[i] = "\ue000"
    return "".join(chars)


def calculate_overmasking(source_text, predicted_text, gold_spans, inserted_ranges):
    pii_mask = [False] * len(source_text)

    for span in gold_spans:
        start = max(0, min(len(source_text), int(span["start"])))
        end = max(start, min(len(source_text), int(span["end"])))
        for i in range(start, end):
            pii_mask[i] = True

    total_non_pii = sum(not x for x in pii_mask)
    if total_non_pii == 0:
        return 0, 0

    safe_prediction = protected_prediction(predicted_text, inserted_ranges)
    matcher = SequenceMatcher(None, source_text, safe_prediction, autojunk=False)
    preserved = set()

    for block in matcher.get_matching_blocks():
        for i in range(block.a, block.a + block.size):
            if i < len(pii_mask) and not pii_mask[i]:
                preserved.add(i)

    return total_non_pii - len(preserved), total_non_pii


def summarize_counts(counts):
    total_tp = sum(x["tp"] for x in counts.values())
    total_fp = sum(x["fp"] for x in counts.values())
    total_fn = sum(x["fn"] for x in counts.values())

    micro_p, micro_r, micro_f1 = calculate_prf(total_tp, total_fp, total_fn)

    per_class = {}
    macro_p_values, macro_r_values, macro_f1_values = [], [], []

    for label, stat in sorted(counts.items()):
        p, r, f1 = calculate_prf(stat["tp"], stat["fp"], stat["fn"])
        per_class[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": stat["support"],
            "tp": stat["tp"],
            "fp": stat["fp"],
            "fn": stat["fn"],
        }
        if stat["support"] > 0:
            macro_p_values.append(p)
            macro_r_values.append(r)
            macro_f1_values.append(f1)

    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_precision": (
            sum(macro_p_values) / len(macro_p_values) if macro_p_values else 0.0
        ),
        "macro_recall": (
            sum(macro_r_values) / len(macro_r_values) if macro_r_values else 0.0
        ),
        "macro_f1": (
            sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0
        ),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class,
    }


def evaluate_predictions(rows, predictions, latencies_ms):
    exact_counts = {}
    overlap_counts = {}
    leaked_values = 0
    total_gold_values = 0
    changed_non_pii = 0
    total_non_pii = 0
    total_alignment_failures = 0
    prediction_records = []

    for row, predicted_text, latency_ms in zip(rows, predictions, latencies_ms):
        gold_spans = [
            {
                "start": int(x["start"]),
                "end": int(x["end"]),
                "label": str(x["label"]),
            }
            for x in row.get("privacy_mask", [])
        ]

        predicted_spans, inserted_ranges, failures = recover_predicted_spans(
            row["source_text"], predicted_text
        )
        total_alignment_failures += failures

        labels_here = (
            {x["label"] for x in gold_spans}
            | {x["label"] for x in predicted_spans}
        )

        for label in labels_here:
            exact_counts.setdefault(
                label, {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            )
            overlap_counts.setdefault(
                label, {"tp": 0, "fp": 0, "fn": 0, "support": 0}
            )

            n_pred = sum(x["label"] == label for x in predicted_spans)
            n_gold = sum(x["label"] == label for x in gold_spans)

            tp_exact = exact_tp(predicted_spans, gold_spans, label)
            tp_overlap = maximum_iou_tp(predicted_spans, gold_spans, label)

            exact_counts[label]["tp"] += tp_exact
            exact_counts[label]["fp"] += n_pred - tp_exact
            exact_counts[label]["fn"] += n_gold - tp_exact
            exact_counts[label]["support"] += n_gold

            overlap_counts[label]["tp"] += tp_overlap
            overlap_counts[label]["fp"] += n_pred - tp_overlap
            overlap_counts[label]["fn"] += n_gold - tp_overlap
            overlap_counts[label]["support"] += n_gold

        normalized_prediction = normalize_text(predicted_text)
        leaked_for_row = []

        for pii in row.get("privacy_mask", []):
            value = normalize_text(pii.get("value", ""))
            if not value:
                continue
            total_gold_values += 1
            if value in normalized_prediction:
                leaked_values += 1
                leaked_for_row.append(pii.get("value", ""))

        changed, total = calculate_overmasking(
            row["source_text"],
            predicted_text,
            gold_spans,
            inserted_ranges,
        )
        changed_non_pii += changed
        total_non_pii += total

        prediction_records.append(
            {
                "id": row.get("id"),
                "language": row.get("language"),
                "source_text": row["source_text"],
                "predicted_text": predicted_text,
                "predicted_spans": predicted_spans,
                "gold_spans": gold_spans,
                "leaked_values": leaked_for_row,
                "alignment_failures": failures,
                "latency_ms": latency_ms,
            }
        )

    exact = summarize_counts(exact_counts)
    overlap = summarize_counts(overlap_counts)

    supports = {
        label: stat["support"]
        for label, stat in exact_counts.items()
    }
    top10_labels = [
        label
        for label, _ in sorted(
            supports.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
    ]

    top10_per_class = {
        label: {
            "support": supports[label],
            "exact": exact["per_class"].get(label, {}),
            "overlap_iou50": overlap["per_class"].get(label, {}),
        }
        for label in top10_labels
    }

    # Simple error taxonomy for the report.
    boundary_or_near_miss = max(0, overlap["tp"] - exact["tp"])
    error_taxonomy = {
        "missed_gold_entities_iou50": overlap["fn"],
        "spurious_predictions_iou50": overlap["fp"],
        "boundary_or_near_miss_entities": boundary_or_near_miss,
        "leaked_gold_values": leaked_values,
        "alignment_failures": total_alignment_failures,
    }

    metrics = {
        "n": len(rows),
        "exact": {k: v for k, v in exact.items() if k != "per_class"},
        "overlap_iou50": {
            k: v for k, v in overlap.items() if k != "per_class"
        },
        "per_class_top10": top10_per_class,
        "leak_rate": (
            leaked_values / total_gold_values if total_gold_values else 0.0
        ),
        "over_masking_rate": (
            changed_non_pii / total_non_pii if total_non_pii else 0.0
        ),
        "alignment_failure_count": total_alignment_failures,
        "error_taxonomy": error_taxonomy,
        "mean_latency_ms": (
            sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        ),
    }

    return metrics, prediction_records


def main():
    args = parse_args()
    seed_everything(args.seed)

    rows = load_jsonl(args.test_file, args.debug_rows)
    print(f"Loaded {len(rows)} rows from {args.test_file}")
    print(f"System: {args.system}")

    truncation_count = 0

    if args.system == "finetuned":
        predictions, latencies, truncation_count = generate_model_predictions(
            rows=rows,
            base_model=args.base_model,
            adapter_path=args.adapter_path,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )

    elif args.system == "zeroshot":
        predictions, latencies, truncation_count = generate_model_predictions(
            rows=rows,
            base_model=args.base_model,
            adapter_path=None,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
        )

    elif args.system == "regex":
        predictions, latencies = generate_regex_predictions(rows)

    elif args.system == "presidio":
        predictions, latencies = generate_presidio_predictions(rows)

    else:
        raise ValueError(args.system)

    metrics, records = evaluate_predictions(rows, predictions, latencies)

    report = {
        "system": args.system,
        "adapter": args.adapter_path if args.system == "finetuned" else None,
        "test_set": str(args.test_file),
        "base_model": args.base_model if args.system in {"finetuned", "zeroshot"} else None,
        "seed": args.seed,
        "decoding": (
            {
                "do_sample": False,
                "num_beams": 1,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "truncated_prompt_count": truncation_count,
            }
            if args.system in {"finetuned", "zeroshot"}
            else None
        ),
        **metrics,
    }

    save_jsonl(args.output_predictions, records)

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n===== PART B RESULTS =====")
    print("system:", args.system)
    print("n:", report["n"])
    print("Exact micro P/R/F1:",
          report["exact"]["micro_precision"],
          report["exact"]["micro_recall"],
          report["exact"]["micro_f1"])
    print("Exact macro P/R/F1:",
          report["exact"]["macro_precision"],
          report["exact"]["macro_recall"],
          report["exact"]["macro_f1"])
    print("IoU>=0.5 micro P/R/F1:",
          report["overlap_iou50"]["micro_precision"],
          report["overlap_iou50"]["micro_recall"],
          report["overlap_iou50"]["micro_f1"])
    print("IoU>=0.5 macro P/R/F1:",
          report["overlap_iou50"]["macro_precision"],
          report["overlap_iou50"]["macro_recall"],
          report["overlap_iou50"]["macro_f1"])
    print("Leak rate:", report["leak_rate"])
    print("Over-masking rate:", report["over_masking_rate"])
    print("Alignment failures:", report["alignment_failure_count"])
    print("Mean latency (ms):", report["mean_latency_ms"])
    print("Saved predictions:", args.output_predictions)
    print("Saved report:", args.report_file)


if __name__ == "__main__":
    main()
