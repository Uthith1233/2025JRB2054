from collections import Counter
import json
from pathlib import Path
import argparse
import csv 
import math
import os
import random
import re
import time
from difflib import SequenceMatcher
import numpy as np

import torch 

from datasets import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from torch.utils.data import SequentialSampler

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)


Data_dir = Path("F:/ELL8299_A1/data/trustworthy_llms_a1_bundle/data")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MODULES = [
    "q_proj",
    "v_proj"
]

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


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

PLACEHOLDER_RE = re.compile(
    r"\[([A-Z][A-Z0-9_]*)\]"
)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct"
    )

    parser.add_argument(
        "--train_file",
        required=True
    )

    parser.add_argument(
        "--val_file",
        required=True
    )

    parser.add_argument(
        "--target_modules",
        required=True
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        required=True
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=True
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05
    )

    parser.add_argument(
        "--quant",
        choices=["none", "4bit"],
        default="none"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--run_id",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--results_csv",
        default="results/partA_sweep.csv"
    )

    parser.add_argument(
        "--figure_path",
        default="results/partA_fig.png"
    )

    parser.add_argument(
        "--debug_train_rows",
        type=int,
        default=None
    )

    parser.add_argument(
        "--debug_val_rows", 
        type=int,
        default=None
    )



    # --------------------------------------------------------
    # KEEP THESE IDENTICAL FOR A1-A6
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4
    )

    parser.add_argument(
        "--lr_scheduler_type",
        default="cosine"
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=20
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=1
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512
    )


    args = parser.parse_args()

    if not (1 <= args.epochs <= 10):
        raise ValueError("--epochs must be between 1 and 10.")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup_ratio must be in [0, 1).")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive.")
    if args.gen_batch_size <= 0:
        raise ValueError("--gen_batch_size must be positive.")

    return args

def load_json(path):
    rows = []

    with open(path , "r" , encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            
    
    return rows


def seed_everything(seed):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Keep runs reproducible while warning (rather than crashing) if a CUDA
    # kernel has no deterministic implementation.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_input_ids(tokenized):

    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        tokenized = tokenized["input_ids"]

    if torch.is_tensor(tokenized):
        tokenized = tokenized.tolist()

    if (
        len(tokenized) > 0
        and isinstance(tokenized[0], list)
    ):
        tokenized = tokenized[0]

    return list(tokenized)

def inspect(path , max_length=1024, max_rows=None):
    rows = load_json(path)

    # Debug mode should actually tokenize only the requested rows. In the full
    # assignment run max_rows=None, so every supplied row is used in order.
    if max_rows is not None:
        rows = rows[:max_rows]

    full_lens = []
    labels = []
    all_full_ids = []
    prompt_lens = []

    for i, r in enumerate(rows):
        source_text = r.get("source_text")
        target_text = r.get("target_text")

        if source_text is None or target_text is None:
            raise ValueError(
                f"Missing source_text/target_text in row {i} of {path}."
            )

        prompt_msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source_text},
        ]

        full_msg = prompt_msg + [
            {"role": "assistant", "content": target_text}
        ]

        # Render first, then tokenize prompt and completion separately. This
        # avoids the Qwen chat-template boundary mismatch seen in the debug log
        # while guaranteeing that ONLY assistant-completion tokens get loss.
        prompt_text = tokenizer.apply_chat_template(
            prompt_msg,
            tokenize=False,
            add_generation_prompt=True
        )

        full_text = tokenizer.apply_chat_template(
            full_msg,
            tokenize=False,
            add_generation_prompt=False
        )

        if not full_text.startswith(prompt_text):
            raise RuntimeError(
                "The rendered full conversation does not start with the "
                "rendered system/user generation prompt. Cannot guarantee "
                "completion-only masking with this chat template."
            )

        completion_text = full_text[len(prompt_text):]

        prompt_id = tokenizer(
            prompt_text,
            add_special_tokens=False
        )["input_ids"]
        prompt_id = list(get_input_ids(prompt_id))

        completion_ids = tokenizer(
            completion_text,
            add_special_tokens=False
        )["input_ids"]
        completion_ids = list(get_input_ids(completion_ids))

        full_ids = prompt_id + completion_ids
        prompt_len = len(prompt_id)
        prompt_lens.append(prompt_len)

        if len(full_ids) > max_length:
            raise ValueError(
                f"Row {r.get('id', i)} has {len(full_ids)} tokens, which "
                f"exceeds max_length={max_length}. Increase max_length; "
                "silent truncation is intentionally disabled."
            )

        if len(completion_ids) == 0:
            raise ValueError(
                f"Row {r.get('id', i)} has an empty assistant completion."
            )

        full_lens.append(len(full_ids))
        all_full_ids.append(full_ids)

        # Completion-only loss:
        # system + user + assistant prefix -> -100
        # assistant target + assistant end token -> supervised token IDs
        label = ([-100] * prompt_len) + completion_ids

        if len(label) != len(full_ids):
            raise RuntimeError("input_ids/labels length mismatch.")

        labels.append(label)

    return labels, full_lens, all_full_ids


class Collator:

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):

        max_len = max(len(x["input_ids"]) for x in features)
        
        batch_size = len(features)

        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)

        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        
        for i , example in enumerate(features):
            
            n = len(example["input_ids"])

            input_ids[i, :n] = torch.tensor(example["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, :n] = torch.tensor(example["labels"], dtype=torch.long)

        return {
            "input_ids": input_ids, 
            "attention_mask": attention_mask,
            "labels": labels
        }

class OrderedTrainer(Trainer):

    def _get_train_sampler(self, train_dataset=None):

        ds = (
            train_dataset
            if train_dataset is not None
            else self.train_dataset
        )

        if ds is None:
            return None

        return SequentialSampler(ds)



def build_model(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your environment.")
    
    # Keep A1-A5 behavior unchanged. For A6 (4-bit QLoRA), use the same
    # compute precision that TrainingArguments will use on this GPU.
    dtype = torch.float16

    target_modules = [
        x.strip() for x in args.target_modules.split(",") if x.strip()
    ]

    if not target_modules:
        raise ValueError("No valid target modules were supplied.")

    if args.quant == "4bit":

        # A6 only: match bitsandbytes compute dtype to Trainer precision.
        # On L4/Ampere-or-newer GPUs this is BF16; otherwise it falls back to FP16.
        qlora_compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        print(
            "Using 4-bit QLoRA with compute dtype:",
            "BF16" if qlora_compute_dtype == torch.bfloat16 else "FP16"
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=qlora_compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quant_config,
            device_map={"": torch.cuda.current_device()},
            dtype=qlora_compute_dtype,
            low_cpu_mem_usage=True
        )

        model = prepare_model_for_kbit_training(model , use_gradient_checkpointing=True)
    
    else:
        # Preserve the exact A1-A5 model-loading precision used previously.
        print("Using FP16 precision.")
        
        model = AutoModelForCausalLM.from_pretrained(
             args.model,
             dtype=dtype,
             low_cpu_mem_usage=True
        )

        model = model.to("cuda")

        model.gradient_checkpointing_enable()

        if hasattr(model, "enable_input_require_grads"):
             model.enable_input_require_grads()
    
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Exact PEFT parameter counts. This is especially useful for 4-bit QLoRA,
    # where PEFT knows how to report the logical parameter count correctly.
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable_params, total_params = model.get_nb_trainable_parameters()
    else:
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())

    trainable_params = int(trainable_params)
    total_params = int(total_params)
    trainable_percentage = (trainable_params / total_params) * 100

    return model , trainable_params , total_params , trainable_percentage

@torch.inference_mode()
def generate_validation(
    model,
    tokenizer,
    rows,
    max_new_tokens=512,
    batch_size=1
):

    model.eval()
    predictions = []

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    model.config.use_cache = True

    # Explicit deterministic decoding avoids model-repository generation
    # defaults changing the evaluation behavior.
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda")

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        prompts = []

        for row in batch_rows:
            source_text = row.get("source_text")

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": source_text
                },
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        )
        inputs = {
            key: value.to(model_device)
            for key, value in inputs.items()
        }

        output_ids = model.generate(
            **inputs,
            generation_config=generation_config
        )

        prompt_width = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_width:]

        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        predictions.extend(decoded)

        completed = min(start + len(batch_rows), len(rows))
        if completed % 100 == 0 or completed == len(rows):
            print(f"Generated {completed}/{len(rows)}")

    tokenizer.padding_side = old_padding_side
    model.config.use_cache = False

    return predictions


def output_pieces(text):

    pieces = []
    current = 0

    for match in PLACEHOLDER_RE.finditer(text):

        if match.start() > current:

            pieces.append((
                "text",
                text[current:match.start()],
                None,
                (current, match.start())
            ))

        pieces.append((
            "placeholder",
            match.group(0),
            match.group(1),
            (match.start(), match.end())
        ))

        current = match.end()


    if current < len(text):

        pieces.append((
            "text",
            text[current:],
            None,
            (current, len(text))
        ))

    return pieces

def recover_predicted_spans(source_text, predicted_text):
    pieces = output_pieces(predicted_text)
    predicted_spans = []
    inserted_ranges = []
    source_pos = 0
    alignment_failures = 0

    def next_literal(index):
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
                    alignment_failures += 1
                    source_pos = pos + len(raw)
                else:
                    alignment_failures += 1
            continue

        # A bracketed placeholder already present in source is not newly predicted PII.
        if source_text.startswith(raw, source_pos):
            source_pos += len(raw)
            continue

        anchor = next_literal(i)

        if anchor is None:
            end = len(source_text)
        else:
            end = source_text.find(anchor, source_pos)
            if end < 0:
                alignment_failures += 1
                end = len(source_text)

        predicted_spans.append({
            "start": int(source_pos),
            "end": int(end),
            "label": label
        })
        inserted_ranges.append(output_range)
        source_pos = end

    return predicted_spans, inserted_ranges, alignment_failures


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def span_iou(a, b):
    intersection = max(
        0,
        min(a["end"], b["end"]) - max(a["start"], b["start"])
    )

    if intersection == 0:
        return 0.0

    union = (
        max(a["end"], b["end"]) - min(a["start"], b["start"])
    )
    return intersection / union if union else 0.0


def exact_tp(predicted, gold, label):
    pred_counter = Counter(
        (x["start"], x["end"], x["label"])
        for x in predicted if x["label"] == label
    )
    gold_counter = Counter(
        (x["start"], x["end"], x["label"])
        for x in gold if x["label"] == label
    )
    return sum((pred_counter & gold_counter).values())


def overlap_tp(predicted, gold, label):
    # Order-preserving one-to-one matching.
    # Maximize number of valid matches first, then total IoU.
    pred = sorted(
        [x for x in predicted if x["label"] == label],
        key=lambda x: (x["start"], x["end"])
    )
    gold = sorted(
        [x for x in gold if x["label"] == label],
        key=lambda x: (x["start"], x["end"])
    )

    n, m = len(pred), len(gold)
    dp = [[(0, 0.0) for _ in range(m + 1)] for _ in range(n + 1)]

    def better(a, b):
        if a[0] != b[0]:
            return a if a[0] > b[0] else b
        return a if a[1] >= b[1] else b

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = better(dp[i - 1][j], dp[i][j - 1])
            score = span_iou(pred[i - 1], gold[j - 1])

            if score >= 0.5:
                candidate = (
                    dp[i - 1][j - 1][0] + 1,
                    dp[i - 1][j - 1][1] + score
                )
                best = better(best, candidate)

            dp[i][j] = best

    return dp[n][m][0]


def calculate_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
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

    changed_non_pii = total_non_pii - len(preserved)
    return changed_non_pii, total_non_pii


def evaluate_predictions(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError(
            f"Expected {len(rows)} predictions, got {len(predictions)}."
        )

    exact_counts = {}
    overlap_counts = {}
    leaked_values = 0
    total_gold_values = 0
    changed_non_pii = 0
    total_non_pii = 0
    total_alignment_failures = 0
    prediction_records = []

    for row, predicted_text in zip(rows, predictions):
        gold_spans = [
            {
                "start": int(x["start"]),
                "end": int(x["end"]),
                "label": str(x["label"])
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
            tp_overlap = overlap_tp(predicted_spans, gold_spans, label)

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
            inserted_ranges
        )
        changed_non_pii += changed
        total_non_pii += total

        prediction_records.append({
            "id": row.get("id"),
            "language": row.get("language"),
            "source_text": row["source_text"],
            "target_text": row.get("target_text"),
            "predicted_text": predicted_text,
            "predicted_spans": predicted_spans,
            "gold_spans": gold_spans,
            "leaked_values": leaked_for_row,
            "alignment_failures": failures
        })

    def summarize(counts):
        total_tp = sum(x["tp"] for x in counts.values())
        total_fp = sum(x["fp"] for x in counts.values())
        total_fn = sum(x["fn"] for x in counts.values())
        micro_p, micro_r, micro_f1 = calculate_prf(
            total_tp, total_fp, total_fn
        )

        per_class = {}
        macro_f1_values = []

        for label, stat in sorted(counts.items()):
            p, r, f1 = calculate_prf(
                stat["tp"], stat["fp"], stat["fn"]
            )
            per_class[label] = {
                "precision": p,
                "recall": r,
                "f1": f1,
                "support": stat["support"]
            }
            if stat["support"] > 0:
                macro_f1_values.append(f1)

        macro_f1 = (
            sum(macro_f1_values) / len(macro_f1_values)
            if macro_f1_values else 0.0
        )

        return {
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_class": per_class
        }

    exact = summarize(exact_counts)
    overlap = summarize(overlap_counts)

    supports = {
        label: stat["support"] for label, stat in exact_counts.items()
    }
    top10_labels = [
        label for label, _ in sorted(
            supports.items(), key=lambda x: x[1], reverse=True
        )[:10]
    ]

    top10_per_class = {
        label: {
            "support": supports[label],
            "exact": exact["per_class"].get(label, {}),
            "overlap_iou50": overlap["per_class"].get(label, {})
        }
        for label in top10_labels
    }

    metrics = {
        "n": len(rows),
        "exact": {
            k: v for k, v in exact.items() if k != "per_class"
        },
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
        "alignment_failure_count": total_alignment_failures
    }

    return metrics, prediction_records


# ============================================================
# SAVE TASK-A RESULTS
# ============================================================

def append_result_csv(path, result):
    columns = [
        "run_id", "method", "target_modules", "lora_r", "lora_alpha",
        "quant", "trainable_params", "total_params", "trainable_pct",
        "peak_gpu_mem_gb", "train_wallclock_s", "tokens_per_s",
        "final_train_loss", "val_loss", "val_entity_f1", "val_leak_rate"
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({key: result[key] for key in columns})


def refresh_plot(csv_path, figure_path):
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # If a run is repeated, plot its latest row.
    latest = {}
    for row in rows:
        latest[row["run_id"]] = row
    rows = list(latest.values())

    if not rows:
        return

    trainable_pct = [float(x["trainable_pct"]) for x in rows]
    val_f1 = [float(x["val_entity_f1"]) for x in rows]
    gpu_mem = [float(x["peak_gpu_mem_gb"]) for x in rows]
    run_ids = [x["run_id"] for x in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(trainable_pct, val_f1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Trainable parameters (%)")
    axes[0].set_ylabel("Validation exact micro F1")
    axes[0].set_title("F1 vs trainable parameters")

    axes[1].scatter(gpu_mem, val_f1)
    axes[1].set_xlabel("Peak GPU memory (GB)")
    axes[1].set_ylabel("Validation exact micro F1")
    axes[1].set_title("F1 vs GPU memory")

    for i, run_id in enumerate(run_ids):
        axes[0].annotate(run_id, (trainable_pct[i], val_f1[i]))
        axes[1].annotate(run_id, (gpu_mem[i], val_f1[i]))

    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    


if __name__ == "__main__":

    args = parse_args()

    seed_everything(args.seed)


    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\n pad token : {tokenizer.pad_token} , pad token id : {tokenizer.pad_token_id}")

    labels , full_lens , all_full_ids = inspect(
        args.train_file,
        args.max_length,
        max_rows=args.debug_train_rows
    )

    if args.debug_train_rows is None and len(all_full_ids) != 20000:
        raise ValueError(
            f"Part A requires exactly 20,000 training rows; found {len(all_full_ids)}."
        )

    print(f"\n Training examples prepared: {len(all_full_ids)}")
    print(
        f" Sequence length min={min(full_lens)} "
        f"max={max(full_lens)} median={sorted(full_lens)[len(full_lens)//2]}"
    )

    train_ds = Dataset.from_dict({
        "input_ids": all_full_ids,
        "labels": labels,
        "seq_len": full_lens
    })


    
    collator = Collator(tokenizer.pad_token_id)

    batch_size = min(3, len(train_ds))

    features=[train_ds[i] for i in range(batch_size)]

    batch = collator(features)

    print(f"\n batch input ids : {batch['input_ids']}")
    print(f"\n batch attention mask : {batch['attention_mask']}")
    print(f"\n batch labels : {batch['labels']}")

    model , trainable_params , total_params , trainable_percentage = build_model(args)
    
    print("\nModel built successfully.")

    print("\nTrainable params:", trainable_params)
    print("\nTotal params:", total_params)
    print("\nTrainable %:", trainable_percentage)
    print("\nModel built successfully.")

    training_args = TrainingArguments(

       output_dir=args.output_dir,
       num_train_epochs=args.epochs,
       per_device_train_batch_size=args.train_batch_size,
       per_device_eval_batch_size=args.eval_batch_size,
       gradient_accumulation_steps=args.gradient_accumulation_steps,
       learning_rate=args.learning_rate,
       lr_scheduler_type=args.lr_scheduler_type,
       warmup_steps=args.warmup_ratio,
       weight_decay=args.weight_decay,
       logging_steps=args.logging_steps,
       save_strategy="no",
       report_to=[],
       fp16=not torch.cuda.is_bf16_supported(),
       bf16=torch.cuda.is_bf16_supported(),
       gradient_checkpointing=True,
       remove_unused_columns=False,
       dataloader_pin_memory=True,
       optim="adamw_torch",
       seed=args.seed,
       data_seed=args.seed
    )
    

    trainer = OrderedTrainer(

       model=model,

       args=training_args,

       train_dataset=train_ds,

       data_collator=collator
    )

    tokens_per_epoch = sum(full_lens)

    total_training_tokens = (tokens_per_epoch * args.epochs)


    


    print("\n Starting training...")

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()

    start_time = time.perf_counter()

    
    

    train_result = trainer.train()

    # Safety check only; this does not change the training configuration.
    # It prevents a numerically corrupted A6 adapter from continuing into
    # validation/generation and producing misleading 0/NaN metrics.
    bad_trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param.detach()).all():
            bad_trainable_params.append(name)

    if bad_trainable_params:
        print("\n Non-finite trainable parameters detected:")
        for name in bad_trainable_params[:20]:
            print("  ", name)
        raise RuntimeError(
            "Training produced NaN/Inf values in LoRA parameters. "
            "Stopping before validation."
        )

    torch.cuda.synchronize()

    train_wallclock_s = (
         time.perf_counter() - start_time
    )

    print("\n Training completed.")

    peak_gpu_mem_gb = (
         torch.cuda.max_memory_allocated()
         / 1024**3
    )

    print(
        "Peak GPU memory:",
        peak_gpu_mem_gb,
        "GB"
    )
    

    print(
       "Training time:",
       train_wallclock_s,
       "seconds"
    )

    logged_losses = [
        item["loss"]
        for item in trainer.state.log_history
        if isinstance(item, dict) and "loss" in item
    ]
    final_train_loss = (
        float(logged_losses[-1])
        if logged_losses
        else float(train_result.metrics["train_loss"])
    )
    
    print(
        "\n Final training loss:",
        final_train_loss
    )

    tokens_per_second = (
        total_training_tokens / train_wallclock_s
    )

    print(
        "\n Tokens per second:",
        tokens_per_second
    )

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n adapter saved successfully.")

    torch.cuda.empty_cache()


    print("\nPreparing validation dataset...")

    val_labels, val_full_lens, val_all_full_ids = inspect(
        args.val_file,
        args.max_length,
        max_rows=args.debug_val_rows
    )

    if args.debug_val_rows is None and len(val_all_full_ids) != 2000:
        raise ValueError(
            f"Part A requires exactly 2,000 validation rows; found {len(val_all_full_ids)}."
        )

    val_ds = Dataset.from_dict({
        "input_ids": val_all_full_ids,
        "labels": val_labels,
        "seq_len": val_full_lens
    })

    print("\nStarting evaluation...")

    eval_results = trainer.evaluate(eval_dataset=val_ds)

    val_loss = eval_results["eval_loss"]

    if not math.isfinite(float(val_loss)):
        raise RuntimeError(
            f"Validation loss is non-finite ({val_loss}). "
            "The adapter is numerically unstable; generation is skipped."
        )

    print(f"\n Validation loss: {val_loss}")

    torch.cuda.empty_cache()

    val_rows = load_json(args.val_file)

    if args.debug_val_rows is not None:
        val_rows = val_rows[:args.debug_val_rows]

    predictions = generate_validation(
        model,
        tokenizer,
        val_rows,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.gen_batch_size
    )

    print(f"\n generated predictions :{len(predictions)}")

    print(f"\n Evaluation results: {eval_results}")

    for i in range(min(3, len(val_rows))):
        print("\n==============================")
        print("SOURCE:")
        print(val_rows[i]["source_text"])
        print("\nTARGET:")
        print(val_rows[i]["target_text"])
        print("\nPREDICTION:")
        print(predictions[i])

    print("\nCalculating validation sanitizer metrics...")

    val_metrics, prediction_records = evaluate_predictions(
        val_rows, predictions
    )

    val_entity_f1 = val_metrics["exact"]["micro_f1"]
    val_leak_rate = val_metrics["leak_rate"]

    print("\n===== VALIDATION SANITIZER RESULTS =====")
    print("Exact micro F1:", val_metrics["exact"]["micro_f1"])
    print("Exact macro F1:", val_metrics["exact"]["macro_f1"])
    print(
        "IoU>=0.5 micro F1:",
        val_metrics["overlap_iou50"]["micro_f1"]
    )
    print(
        "IoU>=0.5 macro F1:",
        val_metrics["overlap_iou50"]["macro_f1"]
    )
    print("Leak rate:", val_metrics["leak_rate"])
    print("Over-masking rate:", val_metrics["over_masking_rate"])
    print(
        "Alignment failures:",
        val_metrics["alignment_failure_count"]
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_file = output_path / "validation_predictions.jsonl"
    with open(predictions_file, "w", encoding="utf-8") as f:
        for record in prediction_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics_file = output_path / "validation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2, ensure_ascii=False)

    result = {
        "run_id": args.run_id,
        "method": "QLoRA" if args.quant == "4bit" else "LoRA",
        "target_modules": args.target_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "quant": args.quant,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_percentage,
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
        "train_wallclock_s": train_wallclock_s,
        "tokens_per_s": tokens_per_second,
        "final_train_loss": final_train_loss,
        "val_loss": val_loss,
        "val_entity_f1": val_entity_f1,
        "val_leak_rate": val_leak_rate
    }

    append_result_csv(args.results_csv, result)
    refresh_plot(args.results_csv, args.figure_path)

    print("\n========== RUN COMPLETE ==========")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nSaved predictions:", predictions_file)
    print("Saved validation metrics:", metrics_file)
    print("Updated sweep CSV:", args.results_csv)
    print("Updated figure:", args.figure_path)




    
