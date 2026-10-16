# ELL8299 / COL8383 / COL873 — Assignment 1

## Fine-Tune, Sanitize, Extract

This repository contains the code, report, and results for Assignment 1. Fine-tuned PEFT adapters are stored in a **public Hugging Face model repository** instead of being committed to GitHub.

**Public adapter repository:** [uthith/ELL8299-Assignment1](https://huggingface.co/uthith/ELL8299-Assignment1)

## Repository structure

```text
2025JRB2054/
├── results/
├── src/
│   ├── attack_extract.py       # Part C: extraction attacks
│   ├── download_adapters.py    # Download saved adapters from Hugging Face
│   ├── peft_sweep.py           # Part A: PEFT sweep
│   ├── sanitize_eval.py        # Part B: sanitization evaluation
│   └── train_victim.py         # Part C: victim training
├── README.md
├── REPORT.pdf
├── requirements.txt
└── run_all.sh
```

After downloading, a new folder is created **at the repository root**:

```text
2025JRB2054/
└── adapters/
    ├── best_sanitizer/
    ├── victim_raw/
    └── victim_san/
```

## Installation

Use an environment with Python and the dependencies required by the assignment. From the repository root:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file must include `huggingface_hub` (alongside the project's existing dependencies).

## Downloading the adapters automatically

All three saved adapters are publicly accessible:

| Adapter | Hugging Face path | Local path |
| --- | --- | --- |
| Selected PII sanitizer (A4) | `adapters/best_sanitizer/` | `adapters/best_sanitizer/` |
| Victim fine-tuned on raw data | `adapters/victim_raw/` | `adapters/victim_raw/` |
| Victim fine-tuned on sanitized data | `adapters/victim_san/` | `adapters/victim_san/` |

To download them from Hugging Face into the repository root, run:

```bash
python src/download_adapters.py
```

The downloader retrieves `adapters/` from `uthith/ELL8299-Assignment1`, preserves all three subdirectories, and does **not** require a Hugging Face login for this public repository.

**For evaluation with no manual download steps:** `run_all.sh` should run `python src/download_adapters.py` before executing any scripts that load saved adapters. All scripts that use local adapters must resolve paths under the **root** `adapters/` directory, not `src/adapters/`. Any required base models (for example `Qwen/Qwen2.5-0.5B`) must likewise be retrieved automatically from their public model repositories using `from_pretrained(...)` or an equivalent download method.

The downloader handles **adapter files only**; it does not automatically alter training or evaluation scripts, download task datasets, or confirm the full pipeline runs. Check that the other required inputs and paths are accessible before submission.

## Run the assignment

From the repository root:

```bash
bash run_all.sh
```

**Note:** Verify that `run_all.sh` calls `python src/download_adapters.py` before any adapter-dependent stages. The contents of `run_all.sh` have not been inspected as part of this README update.

## Part A — PEFT sweep

The PEFT sweep evaluates LoRA/QLoRA configurations for the PII sanitizer. The selected sanitizer is **configuration A4**.

Main source file: `src/peft_sweep.py`.

## Part B — Sanitization evaluation

The sanitizer is evaluated on in-distribution (ID) and out-of-distribution (OOD) data against regex, zero-shot, and other required baselines.

Main source file: `src/sanitize_eval.py`.

## Part C — Victim training and extraction

Three victim variants are compared:

- **V-base:** untouched `Qwen/Qwen2.5-0.5B`
- **V-raw:** fine-tuned on the raw corpus
- **V-san:** fine-tuned on the sanitized corpus

Implemented extraction analyses:

- Discoverable memorization
- Untargeted extraction
- Canary exposure
- Targeted PII extraction

Main source files: `src/train_victim.py` and `src/attack_extract.py`.

All experiments use **seed 42**.

## PII-index limitation

The released assignment bundle did not contain `partC_pii_index.jsonl`. For untargeted PII recovery, the implementation therefore uses the 200 known member PII values contained in `partC_targeted.jsonl`.

## Privacy

No base-model weights, full corpora, raw generation dumps, or private datasets are included in the GitHub submission. Fine-tuned PEFT adapters are made available separately through the public Hugging Face repository linked above.
