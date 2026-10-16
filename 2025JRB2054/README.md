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

**Automatic downloads:** `run_all.sh` invokes `python src/download_adapters.py` so all three adapters are fetched without a manual download step. Code that loads local adapters must resolve paths under the repository-root `adapters/` directory, not `src/adapters/`. Any required base models (for example `Qwen/Qwen2.5-0.5B`) must also be retrieved automatically by the relevant model-loading code (such as `from_pretrained(...)`). Base-model loading has not been verified here.

The downloader handles **adapter files only**; it does not automatically alter training or evaluation scripts, download task datasets, or confirm the full pipeline runs. Check that the other required inputs and paths are accessible before submission.

## Run the assignment

From the repository root:

```bash
bash run_all.sh
```

The script checks that the source files exist, compiles their Python syntax, downloads the three adapters from Hugging Face, and verifies that the downloaded directories are non-empty. **It is a repository check, not a full training or evaluation runner.** Run the assignment evaluation scripts using their existing commands and required datasets; check that they use the downloaded adapter paths and automatically retrieve any required base models.

On Windows, run this Bash script from **Git Bash** or **WSL**. The adapter download can also be invoked independently from PowerShell: `python src/download_adapters.py`.

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
