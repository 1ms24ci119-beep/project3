# Crop Advice Snippet AI

A small, CPU-friendly educational project that trains a GPT-style causal language model from scratch on agricultural text and generates short crop/problem-conditioned advice snippets from a command line.

> **Important:** This is an educational drafting aid, not an agronomist or diagnostic system. Verify farm decisions with local agricultural extension services or a qualified agronomist. Do not rely on generated text alone for pesticide, fertilizer, irrigation, disease, or food-safety decisions.

## Features

- Byte-pair encoding (BPE) tokenizer implemented in the project.
- Learned token and positional embeddings.
- Multi-head causal self-attention with a future-token mask.
- Transformer blocks with residual connections, layer normalization, and feed-forward layers.
- Autoregressive next-token training with cross-entropy and AdamW.
- Temperature/top-k generation.
- CLI commands for training and advice generation.
- Unit tests and GitHub Actions workflow.

## Quick start

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train a small model:

```bash
python -m app train --corpus data/agriculture_corpus.txt --steps 300
```

Generate a snippet:

```bash
python -m app advise --crop rice --problem "water management"
```

Use `python -m app --help`, `python -m app train --help`, and `python -m app advise --help` for options.

## Architecture

1. `app/tokenizer.py` learns BPE merges and maps text to token IDs.
2. `app/core.py` implements embeddings, causal multi-head attention, transformer blocks, and the language-model head.
3. `app/llm.py` creates shifted next-token batches, trains the model, and samples continuations.
4. `app/storage.py` loads corpus records and saves/loads checkpoints with tokenizer metadata.
5. `app/cli.py` exposes training and inference commands.

## Tests

```bash
pytest -q
```

## Data provenance and limitations

`data/agriculture_corpus.txt` is a small illustrative starter corpus written for this project. It is not a verified agronomy dataset and must not be presented as expert-validated agricultural advice. Replace or extend it only with public-domain, openly licensed, or user-provided material, and record source, license, region, date, and preprocessing decisions.

A tiny model trained on a small corpus demonstrates GPT mechanics; it is not expected to produce reliable or comprehensive agricultural recommendations. It may repeat, contradict, or invent information. Generated output must be reviewed by a human.

## Repository hygiene

The `.gitignore` excludes virtual environments, caches, local environment files, model checkpoints, and OS-generated files. Do not commit secrets, private farm records, or unlicensed source material.
