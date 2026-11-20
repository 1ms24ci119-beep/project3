"""Corpus loading and checkpoint helpers."""
from pathlib import Path
import torch

def load_corpus(path):
    text = Path(path).read_text(encoding="utf-8")
    records = [line.strip() for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No usable text found in corpus: {path}")
    return records

def save_checkpoint(path, model, tokenizer, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_path = path.with_suffix(".tokenizer.json")
    tokenizer.save(tokenizer_path)
    torch.save({"model": model.state_dict(), "config": config, "tokenizer": str(tokenizer_path)}, path)

def load_checkpoint(path, device="cpu"):
    from .tokenizer import BPETokenizer
    from .core import CropGPT
    payload = torch.load(path, map_location=device, weights_only=False)
    tokenizer = BPETokenizer.load(payload["tokenizer"])
    model = CropGPT(**payload["config"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, tokenizer, payload["config"]
