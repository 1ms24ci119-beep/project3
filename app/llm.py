"""Training and generation pipeline."""
import random
import torch
from .core import CropGPT
from .tokenizer import BPETokenizer
from .storage import save_checkpoint

def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_training_data(texts, tokenizer, block_size):
    encoded = []
    for text in texts:
        encoded.extend(tokenizer.encode(text, add_special_tokens=True))
    data = torch.tensor(encoded, dtype=torch.long)
    if len(data) < 4:
        raise ValueError("Corpus is too small to create training sequences.")
    return data

def get_batch(data, batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    if max_start < 1:
        block_size = max(2, len(data) - 2)
        max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

def train_model(texts, output, steps=300, batch_size=16, block_size=96,
                vocab_size=400, learning_rate=3e-4, device="auto", seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(device)
    tokenizer = BPETokenizer().train(texts, vocab_size=vocab_size)
    data = build_training_data(texts, tokenizer, block_size)
    config = dict(vocab_size=tokenizer.vocab_size, block_size=block_size,
                  n_embd=96, n_head=4, n_layer=2, dropout=0.1)
    model = CropGPT(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(steps):
        xb, yb = get_batch(data, batch_size, block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(1, steps // 5) == 0 or step == 0:
            print(f"step {step + 1}/{steps} | loss {loss.item():.4f}")
    save_checkpoint(output, model, tokenizer, config)
    return model, tokenizer, config

@torch.no_grad()
def generate_advice(model, tokenizer, prompt, max_new_tokens=70, temperature=0.75, device="cpu"):
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.token_to_id["<bos>"]]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature)
    continuation = generated[0, len(ids):].tolist()
    return tokenizer.decode(continuation)
