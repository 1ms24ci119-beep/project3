import torch
from app.core import CropGPT, CausalSelfAttention

def test_model_logits_shape_and_loss():
    model = CropGPT(vocab_size=32, block_size=8, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and torch.isfinite(loss)

def test_causal_attention_output_shape():
    layer = CausalSelfAttention(n_embd=16, n_head=4, block_size=8, dropout=0.0)
    x = torch.randn(2, 5, 16)
    assert layer(x).shape == x.shape

def test_generation_adds_tokens():
    model = CropGPT(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_layer=1, dropout=0.0)
    x = torch.tensor([[1, 2, 3]])
    out = model.generate(x, max_new_tokens=2, top_k=5)
    assert out.shape == (1, 5)


def test_attention_is_causal():
    torch.manual_seed(7)
    layer = CausalSelfAttention(n_embd=16, n_head=4, block_size=8, dropout=0.0)
    layer.eval()
    first = torch.randn(1, 5, 16)
    changed_future = first.clone()
    changed_future[:, 4, :] += 100.0
    out_a = layer(first)
    out_b = layer(changed_future)
    assert torch.allclose(out_a[:, :4, :], out_b[:, :4, :], atol=1e-6)
