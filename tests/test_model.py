import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import GPT
from config import GPTConfig


def test_forward_backward_shapes():
    cfg = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg)
    x = torch.randint(0, 65, (4, 16))
    y = torch.randint(0, 65, (4, 16))

    logits, loss, _ = model(x, y)
    assert logits.shape == (4, 16, 65), f"unexpected logits shape {logits.shape}"
    assert loss.item() > 0

    loss.backward()
    assert model.wte.weight.grad is not None, "no gradient reached the embedding table"
    print("test_forward_backward_shapes: OK")


def test_inference_mode_returns_last_position_only():
    cfg = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg)
    x = torch.randint(0, 65, (2, 10))
    logits, loss, _ = model(x)
    assert logits.shape == (2, 1, 65)
    assert loss is None
    print("test_inference_mode_returns_last_position_only: OK")


def test_weight_tying():
    cfg = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg)
    assert model.wte.weight is model.lm_head.weight, "embedding and output head should share weights"
    print("test_weight_tying: OK")


def test_kv_cache_matches_full_recompute():
    cfg = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg)
    model.eval()

    x = torch.randint(0, 65, (1, 8))
    with torch.no_grad():
        full_logits, _, _ = model(x, targets=torch.zeros_like(x))  

        past_kv, incremental = None, []
        for t in range(x.size(1)):
            inp = x[:, :1] if past_kv is None else x[:, t:t + 1]
            logits, _, past_kv = model(inp, use_cache=True, past_kv_list=past_kv)
            incremental.append(logits[:, -1, :])
        incremental_logits = torch.stack(incremental, dim=1)

    max_diff = (full_logits - incremental_logits).abs().max().item()
    assert max_diff < 1e-4, f"cached vs full logits diverged by {max_diff}"
    print(f"test_kv_cache_matches_full_recompute: OK (max diff {max_diff:.2e})")


def test_generate_produces_valid_tokens():
    cfg = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg)
    idx = torch.randint(0, 65, (1, 4))
    out = model.generate(idx, max_new_tokens=20, top_k=10)
    assert out.shape == (1, 24)
    assert out.min() >= 0 and out.max() < 65
    print("test_generate_produces_valid_tokens: OK")


def test_generate_cache_matches_nocache():
    cfg = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=32, dropout=0.0, bias=False)
    model = GPT(cfg)
    idx = torch.randint(0, 65, (1, 4))

    out_nocache = model.generate(idx.clone(), max_new_tokens=20, use_cache=False, top_k=1)
    out_cache = model.generate(idx.clone(), max_new_tokens=20, use_cache=True, top_k=1)

    assert torch.equal(out_nocache, out_cache), (
        "cached and non-cached generate() diverged -- the KV cache is not being threaded through forward() correctly"
    )
    print("test_generate_cache_matches_nocache: OK")


if __name__ == "__main__":
    test_forward_backward_shapes()
    test_inference_mode_returns_last_position_only()
    test_weight_tying()
    test_kv_cache_matches_full_recompute()
    test_generate_produces_valid_tokens()
    test_generate_cache_matches_nocache()
    print("\nAll tests passed.")
