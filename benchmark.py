import time
import argparse

import torch

from model import GPT
from config import GPTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None, help="optional checkpoint; otherwise uses a random small model")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = get_args()

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config).to(args.device)
        model.load_state_dict(ckpt["model"])
        vocab_size = config.vocab_size
    else:
        config = GPTConfig(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=False)
        model = GPT(config).to(args.device)
        vocab_size = config.vocab_size

    model.eval()
    idx = torch.randint(0, vocab_size, (1, 1), device=args.device)

    speeds = {}
    for use_cache in (False, True):
        torch.manual_seed(0)
        t0 = time.time()
        with torch.no_grad():
            model.generate(idx, max_new_tokens=args.max_new_tokens, use_cache=use_cache, top_k=50)
        dt = time.time() - t0
        label = "with KV-cache" if use_cache else "no cache (full recompute)"
        tok_per_sec = args.max_new_tokens / dt
        speeds[use_cache] = tok_per_sec
        print(f"{label:28s}: {dt:6.2f}s for {args.max_new_tokens} tokens ({tok_per_sec:.1f} tok/s)")

    if speeds[False] > 0:
        print(f"\nspeedup from KV-cache: {speeds[True] / speeds[False]:.1f}x")


if __name__ == "__main__":
    main()
