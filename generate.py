import argparse

import torch

from model import GPT
from config import GPTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--no_cache", action="store_true", help="disable KV-cache (slower, mainly for comparison)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = get_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = GPTConfig(**ckpt["config"])
    meta = ckpt["meta"]
    stoi, itos = meta["stoi"], meta["itos"]

    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=args.device)

    for i in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
        )
        text = decode(out[0].tolist())
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
