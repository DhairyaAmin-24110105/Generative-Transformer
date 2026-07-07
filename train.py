import os
import time
import math
import pickle
import argparse
import dataclasses

import numpy as np
import torch

from model import GPT
from config import GPTConfig, PRESETS


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--preset", type=str, default=None, choices=list(PRESETS.keys()), help="use a named size preset from config.py (overridden by explicit flags below)")

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bias", action="store_true")

    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=50)

    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--lr_decay_iters", type=int, default=None, help="defaults to max_iters")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.preset is not None:
        for k, v in PRESETS[args.preset].items():
            setattr(args, k, v)
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.max_iters
    return args


def get_batch(split, data_dir, block_size, batch_size, device):
    path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def get_lr(it, args):
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / args.warmup_iters
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / max(1, args.lr_decay_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(model, args):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split, args.data_dir, args.block_size, args.batch_size, args.device)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=meta["vocab_size"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )

    model = GPT(config).to(args.device)
    n_params = model.get_num_params()
    print(f"model: {n_params/1e6:.2f}M parameters (non-embedding)")

    raw_model = model  
    if args.compile:
        model = torch.compile(model)

    optimizer = raw_model.configure_optimizer(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), args.device
    )

    start_iter = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"resumed from {ckpt_path} at iter {start_iter}")

    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = get_lr(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        is_last = it == args.max_iters - 1
        if it % args.eval_interval == 0 or is_last:
            losses = estimate_loss(model, args)
            elapsed = time.time() - t0
            print(f"iter {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {elapsed:.1f}s elapsed")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": dataclasses.asdict(config),  
                    "iter": it,
                    "best_val_loss": best_val_loss,
                    "meta": meta,
                }, ckpt_path)
                print(f"saved checkpoint (val loss {best_val_loss:.4f})")

        x, y = get_batch("train", args.data_dir, args.block_size, args.batch_size, args.device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0 and it % args.eval_interval != 0:
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e}")

    print(f"training complete. best val loss: {best_val_loss:.4f}. checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
