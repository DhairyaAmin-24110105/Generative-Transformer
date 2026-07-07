import os
import pickle
import argparse
import urllib.request

import numpy as np

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists, skipping download.")
        return
    print(f"Downloading tiny shakespeare dataset to {dest_path} ...")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dest_path)


def prepare(input_path, out_dir, val_fraction=0.1):
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    chars = sorted(set(data))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    n = len(data)
    split = int(n * (1 - val_fraction))
    train_data, val_data = data[:split], data[split:]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

    print(f"vocab size: {vocab_size} unique characters")
    print(f"train tokens: {len(train_ids):,} | val tokens: {len(val_ids):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default=None,
        help="path to a custom text file; if omitted, downloads tiny shakespeare",
    )
    parser.add_argument("--out_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.input is None:
        input_path = os.path.join(args.out_dir, "input.txt")
        download_tiny_shakespeare(input_path)
    else:
        input_path = args.input

    prepare(input_path, args.out_dir, args.val_fraction)


if __name__ == "__main__":
    main()
