"""Training loop (shared by the transformer and the LSTM baseline).

python src/train.py --data dummy --epochs 3 --d-model 128 --n-layers 2
"""

import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from batching import IGNORE_INDEX, GameDataset, collate
from dummy_data import load_dummy_splits
from real_data import load_real_splits
from models.transformer import ChessTransformer, count_parameters

ROOT = Path(__file__).resolve().parent.parent


def topk_correct(logits, targets, k):
    """Returns (# targets in the top k, # non-padding targets)."""
    real = targets != IGNORE_INDEX
    top = logits.topk(k, dim=-1).indices
    hit = (top == targets.unsqueeze(-1)).any(dim=-1)
    return int((hit & real).sum()), int(real.sum())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, top1, top3, n = 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                      ignore_index=IGNORE_INDEX, reduction="sum").item()
        c1, count = topk_correct(logits, y, 1)
        c3, _ = topk_correct(logits, y, 3)
        top1 += c1
        top3 += c3
        n += count
    return {"loss": total_loss / n, "top1": top1 / n, "top3": top3 / n}


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=IGNORE_INDEX)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        count = int((y != IGNORE_INDEX).sum())
        total_loss += loss.item() * count
        n += count
    return total_loss / n


def warmup_cosine(total_steps, warmup_steps):
    # linear warmup, then cosine decay to 0
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return factor


def load_data(name):
    if name == "dummy":
        return load_dummy_splits()
    elif name == "real":
        return load_real_splits()
    raise NotImplementedError(f"no loader for '{name}' yet")


def build_model(args, vocab_size):
    if args.model == "transformer":
        return ChessTransformer(vocab_size=vocab_size, max_len=args.max_len, d_model=args.d_model,
                                n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)
    raise ValueError(f"unknown model '{args.model}'")


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dummy")
    p.add_argument("--model", default="transformer")
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--out-dir", default=str(ROOT))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    run_name = args.run_name or f"{args.model}-{args.data}"

    train_games, val_games, _, vocab_size = load_data(args.data)
    train_loader = DataLoader(GameDataset(train_games, args.max_len), batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(GameDataset(val_games, args.max_len), batch_size=args.batch_size,
                            collate_fn=collate, num_workers=args.num_workers)

    model = build_model(args, vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, min(args.warmup_steps, total_steps)))

    print(f"{run_name}: {device}, {count_parameters(model):,} params, "
          f"{len(train_games)} train / {len(val_games)} val games, vocab {vocab_size}")

    out_dir = Path(args.out_dir)
    metrics_path = out_dir / "results" / "metrics" / f"{run_name}.csv"
    ckpt_path = out_dir / "checkpoints" / f"{run_name}.pt"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_top1", "val_top3", "seconds"])
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.grad_clip)
            val = evaluate(model, val_loader, device)
            secs = time.time() - start

            writer.writerow([epoch, f"{train_loss:.4f}", f"{val['loss']:.4f}",
                             f"{val['top1']:.4f}", f"{val['top3']:.4f}", f"{secs:.1f}"])
            f.flush()
            print(f"epoch {epoch} | train {train_loss:.4f} | val {val['loss']:.4f} | "
                  f"top1 {val['top1']:.2%} | top3 {val['top3']:.2%} | {secs:.0f}s")

            # keep the checkpoint with the best val loss
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                torch.save({"model_state": model.state_dict(), "args": vars(args),
                            "vocab_size": vocab_size, "epoch": epoch, "val": val}, ckpt_path)

    print(f"best val loss {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    main()
