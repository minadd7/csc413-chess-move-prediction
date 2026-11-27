"""Load the processed Lichess games made by preprocess.py."""

import json
from pathlib import Path

import numpy as np

from batching import BOS_ID, PAD_ID

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def row_to_moves(row):
    # [BOS, m1, m2, ..., PAD, PAD] -> [m1, m2, ...]; GameDataset adds BOS back
    moves = row[row != PAD_ID]
    moves = moves[1:]
    return [int(m) for m in moves]


def load_real_splits(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    vocab = json.load(open(data_dir/"vocab.json"))

    assert vocab["pad_id"] == PAD_ID and vocab["bos_id"] == BOS_ID

    splits = []
    for name in ("train", "val", "test"):
        arr = np.load(data_dir/f"{name}.npy")
        splits.append([row_to_moves(row) for row in arr])

    train, val, test = splits
    return train, val, test, vocab["vocab_size"]
