import csv

import torch

from batching import BOS_ID, IGNORE_INDEX, PAD_ID, GameDataset, collate
from dummy_data import load_dummy_splits
from train import main, topk_correct, warmup_cosine


def test_dataset_shifts_targets_by_one():
    inputs, targets = GameDataset([[5, 6, 7]], max_len=10)[0]
    assert inputs.tolist() == [BOS_ID, 5, 6]
    assert targets.tolist() == [5, 6, 7]


def test_dataset_truncates_long_games():
    inputs, targets = GameDataset([list(range(2, 20))], max_len=5)[0]
    assert len(inputs) == len(targets) == 5


def test_collate_pads_on_the_right():
    ds = GameDataset([[5, 6, 7], [8]], max_len=10)
    inputs, targets = collate([ds[0], ds[1]])
    assert inputs.tolist() == [[BOS_ID, 5, 6], [BOS_ID, PAD_ID, PAD_ID]]
    assert targets.tolist() == [[5, 6, 7], [8, IGNORE_INDEX, IGNORE_INDEX]]


def test_topk_correct_skips_padding():
    # ranking is 3 > 2 > 1 > 0
    logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0]] * 3])
    targets = torch.tensor([[3, 1, IGNORE_INDEX]])
    assert topk_correct(logits, targets, 1) == (1, 2)
    assert topk_correct(logits, targets, 3) == (2, 2)


def test_warmup_cosine_schedule():
    f = warmup_cosine(total_steps=100, warmup_steps=10)
    assert f(0) == 0.1 and f(9) == 1.0
    assert f(10) == 1.0
    assert abs(f(100)) < 1e-9


def test_dummy_splits_are_60_20_20():
    train, val, test, vocab_size = load_dummy_splits(n_games=100)
    assert (len(train), len(val), len(test)) == (60, 20, 20)
    assert vocab_size == 1896 + 2


def test_training_learns_dummy_pattern(tmp_path):
    main(["--data", "dummy", "--epochs", "4", "--d-model", "64", "--n-heads", "4",
          "--n-layers", "2", "--lr", "3e-3", "--warmup-steps", "20", "--batch-size", "32",
          "--device", "cpu", "--run-name", "test", "--out-dir", str(tmp_path)])

    with open(tmp_path / "results" / "metrics" / "test.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert float(rows[-1]["val_top1"]) > 0.9
    assert (tmp_path / "checkpoints" / "test.pt").exists()
