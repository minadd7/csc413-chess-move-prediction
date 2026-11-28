"""Dataset + collate for games stored as lists of move ids.

game [m1, m2, m3] -> input [BOS, m1, m2], target [m1, m2, m3]
"""

import torch
from torch.utils.data import Dataset

PAD_ID = 0
BOS_ID = 1
IGNORE_INDEX = -100  # cross_entropy skips these targets


class GameDataset(Dataset):
    def __init__(self, games, max_len):
        self.games = [g[:max_len] for g in games if len(g) > 0]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        moves = self.games[i]
        x = torch.tensor([BOS_ID] + moves[:-1], dtype=torch.long)
        y = torch.tensor(moves, dtype=torch.long)
        return x, y


def collate(batch):
    # right-pad to the longest game in the batch
    longest = max(len(x) for x, _ in batch)
    inputs = torch.full((len(batch), longest), PAD_ID, dtype=torch.long)
    targets = torch.full((len(batch), longest), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        inputs[i, :len(x)] = x
        targets[i, :len(y)] = y
    return inputs, targets
