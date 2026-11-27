"""Fake games for testing training before the real data is in.

Each game follows next = (7 * prev + 3) % n_moves, so a working model
should get close to 100% on it.
"""

import random

N_SPECIAL = 2  # PAD, BOS


def make_dummy_games(n_games, n_moves=1896, min_len=20, max_len=120, seed=0):
    rng = random.Random(seed)
    games = []
    for _ in range(n_games):
        length = rng.randint(min_len, max_len)
        move = rng.randrange(n_moves)
        game = []
        for _ in range(length):
            game.append(move + N_SPECIAL)
            move = (7 * move + 3) % n_moves
        games.append(game)
    return games


def load_dummy_splits(n_games=2000, n_moves=1896, seed=42):
    games = make_dummy_games(n_games, n_moves, seed=seed)
    n_train, n_val = int(0.6 * n_games), int(0.2 * n_games)
    train = games[:n_train]
    val = games[n_train:n_train + n_val]
    test = games[n_train + n_val:]
    return train, val, test, n_moves + N_SPECIAL
