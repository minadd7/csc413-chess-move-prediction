"""Tests for src/preprocess.py, using small fake Lichess-style games."""
import json
import random
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as pp  # noqa: E402


def random_game_movetext(rng, n_plies, bad_move=False):
    """Play random legal moves and write them like Lichess (with clock comments)."""
    if bad_move:
        return "1. e4 { [%clk 0:09:58] } 1... e5 { [%clk 0:09:58] } 2. Ke4 1-0"  # illegal
    board = chess.Board()
    parts = []
    for ply in range(n_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        san = board.san(move)
        if ply % 2 == 0:
            parts.append(f"{ply // 2 + 1}.")
        else:
            parts.append(f"{ply // 2 + 1}...")
        parts.append(san)
        parts.append("{ [%clk 0:09:58] }")
        board.push(move)
    parts.append("1-0")
    return " ".join(parts)


def make_game(idx, white=2150, black=2200, tc="600+0", n_plies=40, rng=None,
              result="1-0", extra_tags="", bad_move=False):
    rng = rng or random.Random(idx)
    return (
        f'[Event "Rated Rapid game"]\n[Site "https://lichess.org/g{idx:06d}"]\n'
        f'[White "a{idx}"]\n[Black "b{idx}"]\n[Result "{result}"]\n'
        f'[WhiteElo "{white}"]\n[BlackElo "{black}"]\n[TimeControl "{tc}"]\n'
        f'{extra_tags}\n'
        f'{random_game_movetext(rng, n_plies, bad_move)}\n\n'
    )


def write_zst(path, text):
    with open(path, "wb") as fh:
        fh.write(zstandard.ZstdCompressor().compress(text.encode()))


@pytest.fixture
def raw_dir(tmp_path):
    games = [make_game(1), # keep
        make_game(2, white=2100, black=2299),  # keep (boundaries)
        make_game(3, tc="900+10"),  # keep (boundaries)
        make_game(4, white=2099), # rating too low
        make_game(5, black=2300), # rating too high
        make_game(6, tc="180+0"), # too fast
        make_game(7, tc="1800+0"), # too slow
        make_game(8, tc="600+15"),  # increment too big
        make_game(9, tc="-"),  # no time control
        make_game(10, extra_tags='[WhiteTitle "BOT"]\n'),# bot
        make_game(11, extra_tags='[FEN "8/8/8/8/8/8/8/K6k w - - 0 1"]\n'),
        make_game(12, result="*"),  # unfinished
        make_game(13, bad_move=True), # illegal move
        make_game(14, n_plies=0), # no moves
        make_game(15, white=0, tc="600+0"), # low rating
    ]
    raw = tmp_path / "raw"
    raw.mkdir()
    write_zst(raw / "month1.pgn.zst", "".join(games[:8]))
    write_zst(raw / "month2.pgn.zst", "".join(games[8:]))
    return raw


def test_filter_keeps_only_matching_games(raw_dir, tmp_path):
    interim = tmp_path / "interim"
    pp.run_filter(raw_dir, interim, workers=2)
    kept = []
    for f in sorted(interim.glob("*.jsonl")):
        kept += [json.loads(line) for line in open(f)]
    sites = sorted(g["site"].split("/")[-1] for g in kept)
    assert sites == ["g000001", "g000002", "g000003"]
    stats = json.load(open(interim / "filter_stats.json"))
    assert stats["total"] == 15 and stats["kept"] == 3
    assert stats["rejected_rating_range"] == 3
    assert stats["rejected_time_control"] == 3
    assert stats["rejected_missing_time_control"] == 1
    assert stats["rejected_bot"] == 1
    assert stats["rejected_nonstandard_start"] == 1
    assert stats["rejected_bad_result"] == 1
    assert stats["rejected_parse_error"] == 1
    assert stats["rejected_no_moves"] == 1


def test_uci_moves_replay_legally(raw_dir, tmp_path):
    interim = tmp_path / "interim"
    pp.run_filter(raw_dir, interim, workers=1)
    for f in interim.glob("*.jsonl"):
        for line in open(f):
            g = json.loads(line)
            board = chess.Board()
            for uci in g["moves"]:
                move = chess.Move.from_uci(uci)
                assert move in board.legal_moves
                board.push(move)
            assert len(g["moves"]) > 0


def test_build_split_vocab_and_encoding(tmp_path):
    interim = tmp_path / "interim"
    interim.mkdir()
    rng = random.Random(0)
    with open(interim / "x.jsonl", "w") as fh:
        for i in range(50):
            board, moves = chess.Board(), []
            for _ in range(rng.randint(5, 30)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                m = rng.choice(legal)
                moves.append(m.uci())
                board.push(m)
            fh.write(json.dumps({"site": f"s{i:03d}", "white": 2100 + i,
                                 "black": 2200, "tc": "600+0",
                                 "result": "1-0", "moves": moves}) + "\n")
    out = tmp_path / "processed"
    pp.run_build(interim, out, max_len=16, seed=42)

    train, val, test = (np.load(out / f"{n}.npy") for n in ("train", "val", "test"))
    assert (len(train), len(val), len(test)) == (30, 10, 10)
    assert train.shape[1] == 16 and train.dtype == np.int64
    assert (train[:, 0] == pp.BOS_ID).all()
    # every game starts with BOS
    vocab = json.load(open(out / "vocab.json"))
    assert vocab["vocab_size"] == len(vocab["move_to_id"]) + 2
    assert train.max() < vocab["vocab_size"]
    # PAD only appears as trailing padding
    for row in train:
        nonpad = np.nonzero(row != pp.PAD_ID)[0]
        assert (nonpad == np.arange(len(nonpad))).all()
    # ratings saved alongside, same row order
    assert np.load(out / "train_ratings.npy").shape == (30, 2)

    # determinism: same seed gives identical splits, different seed does not
    out2 = tmp_path / "processed2"
    pp.run_build(interim, out2, max_len=16, seed=42)
    assert (np.load(out2 / "train.npy") == train).all()
    out3 = tmp_path / "processed3"
    pp.run_build(interim, out3, max_len=16, seed=7)
    assert not (np.load(out3 / "train.npy") == train).all()


def test_split_sizes_match_proposal():
    tr, va, te = pp.split_indices(44626, 42)
    assert (len(tr), len(va), len(te)) == (26775, 8925, 8926)
    assert len(set(tr) | set(va) | set(te)) == 44626
