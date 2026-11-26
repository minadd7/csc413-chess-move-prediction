"""Data pipeline for the chess move prediction.

filter: python src/preprocess.py filter
    Keeps the games we want from the raw Lichess files and saves them to data/interim/.
build:  python src/preprocess.py build
    Builds the vocab, splits the games 60/20/20, and saves token arrays to data/processed/.

Token ids: 0 = PAD, 1 = BOS, moves start at 2.
"""

import argparse
import io
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import chess
import chess.pgn
import numpy as np
import zstandard


# Filter settings (from our proposal)
# ---------------------------------------------------------------------------
# Both players must be in this rating range (inclusive)
MIN_RATING, MAX_RATING = 2100, 2299
# 10 to 15 minutes
MIN_BASE_SECONDS, MAX_BASE_SECONDS = 600, 900
# Increment is the seconds added to the clock after each move
MIN_INC_SECONDS, MAX_INC_SECONDS = 0, 10
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}

PAD_ID, BOS_ID, FIRST_MOVE_ID = 0, 1, 2

# Numbers reported in the proposal, used as a sanity check after the build
EXPECTED = {"games": 44626, "moves": 3419401, "distinct_moves": 1896}

# -----------------------------------------------------------------------
# Stage 1: filter
# ---------------------------------------------------------------------------
def _tag(line):
    """Return (key, value) for a PGN header line like '[WhiteElo "2150"]'."""
    inner = line[1:-1]
    key, _, rest = inner.partition(" ")
    return key, rest.strip().strip('"')

def _header_reject_reason(h):
    """Return the first reason a game fails the header filters, or None."""
    # FEN means the game didn't start from the normal position
    if "FEN" in h or h.get("Variant", "Standard") != "Standard":
        return "nonstandard_start"
    # Skip bot accounts (this tag may be missing in the older dumps)
    if h.get("WhiteTitle") == "BOT" or h.get("BlackTitle") == "BOT":
        return "bot"
    try:
        white, black = int(h["WhiteElo"]), int(h["BlackElo"])
    except (KeyError, ValueError):
        return "missing_rating"
    if not (MIN_RATING <= white <= MAX_RATING and MIN_RATING <= black <= MAX_RATING):
        return "rating_range"
    try:
        # TimeControl looks like "600+5" (start seconds + increment)
        base, inc = h.get("TimeControl", "-").split("+")
        base, inc = int(base), int(inc)
    except ValueError:
        return "missing_time_control"
    if not (MIN_BASE_SECONDS <= base <= MAX_BASE_SECONDS
            and MIN_INC_SECONDS <= inc <= MAX_INC_SECONDS):
        return "time_control"
    if h.get("Result") not in VALID_RESULTS:
        return "bad_result"
    return None


def parse_movetext(movetext):
    """Replay a game from the start and return its moves in UCI.
    Returns None if there is an error or an illegal move (empty list if no moves)."""
    # read_game doesn't raise on bad moves, it stores them in game.errors
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None or game.errors:
        return None
    board = game.board()
    ucis = []
    for move in game.mainline_moves():
        if move not in board.legal_moves:
            return None
        ucis.append(move.uci())
        board.push(move)
    return ucis


def _iter_games(text_stream):
    """Go through the PGN text and give back (headers, movetext) for each game.
    We only keep the header tags we need for filtering."""
    keep_tags = {"Site", "Result", "WhiteElo", "BlackElo", "WhiteTitle",
              "BlackTitle", "TimeControl", "Variant", "FEN"}
    headers, movetext = {}, None
    for raw in text_stream:
        line = raw.strip()
        if not line:
            continue
        # A new [Event line means the previous game is done
        if line.startswith("["):
            if line.startswith("[Event ") and (headers or movetext is not None):
                yield headers, movetext
                headers, movetext = {}, None
            key, value = _tag(line)
            if key in keep_tags:
                headers[key] = value
        else:
            movetext = line if movetext is None else movetext + " " + line
    if headers or movetext is not None:
        yield headers, movetext


def filter_archive(path, out_dir):
    """Filter one archive, write the kept games to a .jsonl file, return the counts."""
    path, out_dir = Path(path), Path(out_dir)
    out_path = out_dir / (path.name.replace(".pgn.zst", "") + ".jsonl")
    stats = Counter()
    dctx = zstandard.ZstdDecompressor()
    # Read the file as a stream so we never load the full archive in memory
    with open(path, "rb") as fh, open(out_path, "w") as out:
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        for headers, movetext in _iter_games(text):
            stats["total"] += 1
            # Check the headers first, it's cheap and rejects almost every game
            reason = _header_reject_reason(headers)
            if reason is None and not movetext:
                reason = "no_moves"
            if reason is None:
                ucis = parse_movetext(movetext)
                if ucis is None:
                    reason = "parse_error"
                elif not ucis:
                    reason = "no_moves"
            if reason is not None:
                stats["rejected_" + reason] += 1
                continue
            stats["kept"] += 1
            out.write(json.dumps({
                "site": headers.get("Site", ""),
                "white": int(headers["WhiteElo"]),
                "black": int(headers["BlackElo"]),
                "tc": headers["TimeControl"],
                "result": headers["Result"],
                "moves": ucis,
            }) + "\n")
    return str(path.name), dict(stats)


def run_filter(raw_dir, interim_dir, workers):
    raw_dir, interim_dir = Path(raw_dir), Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(raw_dir.glob("*.pgn.zst"))
    if not archives:
        sys.exit(f"No .pgn.zst files found in {raw_dir}")
    total = Counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(filter_archive, a, interim_dir) for a in archives]
        for fut in futures:
            name, stats = fut.result()
            print(f"{name}: kept {stats.get('kept', 0)} of {stats.get('total', 0)}")
            total.update(stats)
    print("\nFilter summary (first failing reason per rejected game):")
    for key, val in sorted(total.items()):
        print(f"  {key}: {val}")
    with open(interim_dir / "filter_stats.json", "w") as fh:
        json.dump(dict(total), fh, indent=2)


# --------------------------------------------------
# Stage 2: build
# ----------------------------------------------------------
def load_games(interim_dir):
    games, seen = [], set()
    for path in sorted(Path(interim_dir).glob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                g = json.loads(line)
                if g["site"] and g["site"] in seen:   # drop duplicate games
                    continue
                seen.add(g["site"])
                games.append(g)
    games.sort(key=lambda g: g["site"]) # deterministic order
    return games


def build_vocab(games):
    moves = sorted({m for g in games for m in g["moves"]})
    return {m: i + FIRST_MOVE_ID for i, m in enumerate(moves)}


def encode(games, vocab, max_len):
    """Each row is [BOS, move1, move2, ...], cut at max_len and padded with PAD."""
    arr = np.full((len(games), max_len), PAD_ID, dtype=np.int64)
    for i, g in enumerate(games):
        ids = [BOS_ID] + [vocab[m] for m in g["moves"]]
        ids = ids[:max_len]
        arr[i, :len(ids)] = ids
    return arr


def split_indices(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train, n_val = int(0.6 * n), int(0.2 * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


def run_build(interim_dir, out_dir, max_len, seed):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    games = load_games(interim_dir)
    if not games:
        sys.exit(f"No games found in {interim_dir}")
    vocab = build_vocab(games)
    lengths = np.array([len(g["moves"]) for g in games])

    splits = dict(zip(("train", "val", "test"), split_indices(len(games), seed)))
    for name, idx in splits.items():
        subset = [games[i] for i in idx]
        np.save(out_dir / f"{name}.npy", encode(subset, vocab, max_len))
        np.save(out_dir / f"{name}_ratings.npy",
                np.array([[g["white"], g["black"]] for g in subset], dtype=np.int16))
        print(f"{name}: {len(subset)} games")

    with open(out_dir / "vocab.json", "w") as fh:
        json.dump({"pad_id": PAD_ID, "bos_id": BOS_ID, "max_len": max_len,
                   "vocab_size": len(vocab) + FIRST_MOVE_ID, "move_to_id": vocab},
                  fh)

    covered = float((lengths + 1 <= max_len).mean())
    print(f"\nGames: {len(games)}  Moves: {int(lengths.sum())}  "
          f"Distinct moves: {len(vocab)}  vocab_size (with PAD, BOS): "
          f"{len(vocab) + FIRST_MOVE_ID}")
    print(f"Mean game length: {lengths.mean():.1f} moves, max {lengths.max()}; "
          f"{covered:.1%} of games fit in max_len={max_len} without truncation")
    got = {"games": len(games), "moves": int(lengths.sum()),
           "distinct_moves": len(vocab)}
    for key, want in EXPECTED.items():
        flag = "OK" if got[key] == want else "MISMATCH"
        print(f"  {key}: got {got[key]}, proposal says {want}  [{flag}]")


# -----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter", help="filter raw archives to JSONL")
    f.add_argument("--raw", default="data/raw")
    f.add_argument("--interim", default="data/interim")
    f.add_argument("--workers", type=int, default=6)
    b = sub.add_parser("build", help="build vocab, split, and token arrays")
    b.add_argument("--interim", default="data/interim")
    b.add_argument("--out", default="data/processed")
    b.add_argument("--max-len", type=int, default=256)
    b.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.cmd == "filter":
        run_filter(args.raw, args.interim, args.workers)
    else:
        run_build(args.interim, args.out, args.max_len, args.seed)


if __name__ == "__main__":
    main()
