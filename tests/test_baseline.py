"""Checks for the LSTM/GRU baseline, run on random tokens (no data needed).

Run from the repo root: ``python -m pytest tests``
"""

import pytest
import torch
import torch.nn.functional as F

from src.models.baseline import RecurrentBaseline, count_parameters

VOCAB = 1899  # matches data/processed/vocab.json (1,897 moves + PAD + BOS)
PAD, BOS = 0, 1


def random_batch(batch=4, length=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(2, VOCAB, (batch, length), generator=g)
    tokens[:, 0] = BOS
    tokens[:, -5:] = PAD  # padded tail, like shorter games
    return tokens


def next_token_loss(model, tokens):
    logits = model(tokens[:, :-1])
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1), ignore_index=PAD
    )


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_output_shape(cell):
    model = RecurrentBaseline(VOCAB, cell=cell)
    tokens = random_batch()
    assert model(tokens).shape == (4, 32, VOCAB)


def test_is_causal():
    """Changing a later move must not change predictions for earlier positions."""
    model = RecurrentBaseline(VOCAB).eval()
    a = random_batch()
    b = a.clone()
    b[:, 20:] = 7
    with torch.no_grad():
        assert torch.allclose(model(a)[:, :20], model(b)[:, :20])


def test_step_by_step_matches_full_sequence():
    """Feeding moves one at a time (demo use) gives the same logits as one full pass."""
    model = RecurrentBaseline(VOCAB).eval()
    tokens = random_batch(batch=1, length=10)
    with torch.no_grad():
        full = model(tokens)
        state, steps = None, []
        for t in range(tokens.size(1)):
            logits, state = model(tokens[:, t : t + 1], state, return_state=True)
            steps.append(logits)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_bad_cell_rejected():
    with pytest.raises(ValueError):
        RecurrentBaseline(VOCAB, cell="rnn")


def test_parameter_count_default_size():
    """Default size is ~5M parameters; change it to sit within ±20% of the transformer."""
    n = count_parameters(RecurrentBaseline(VOCAB))
    assert 4_000_000 < n < 6_500_000


@pytest.mark.parametrize("cell", ["lstm", "gru"])
def test_overfits_one_batch(cell):
    torch.manual_seed(0)
    model = RecurrentBaseline(
        VOCAB, embed_dim=128, hidden_dim=256, dropout=0.0, cell=cell
    )
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    tokens = random_batch(batch=8, length=32)
    for _ in range(300):
        loss = next_token_loss(model, tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.1:
            break
    assert loss.item() < 0.1
