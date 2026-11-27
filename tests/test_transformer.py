import pytest
import torch

from models.transformer import ChessTransformer, count_parameters

VOCAB = 50


def small_model():
    torch.manual_seed(0)
    model = ChessTransformer(vocab_size=VOCAB, max_len=32, d_model=32, n_heads=4, n_layers=2)
    return model.eval()  # no dropout


def test_output_shape():
    model = small_model()
    tokens = torch.randint(0, VOCAB, (4, 10))
    assert model(tokens).shape == (4, 10, VOCAB)


def test_cannot_see_future_moves():
    # changing moves after position 5 shouldn't affect outputs at 0..5
    model = small_model()
    tokens = torch.randint(0, VOCAB, (2, 12))
    changed = tokens.clone()
    changed[:, 6:] = torch.randint(0, VOCAB, (2, 6))

    with torch.no_grad():
        out_a = model(tokens)
        out_b = model(changed)

    assert torch.allclose(out_a[:, :6], out_b[:, :6], atol=1e-5)
    assert not torch.allclose(out_a[:, 6:], out_b[:, 6:], atol=1e-5)


def test_right_padding_does_not_change_real_positions():
    model = small_model()
    pad_id = 0
    tokens = torch.randint(1, VOCAB, (1, 8))
    padded = torch.cat([tokens, torch.full((1, 5), pad_id)], dim=1)

    with torch.no_grad():
        out = model(tokens)
        out_padded = model(padded)

    assert torch.allclose(out, out_padded[:, :8], atol=1e-5)


def test_too_long_sequence_raises():
    model = small_model()
    with pytest.raises(ValueError):
        model(torch.zeros(1, 33, dtype=torch.long))


def test_overfits_one_batch():
    torch.manual_seed(0)
    model = ChessTransformer(vocab_size=VOCAB, max_len=32, d_model=64, n_heads=4, n_layers=2, dropout=0.0)
    tokens = torch.randint(0, VOCAB, (4, 16))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for _ in range(300):
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05


def test_count_parameters():
    assert count_parameters(small_model()) > 0
