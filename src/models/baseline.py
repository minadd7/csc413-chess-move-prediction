"""Recurrent (LSTM/GRU) baseline for next-move prediction.

Follows the same contract as the transformer: a padded batch of token ids
``(B, T)`` goes in, next-token logits ``(B, T, V)`` come out. An RNN only reads
left to right, so the output at position t depends only on tokens 0..t, the
same causal setup as the transformer's masked attention.

Inputs must be right-padded (BOS first, PAD at the end), as in
``data/processed/``; left padding would make the RNN read PAD before BOS.
"""

import torch
from torch import nn


class RecurrentBaseline(nn.Module):
    """Token embedding -> multi-layer LSTM/GRU -> linear layer over the move vocab."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        cell: str = "lstm",
        pad_id: int = 0,
    ):
        super().__init__()
        rnn_classes = {"lstm": nn.LSTM, "gru": nn.GRU}
        if cell not in rnn_classes:
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")

        self.cell = cell
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = rnn_classes[cell](
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens: torch.Tensor, state=None, return_state: bool = False):
        """Return logits of shape ``(B, T, V)``.

        ``state`` and ``return_state`` let the demo feed one move at a time
        without re-reading the whole game. Training ignores them.
        """
        x = self.dropout(self.embed(tokens))
        out, new_state = self.rnn(x, state)
        logits = self.head(self.dropout(out))
        return (logits, new_state) if return_state else logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters, used to size the baseline near the transformer."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
