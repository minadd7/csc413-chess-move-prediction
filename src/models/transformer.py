"""Causal transformer for next-move prediction.

tokens [B, T] -> logits [B, T, vocab_size]. Padding has to be on the right
(the causal mask then keeps real positions from attending to it).
"""

import torch
import torch.nn as nn


class ChessTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=256, d_model=256, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # pre-norm needs a final layer norm
        self.blocks = nn.TransformerEncoder(
            block, num_layers=n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        seq_len = tokens.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} > max_len {self.max_len}")

        pos = torch.arange(seq_len, device=tokens.device)
        h = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(h)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
