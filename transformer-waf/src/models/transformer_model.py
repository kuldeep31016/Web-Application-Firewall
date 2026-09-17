from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class WAFTransformer(nn.Module):
    """Transformer-based autoencoder for sequence reconstruction."""

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, ff_dim: int = 512, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        # enable_nested_tensor=False: the "fast path" only changes speed, not results, and emits prototype warnings
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
        """Run embedding + encoder and return hidden states (bsz, seqlen, embed_dim).

        Shared by the reconstruction objective below and by the supervised URL
        classifier, which reuses this encoder with a pooled classification head.
        """
        bsz, seqlen = input_ids.size()
        positions = torch.arange(0, seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.embed(input_ids) + self.positional(positions)
        key_padding_mask = attention_mask == 0
        src_mask = self._causal_mask(seqlen, input_ids.device) if causal else None
        return self.encoder(x, mask=src_mask, src_key_padding_mask=key_padding_mask)

    @staticmethod
    def _causal_mask(seqlen: int, device: torch.device) -> torch.Tensor:
        # Boolean mask (True = blocked) so it matches the boolean key_padding_mask type
        return torch.triu(torch.ones((seqlen, seqlen), dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return next-token logits: logits[:, t] predicts input_ids[:, t + 1].

        Encoder, decoder self-attention and decoder cross-attention are all
        causally masked, so position t only ever sees tokens <= t. Without this
        the decoder received the token it had to reconstruct and learned a
        trivial copy, which gave no separation between benign and attack
        traffic.
        """
        bsz, seqlen = input_ids.size()
        positions = torch.arange(0, seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, seqlen)
        x = self.embed(input_ids) + self.positional(positions)
        key_padding_mask = attention_mask == 0
        causal = self._causal_mask(seqlen, input_ids.device)
        memory = self.encoder(x, mask=causal, src_key_padding_mask=key_padding_mask)
        decoded = self.decoder(
            x,
            memory,
            tgt_mask=causal,
            memory_mask=causal,
            tgt_key_padding_mask=key_padding_mask,
            memory_key_padding_mask=key_padding_mask,
        )
        logits = self.proj(decoded)
        return logits

    @staticmethod
    def sequence_nll(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-sequence mean next-token negative log-likelihood (the anomaly score)."""
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        pred = logits[:, :-1, :]
        target = input_ids[:, 1:]
        mask = attention_mask[:, 1:].float()
        bsz, seqlen, vocab = pred.size()
        loss = loss_fn(pred.reshape(bsz * seqlen, vocab), target.reshape(-1)).view(bsz, seqlen)
        return (loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    @torch.no_grad()
    def get_reconstruction_error(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids, attention_mask)
        return self.sequence_nll(logits, input_ids, attention_mask)


class AnomalyDetector:
    """Wraps a model and exposes thresholded anomaly detection."""

    def __init__(self, model: WAFTransformer, threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def set_threshold(self, threshold: float) -> None:
        self.threshold = float(threshold)

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        err = self.model.get_reconstruction_error(input_ids, attention_mask)
        is_anom = err > self.threshold
        return is_anom, err


__all__ = ["WAFTransformer", "AnomalyDetector"]


