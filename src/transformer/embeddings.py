"""
Embeddings Module — From Scratch
=================================
Implements:
  1. TokenEmbedding           — standard learnable token lookup
  2. SinusoidalPositionalEncoding  — original Vaswani et al. 2017
  3. LearnablePositionalEncoding   — BERT-style (our choice)
  4. RotaryPositionalEncoding (RoPE) — Su et al. 2021 [MODIFICATION #1]
  5. HybridEmbedding          — combines token + rotary + segment embeddings

Research Rationale (MODIFICATION #1 — RoPE):
  Standard sinusoidal or learnable absolute position embeddings lose
  relative position information when sequences are truncated or when
  two tokens are compared across long documents. RoPE encodes position
  as a rotation in complex space, so the dot-product attention naturally
  captures relative distances. This is especially useful for news articles
  which vary widely in length (50–800 tokens). Papers like LLaMA and
  PaLM use RoPE; we adapt it for classification.

  Expected gain: +1.5–2% on long-article samples, better calibration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Token Embedding
# ---------------------------------------------------------------------------
class TokenEmbedding(nn.Module):
    """
    Standard learnable lookup table.
    Shape: (vocab_size, d_model)
    Weight is scaled by sqrt(d_model) as in Vaswani et al. to keep
    embedding norms from dominating early training.
    """
    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        if self.embedding.padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[self.embedding.padding_idx].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T)  →  out: (B, T, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


# ---------------------------------------------------------------------------
# 2. Sinusoidal Positional Encoding (Vaswani et al., 2017 — baseline)
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Not learnable; extrapolates beyond training length.
    Kept here for ablation studies (compare vs RoPE).
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe    = torch.zeros(max_len, d_model)
        pos   = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div   = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                    # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# 3. Learnable Positional Encoding (BERT-style — baseline)
# ---------------------------------------------------------------------------
class LearnablePositionalEncoding(nn.Module):
    """
    Fully learnable position embeddings (BERT/GPT style).
    Limited to max_len; does not extrapolate beyond training.
    Used in ablation studies.
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout  = nn.Dropout(p=dropout)
        self.pos_emb  = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return self.dropout(x + self.pos_emb(positions))


# ---------------------------------------------------------------------------
# 4. Rotary Positional Encoding — RoPE [MODIFICATION #1]
# ---------------------------------------------------------------------------
class RotaryPositionalEncoding(nn.Module):
    """
    RoPE: Rotary Position Embedding (Su et al., 2021)
    ArXiv: https://arxiv.org/abs/2104.09864

    Key idea: Instead of adding position info to embeddings, we ROTATE
    the query and key vectors in attention by a position-dependent angle.
    The inner product <Rθ_m · q, Rθ_n · k> = <q, R(θ_n−θ_m) · k>
    which means attention scores depend on RELATIVE position (m-n),
    not absolute position. Perfect for variable-length news articles.

    Applied inside attention (not as a standalone layer), but we pre-
    compute the cos/sin cache here for efficiency.

    Args:
        head_dim: dimension per attention head
        max_len : maximum sequence length
        base    : theta base (10000 in original; we use 10000)
    """
    def __init__(self, head_dim: int, max_len: int = 512, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim

        # Precompute inverse frequencies: θ_i = 1 / 10000^(2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin cache
        self._build_cache(max_len)

    def _build_cache(self, max_len: int):
        t      = torch.arange(max_len, device=self.inv_freq.device).float()
        freqs  = torch.outer(t, self.inv_freq)       # (max_len, head_dim/2)
        emb    = torch.cat([freqs, freqs], dim=-1)   # (max_len, head_dim)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x1, x2, ..., xd] → [-x_{d/2+1},...,-xd, x1,...,x_{d/2}]"""
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, num_heads, T, head_dim)
        Returns x rotated by position-dependent angles.
        """
        seq_len = x.size(2)
        cos = self.cos_cache[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,T,head_dim)
        sin = self.sin_cache[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE to both query and key tensors."""
        return self.apply_rope(q), self.apply_rope(k)


# ---------------------------------------------------------------------------
# 5. Segment Embedding (for fake/real dual-stream — MODIFICATION #2 hint)
# ---------------------------------------------------------------------------
class SegmentEmbedding(nn.Module):
    """
    Learnable segment (type) embedding.
    We use 3 segments: 0=headline, 1=body, 2=metadata (source/date).
    This lets the model learn different representations per article section.
    Inspired by BERT's sentence A/B embeddings.
    """
    def __init__(self, d_model: int, n_segments: int = 3):
        super().__init__()
        self.embedding = nn.Embedding(n_segments, d_model)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(segment_ids)


# ---------------------------------------------------------------------------
# 6. Hybrid Embedding — the one we actually use
# ---------------------------------------------------------------------------
class HybridEmbedding(nn.Module):
    """
    Combines: TokenEmbedding + SegmentEmbedding + LayerNorm + Dropout
    RoPE is NOT added here — it is applied inside attention heads directly
    to Q and K, preserving the relative-position property.

    Design choice: We keep the embedding module clean (no absolute position
    bias) and rely entirely on RoPE inside attention. This is the modern
    approach used in LLaMA-2, Mistral, Falcon etc.
    """
    def __init__(
        self,
        vocab_size   : int,
        d_model      : int,
        max_len      : int  = 512,
        n_segments   : int  = 3,
        dropout      : float = 0.1,
        padding_idx  : int  = 0,
    ):
        super().__init__()
        self.token_emb   = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.segment_emb = SegmentEmbedding(d_model, n_segments)
        self.norm        = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout     = nn.Dropout(p=dropout)

    def forward(
        self,
        token_ids   : torch.Tensor,              # (B, T)
        segment_ids : torch.Tensor | None = None # (B, T)
    ) -> torch.Tensor:
        x = self.token_emb(token_ids)
        if segment_ids is not None:
            x = x + self.segment_emb(segment_ids)
        return self.dropout(self.norm(x))
