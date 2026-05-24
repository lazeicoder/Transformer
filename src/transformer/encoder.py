"""
Transformer Encoder — From Scratch
=====================================
Implements the full encoder stack with all research modifications.

Architecture Overview (our modified Transformer):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Input → HybridEmbedding (Token + Segment, no absolute pos)    │
  │                      ↓                                         │
  │  [TransformerLayer × N]                                        │
  │   • Layers 0..N/2-1 : Dense MHA + RoPE + SwiGLU (Pre-LN)      │
  │   • Layers N/2..N-1 : Sparse MHA + RoPE + MoE-SwiGLU (Pre-LN) │
  │                      ↓                                         │
  │  CrossSentenceGate (CSCG) [MODIFICATION #4]                    │
  │                      ↓                                         │
  │  ClassificationHead (multi-pool: CLS + mean + max)             │
  └─────────────────────────────────────────────────────────────────┘

Normalization Strategy [MODIFICATION #7 — Pre-LN]:
  Original "Post-LN" (Vaswani 2017): LayerNorm applied AFTER residual.
  "Pre-LN" (Xiong et al., 2020): LayerNorm applied BEFORE sub-layer.
  Pre-LN is more stable, requires no warmup scheduling, and converges
  faster on smaller datasets — crucial for academic settings where
  training on a single GPU.
  Reference: Xiong et al. "On Layer Normalization in the Transformer
  Encoder." ICML 2020.

Modifications Summary (for paper Section 3):
  M1 — Rotary Positional Encoding (RoPE) in attention heads
  M2 — Sparse Top-K attention in upper layers
  M3 — Hierarchical Sparse Routing (HSR): dense lower / sparse upper
  M4 — Cross-Sentence Consistency Gate (CSCG) before classifier
  M5 — SwiGLU feed-forward activation
  M6 — Mixture-of-Experts FFN in upper layers
  M7 — Pre-Layer Normalization (Pre-LN)
  M8 — Multi-pool classification head (CLS + mean + max)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict

from .attention   import MultiHeadAttention, SparseMultiHeadAttention, CrossSentenceGate
from .feedforward import SwiGLU_FFN, MoE_FFN
from .embeddings  import HybridEmbedding


# ---------------------------------------------------------------------------
# Encoder Layer (Pre-LN) with HSR logic
# ---------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    """
    Single Pre-LN Transformer encoder layer.

    Pre-LN order:  x → LN → SubLayer → + x  (residual added after sub-layer)

    Args:
        d_model    : model dimension
        num_heads  : attention heads
        d_ff       : feed-forward inner dimension (None → SwiGLU auto-size)
        dropout    : dropout rate
        max_len    : max sequence length for RoPE
        use_sparse : use SparseMultiHeadAttention instead of dense [M2/M3]
        use_moe    : use MoE FFN instead of single SwiGLU [M6]
        n_experts  : number of MoE experts (if use_moe)
        layer_idx  : layer index (for naming attention stores)
    """
    def __init__(
        self,
        d_model   : int,
        num_heads : int,
        d_ff      : Optional[int] = None,
        dropout   : float = 0.1,
        max_len   : int   = 512,
        use_sparse: bool  = False,
        use_moe   : bool  = False,
        n_experts : int   = 4,
        layer_idx : int   = 0,
    ):
        super().__init__()
        store_name = f"layer_{layer_idx}"

        # Self-attention [M1, M2, M3]
        if use_sparse:
            self.self_attn = SparseMultiHeadAttention(
                d_model, num_heads, dropout, max_len,
                sparse_k_ratio=0.125, store_name=store_name
            )
        else:
            self.self_attn = MultiHeadAttention(
                d_model, num_heads, dropout, max_len,
                sparse_k=None, store_name=store_name
            )

        # Feed-forward [M5, M6]
        if use_moe:
            self.ffn = MoE_FFN(d_model, n_experts, dropout)
        else:
            self.ffn = SwiGLU_FFN(d_model, d_ff, dropout)

        # Pre-LN normalization [M7]
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop  = nn.Dropout(p=dropout)

        self.use_moe = use_moe

    def forward(
        self,
        x    : torch.Tensor,                    # (B, T, d_model)
        mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x            : (B, T, d_model)
            attn_weights : (B, T, T) averaged heads
        """
        # --- Pre-LN Self-Attention sub-layer ---
        residual       = x
        x_norm         = self.norm1(x)
        attn_out, attn_w = self.self_attn(x_norm, mask)
        x              = residual + self.drop(attn_out)

        # --- Pre-LN FFN sub-layer ---
        residual       = x
        x_norm         = self.norm2(x)
        ffn_out        = self.ffn(x_norm)
        x              = residual + self.drop(ffn_out)

        return x, attn_w

    def get_moe_aux_loss(self) -> torch.Tensor:
        if self.use_moe:
            return self.ffn.get_aux_loss()
        return torch.tensor(0.0, device=next(self.parameters()).device)


# ---------------------------------------------------------------------------
# Full Transformer Encoder Stack
# ---------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    """
    Full encoder stack with Hierarchical Sparse Routing [M3].

    HSR Strategy:
        First half of layers: dense attention + SwiGLU
        Second half of layers: sparse attention + MoE-SwiGLU
    
    This mirrors the linguistic hierarchy:
        Lower layers  → local syntactic patterns (attend broadly)
        Upper layers  → global semantic claims (attend selectively)

    Final CSCG gate [M4] applied before returning CLS representation.
    """
    def __init__(
        self,
        vocab_size : int,
        d_model    : int  = 256,
        num_heads  : int  = 8,
        num_layers : int  = 6,
        d_ff       : Optional[int] = None,
        dropout    : float = 0.1,
        max_len    : int   = 512,
        n_segments : int   = 3,
        n_experts  : int   = 4,
        padding_idx: int   = 0,
    ):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers

        # Embedding [M1 pre-condition: no absolute PE in embedding]
        self.embedding = HybridEmbedding(
            vocab_size, d_model, max_len, n_segments, dropout, padding_idx
        )

        # Encoder layers with HSR [M2, M3, M5, M6, M7]
        sparse_start = num_layers // 2   # upper half uses sparse + MoE
        self.layers  = nn.ModuleList([
            TransformerEncoderLayer(
                d_model   = d_model,
                num_heads = num_heads,
                d_ff      = d_ff,
                dropout   = dropout,
                max_len   = max_len,
                use_sparse= (i >= sparse_start),   # M3: HSR
                use_moe   = (i >= sparse_start),   # M6: MoE upper layers
                n_experts = n_experts,
                layer_idx = i,
            )
            for i in range(num_layers)
        ])

        # Final normalization (Pre-LN style: norm after last layer)
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)

        # Cross-Sentence Consistency Gate [M4]
        self.cscg = CrossSentenceGate(d_model)

        self._init_output_projections(num_layers)

    def _init_output_projections(self, num_layers: int):
        """
        Scale W_o in each attention layer by 1/sqrt(2*num_layers).
        This prevents representation collapse in deep stacks.
        (Wang et al., 2022 — DeepNet)
        """
        scale = (2 * num_layers) ** -0.5
        for layer in self.layers:
            with torch.no_grad():
                layer.self_attn.W_o.weight.mul_(scale)

    def build_padding_mask(
        self,
        token_ids    : torch.Tensor,  # (B, T)
        padding_idx  : int = 0,
    ) -> torch.Tensor:
        """
        Returns (B, 1, 1, T) float mask: 1.0 for real tokens, 0.0 for padding.
        Used inside scaled_dot_product_attention.
        """
        return (token_ids != padding_idx).unsqueeze(1).unsqueeze(2).float()

    def get_all_moe_aux_loss(self) -> torch.Tensor:
        """Sum of MoE auxiliary losses across all layers."""
        total = torch.tensor(0.0)
        for layer in self.layers:
            loss = layer.get_moe_aux_loss()
            total = total + loss.cpu()
        return total

    def forward(
        self,
        token_ids    : torch.Tensor,                    # (B, T)
        segment_ids  : Optional[torch.Tensor] = None,  # (B, T)
        mask         : Optional[torch.Tensor] = None,  # override mask
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
            'last_hidden'  : (B, T, d_model) — all token representations
            'cls_hidden'   : (B, d_model)    — [CLS] token (pos 0)
            'mean_hidden'  : (B, d_model)    — mean-pooled (excl. padding)
            'max_hidden'   : (B, d_model)    — max-pooled
            'attn_weights' : List[(B, T, T)] — per-layer attention weights
        """
        if mask is None:
            mask = self.build_padding_mask(token_ids)   # (B, 1, 1, T)

        x = self.embedding(token_ids, segment_ids)     # (B, T, d_model)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            attn_weights_all.append(attn_w)

        x = self.final_norm(x)

        # Cross-Sentence Consistency Gate [M4]
        x = self.cscg(x)

        # Multi-pool [M8]
        cls_hidden  = x[:, 0, :]               # (B, d_model)

        # Mean pool over non-padding tokens
        pad_mask = mask.squeeze(1).squeeze(1)  # (B, T) — 1.0 real, 0.0 pad
        pad_mask_exp = pad_mask.unsqueeze(-1)  # (B, T, 1)
        mean_hidden = (x * pad_mask_exp).sum(dim=1) / pad_mask_exp.sum(dim=1).clamp(min=1e-9)

        # Max pool
        x_masked     = x.masked_fill(pad_mask_exp == 0, -1e9)
        max_hidden   = x_masked.max(dim=1).values   # (B, d_model)

        return {
            'last_hidden' : x,
            'cls_hidden'  : cls_hidden,
            'mean_hidden' : mean_hidden,
            'max_hidden'  : max_hidden,
            'attn_weights': attn_weights_all,
        }


# ---------------------------------------------------------------------------
# Classification Head [MODIFICATION #8 — Multi-Pool]
# ---------------------------------------------------------------------------
class MultiPoolClassificationHead(nn.Module):
    """
    MODIFICATION #8 — Multi-Pool Classification Head.

    Rather than using only [CLS], we concatenate:
      - CLS token   : global document representation (BERT-style)
      - Mean pool   : robust average over all content tokens
      - Max pool    : strongest activated feature per dimension

    This gives a 3*d_model input to the classifier. The intuition:
      CLS  → "what is the overall claim?"
      Mean → "what is the average tone/style?"
      Max  → "what is the strongest signal word?"

    For fake news: clickbait words, ALL-CAPS phrases, emotionally loaded
    language create strong max-pool activations that pure CLS misses.

    Architecture:
        [CLS ; mean ; max] → Linear → LayerNorm → GELU → Dropout → Linear
    """
    def __init__(self, d_model: int, n_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.fc2  = nn.Linear(d_model, n_classes)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        cls_hidden : torch.Tensor,   # (B, d_model)
        mean_hidden: torch.Tensor,
        max_hidden : torch.Tensor,
    ) -> torch.Tensor:
        pooled  = torch.cat([cls_hidden, mean_hidden, max_hidden], dim=-1)  # (B, 3*d_model)
        hidden  = self.act(self.norm(self.fc1(pooled)))
        hidden  = self.drop(hidden)
        return self.fc2(hidden)      # (B, n_classes) — raw logits


# ---------------------------------------------------------------------------
# Full Model: FakeNewsTransformer
# ---------------------------------------------------------------------------
class FakeNewsTransformer(nn.Module):
    """
    The complete model. Combines TransformerEncoder + MultiPoolClassificationHead.
    This is what gets trained, saved, and deployed.

    Total parameters at default config (d_model=256, 6 layers, 8 heads):
        ≈ 18–22M parameters — lightweight enough for a single GPU/Colab.

    For a stronger research setup (d_model=512, 12 layers):
        ≈ 85M parameters — comparable to BERT-base.
    """
    def __init__(
        self,
        vocab_size  : int,
        d_model     : int   = 256,
        num_heads   : int   = 8,
        num_layers  : int   = 6,
        d_ff        : Optional[int] = None,
        dropout     : float = 0.1,
        max_len     : int   = 512,
        n_segments  : int   = 3,
        n_experts   : int   = 4,
        n_classes   : int   = 2,
        padding_idx : int   = 0,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder    = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff,
            dropout, max_len, n_segments, n_experts, padding_idx
        )
        self.classifier = MultiPoolClassificationHead(
            d_model, n_classes, classifier_dropout
        )

    def forward(
        self,
        token_ids   : torch.Tensor,
        segment_ids : Optional[torch.Tensor] = None,
        mask        : Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            'logits'       : (B, n_classes)
            'cls_hidden'   : (B, d_model)    — for contrastive probing
            'attn_weights' : List[(B,T,T)]   — for visualization / SHAP
            'moe_aux_loss' : scalar          — added to training loss
        """
        enc_out  = self.encoder(token_ids, segment_ids, mask)

        logits   = self.classifier(
            enc_out['cls_hidden'],
            enc_out['mean_hidden'],
            enc_out['max_hidden'],
        )

        moe_loss = self.encoder.get_all_moe_aux_loss()

        return {
            'logits'      : logits,
            'cls_hidden'  : enc_out['cls_hidden'],
            'attn_weights': enc_out['attn_weights'],
            'moe_aux_loss': moe_loss,
        }

    def count_parameters(self) -> Dict[str, int]:
        total    = sum(p.numel() for p in self.parameters())
        trainable= sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
