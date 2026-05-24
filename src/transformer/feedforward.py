"""
Feed-Forward Networks — From Scratch
======================================
Implements:
  1. PositionWiseFFN   — standard ReLU FFN (Vaswani et al., baseline)
  2. GeLU_FFN          — BERT-style GeLU activation
  3. SwiGLU_FFN        — Gated Linear Unit with SiLU [MODIFICATION #5]
  4. MixtureOfExperts  — token-routing MoE (lightweight) [MODIFICATION #6]

Research Rationale:

  MODIFICATION #5 — SwiGLU Activation:
    Original FFN: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    SwiGLU FFN:   FFN(x) = (xW_1 ⊙ SiLU(xW_3)) W_2
    where SiLU(x) = x · σ(x) (Swish activation).

    SwiGLU (Noam Shazeer, 2020; used in PaLM, LLaMA, Mistral) consistently
    outperforms ReLU/GeLU by ~0.5–1% perplexity on language tasks because:
    (a) The gate (W_3 branch) can learn to suppress irrelevant activations.
    (b) SiLU's smooth gradient improves convergence stability.
    For classification, this translates to better feature selection.

  MODIFICATION #6 — Lightweight Mixture of Experts:
    Rather than routing ALL tokens to ALL FFN weights, we use 2–4 expert
    FFNs and a learned soft router to blend their outputs. This gives the
    model specialised sub-networks: one expert may handle named entities,
    another emotional language (common in fake news clickbait). We use
    soft routing (weighted sum) rather than hard routing (Top-1) for
    gradient stability during fine-tuning.

    Ablation hypothesis: MoE in the last 2 layers outperforms uniform FFN
    by ~1.3% F1, especially on WELFake's diverse domains.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Standard ReLU FFN (baseline)
# ---------------------------------------------------------------------------
class PositionWiseFFN(nn.Module):
    """
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    d_ff is typically 4 * d_model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 2. GeLU FFN (BERT-style)
# ---------------------------------------------------------------------------
class GeLU_FFN(nn.Module):
    """
    GeLU: x * Φ(x) where Φ is the Gaussian CDF.
    Used in BERT, GPT-2. Smoother gradient than ReLU.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(p=dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ---------------------------------------------------------------------------
# 3. SwiGLU FFN [MODIFICATION #5]
# ---------------------------------------------------------------------------
class SwiGLU_FFN(nn.Module):
    """
    SwiGLU: FFN(x) = (xW_1 ⊙ SiLU(xW_3)) W_2
    Note: uses 3 weight matrices. To keep param count ≈ standard FFN,
    we set d_ff = int(2/3 * 4 * d_model) which Shazeer recommends.

    SiLU(x) = x * sigmoid(x)  (also called Swish-1)

    Research reference:
      Shazeer, N. (2020). GLU Variants Improve Transformer.
      arXiv:2002.05202
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        # Recommended d_ff for SwiGLU to match param count of 4*d_model ReLU FFN
        if d_ff is None:
            d_ff = int(2 / 3 * 4 * d_model)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.W_gate = nn.Linear(d_model, d_ff, bias=False)  # gating branch
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)  # value branch
        self.W_down = nn.Linear(d_ff, d_model, bias=False)  # projection
        self.drop   = nn.Dropout(p=dropout)
        self.silu   = nn.SiLU()   # x * sigmoid(x)

        self._init_weights()

    def _init_weights(self):
        for w in [self.W_gate, self.W_up, self.W_down]:
            nn.init.kaiming_uniform_(w.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate: SiLU(xW_gate) ⊙ xW_up
        gate   = self.silu(self.W_gate(x))
        value  = self.W_up(x)
        hidden = gate * value                   # element-wise gating
        return self.drop(self.W_down(hidden))


# ---------------------------------------------------------------------------
# 4. Lightweight Mixture of Experts FFN [MODIFICATION #6]
# ---------------------------------------------------------------------------
class MoE_FFN(nn.Module):
    """
    Soft Mixture-of-Experts FFN.

    Each token is processed by ALL n_experts FFNs (SwiGLU), and the outputs
    are combined via a learned soft router (weighted sum, weights sum to 1).
    This differs from sparse MoE (top-1/top-2 routing) — we use soft routing
    for gradient stability.

    Architecture:
        router: Linear(d_model, n_experts) → softmax
        experts: [SwiGLU_FFN] * n_experts
        output: Σ_e  router_e * expert_e(x)

    For n_experts=4, each expert has d_ff = d_model (so total params ≈
    one standard 4*d_model FFN). Load balancing is achieved implicitly
    by the router's entropy regularization term in the loss.

    Note on paper contribution:
      We apply MoE only in the final 2 transformer layers (controlled by
      `use_moe` flag in TransformerLayer). This "top-layer MoE" strategy
      is computationally efficient and focuses expert specialization on
      high-level semantic features where it matters most.
    """
    def __init__(
        self,
        d_model   : int,
        n_experts : int   = 4,
        dropout   : float = 0.1,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        self.n_experts      = n_experts
        self.aux_loss_coeff = aux_loss_coeff

        # Expert d_ff: divide total capacity among experts
        expert_d_ff = max(d_model, (4 * d_model) // n_experts)

        self.experts = nn.ModuleList([
            SwiGLU_FFN(d_model, expert_d_ff, dropout) for _ in range(n_experts)
        ])
        self.router = nn.Linear(d_model, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        # Stored for aux loss computation
        self._last_router_logits: Optional[torch.Tensor] = None

    def get_aux_loss(self) -> torch.Tensor:
        """
        Load balancing auxiliary loss (Fedus et al., 2022 — Switch Transformer).
        Encourages uniform expert utilization.
        L_aux = n_experts * Σ_e (f_e * p_e)
        where f_e = fraction of tokens routed to expert e,
              p_e = mean router probability for expert e.
        """
        if self._last_router_logits is None:
            return torch.tensor(0.0)

        router_probs = F.softmax(self._last_router_logits, dim=-1)  # (B*T, n_experts)
        # f_e: mean probability assigned to each expert across tokens
        f_e = router_probs.mean(dim=0)   # (n_experts,)
        p_e = router_probs.mean(dim=0)   # same here for soft routing
        aux_loss = self.n_experts * (f_e * p_e).sum()
        return self.aux_loss_coeff * aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        x_flat = x.view(B * T, D)

        logits = self.router(x_flat)                      # (B*T, n_experts)
        self._last_router_logits = logits.detach()
        weights = F.softmax(logits, dim=-1)               # (B*T, n_experts)

        # Compute expert outputs
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1   # (B, T, D, n_experts)
        )

        # Weighted sum: weights shape (B*T, n_experts) → (B, T, 1, n_experts)
        weights_reshaped = weights.view(B, T, 1, self.n_experts)
        output = (expert_outs * weights_reshaped).sum(dim=-1)   # (B, T, D)
        return output
