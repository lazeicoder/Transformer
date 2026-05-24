"""
Explainability & Interpretability — From Scratch
==================================================
Implements:
  1. Attention Rollout (Abnar & Zuidema, 2020)
  2. Gradient × Input Saliency
  3. Integrated Gradients (Sundararajan et al., 2017)
  4. Attention Head Analysis (per-head entropy, specialization)
  5. Token Attribution Visualization (HTML heatmap)
  6. Reliability Diagram (for calibration visualization)

Why explainability matters for this paper:
  Fake news detection is a high-stakes decision system. Any paper claiming
  SOTA must also provide qualitative evidence that the model is attending
  to the right signals (linguistic cues, named entities, emotional language)
  and not exploiting spurious correlations (source name, article length).
  Reviewers at ACL/EMNLP/AAAI now routinely request error analysis and
  attention/saliency visualizations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# 1. Attention Rollout (Abnar & Zuidema, 2020)
# ===========================================================================
class AttentionRollout:
    """
    Attention Rollout: propagates attention across layers to produce a
    single token-attribution map for the [CLS] token.

    Algorithm:
      Let A_l = attention_matrix of layer l (averaged across heads).
      Add residual: Â_l = 0.5 * A_l + 0.5 * I
      Normalize rows: Â_l = Â_l / Â_l.sum(axis=-1, keepdims=True)
      Rollout = Â_1 @ Â_2 @ ... @ Â_L
      Attribution = Rollout[0, :]  (row corresponding to [CLS] token)

    Reference:
      Abnar & Zuidema (2020). Quantifying Attention Flow in Transformers.
      ACL 2020. arXiv:2005.00928
    """
    def __init__(self, n_classes: int = 2, head_fusion: str = 'mean'):
        """
        head_fusion: 'mean' | 'max' | 'min'
          How to aggregate across attention heads before rollout.
        """
        assert head_fusion in ('mean', 'max', 'min')
        self.head_fusion = head_fusion

    def compute(
        self,
        attn_weights_per_layer: List[torch.Tensor],  # each (B, T, T) — already head-merged
    ) -> torch.Tensor:
        """
        attn_weights_per_layer: list of (B, T, T) matrices, one per layer.
        Returns attribution: (B, T) — attribution score of each token for [CLS].
        """
        T = attn_weights_per_layer[0].size(-1)
        B = attn_weights_per_layer[0].size(0)

        rollout = torch.eye(T, device=attn_weights_per_layer[0].device).unsqueeze(0).expand(B, -1, -1)

        for attn in attn_weights_per_layer:
            # Add residual connection: (A + I) / 2
            attn_aug = 0.5 * attn + 0.5 * torch.eye(T, device=attn.device).unsqueeze(0)
            # Row-normalize
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            # Matrix multiply into rollout
            rollout  = torch.bmm(attn_aug, rollout)

        # Attribution for [CLS] token (position 0)
        return rollout[:, 0, :]   # (B, T)


# ===========================================================================
# 2. Gradient × Input Saliency (from scratch)
# ===========================================================================
class GradientSaliency:
    """
    Gradient × Input saliency: computes the gradient of the predicted class
    score with respect to the input token embeddings, then multiplies by
    the embedding values and takes the L2 norm per token.

    saliency_i = ||∂y/∂e_i ⊙ e_i||₂

    This is one of the most widely used gradient-based explanation methods.
    Reference:
      Simonyan et al. (2014). Deep Inside Convolutional Networks.
    """
    def __init__(self, model: nn.Module, embedding_layer: nn.Module):
        self.model           = model
        self.embedding_layer = embedding_layer

    def compute(
        self,
        token_ids    : torch.Tensor,   # (1, T)
        segment_ids  : torch.Tensor,   # (1, T)
        target_class : int,
    ) -> np.ndarray:
        """
        Returns saliency scores: (T,) numpy array.
        """
        self.model.eval()

        # Enable gradient on embedding output
        emb  = self.embedding_layer(token_ids, segment_ids)  # (1, T, d_model)
        emb.retain_grad()

        # Forward pass with embedding as input (requires custom forward)
        # We hook into the model's embedding output
        out  = self.model(token_ids, segment_ids)
        logits = out['logits']   # (1, n_classes)

        # Backward pass for target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        if emb.grad is None:
            raise RuntimeError("Gradient not computed. Ensure inputs require_grad.")

        # Gradient × Input, then L2 norm over d_model
        grad_x_input = (emb.grad * emb).detach().cpu()    # (1, T, d_model)
        saliency     = grad_x_input.norm(dim=-1)[0]        # (T,)
        return saliency.numpy()


# ===========================================================================
# 3. Integrated Gradients (from scratch)
# ===========================================================================
class IntegratedGradients:
    """
    Integrated Gradients (Sundararajan et al., 2017).

    IG_i = (x_i - x'_i) × ∫₀¹ ∂F(x' + α(x-x'))/∂x_i  dα

    Approximated via Riemann sum with `steps` steps.
    Baseline x' is typically the zero-embedding (all-zeros) or padding token.

    Reference:
      Sundararajan, Taly, Yan (2017). Axiomatic Attribution for Deep Networks.
      ICML 2017. arXiv:1703.01365
    """
    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model   = model
        self.n_steps = n_steps

    def compute(
        self,
        token_ids    : torch.Tensor,   # (1, T)
        segment_ids  : torch.Tensor,   # (1, T)
        target_class : int,
        baseline_ids : Optional[torch.Tensor] = None,  # (1, T) — default: all zeros
        device       : torch.device = torch.device('cpu'),
    ) -> np.ndarray:
        """
        Returns integrated gradient attributions: (T,) numpy array.
        """
        if baseline_ids is None:
            baseline_ids = torch.zeros_like(token_ids)

        self.model.eval()
        token_ids   = token_ids.to(device)
        segment_ids = segment_ids.to(device)
        baseline_ids= baseline_ids.to(device)

        # Get embedding of input and baseline
        with torch.no_grad():
            # We need raw embeddings — access encoder.embedding
            emb_input    = self.model.encoder.embedding(token_ids,    segment_ids)
            emb_baseline = self.model.encoder.embedding(baseline_ids, segment_ids)

        # Compute gradients at each interpolation point
        integrated = torch.zeros_like(emb_input)

        for step in range(1, self.n_steps + 1):
            alpha      = step / self.n_steps
            emb_alpha  = emb_baseline + alpha * (emb_input - emb_baseline)
            emb_alpha.requires_grad_(True)

            # We need a custom forward that takes embeddings directly
            # — pass through encoder layers
            logits = self._forward_from_embedding(emb_alpha, token_ids, segment_ids)
            score  = logits[0, target_class]

            self.model.zero_grad()
            score.backward()

            if emb_alpha.grad is not None:
                integrated += emb_alpha.grad.detach()

        integrated    = integrated / self.n_steps
        ig_scores     = ((emb_input - emb_baseline) * integrated).detach().cpu()
        attributions  = ig_scores.norm(dim=-1)[0].numpy()   # (T,)
        return attributions

    def _forward_from_embedding(
        self,
        emb         : torch.Tensor,
        token_ids   : torch.Tensor,
        segment_ids : torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass starting from pre-computed embeddings."""
        enc   = self.model.encoder
        mask  = enc.build_padding_mask(token_ids)
        x     = emb
        for layer in enc.layers:
            x, _ = layer(x, mask)
        x       = enc.final_norm(x)
        x       = enc.cscg(x)
        # Multi-pool
        pad_mask_exp = mask.squeeze(1).squeeze(1).unsqueeze(-1)
        cls_h  = x[:, 0, :]
        mean_h = (x * pad_mask_exp).sum(1) / pad_mask_exp.sum(1).clamp(min=1e-9)
        max_h  = x.masked_fill(pad_mask_exp == 0, -1e9).max(1).values
        return self.model.classifier(cls_h, mean_h, max_h)


# ===========================================================================
# 4. Attention Head Analysis
# ===========================================================================
class AttentionHeadAnalyzer:
    """
    Analyzes individual attention head behavior.
    Metrics:
      - Head entropy: H(A_h) = -Σ_j A_h[i,j] * log(A_h[i,j])
        Low entropy → head is focused (attends to specific tokens)
        High entropy → head is diffuse (attends broadly)
      - Maximum attention weight position (what does each head attend to?)
      - Diagonal attention ratio (self-attention tendency)
    """

    @staticmethod
    def head_entropy(attn_matrix: torch.Tensor) -> torch.Tensor:
        """
        attn_matrix: (B, H, T, T)
        Returns entropy per head: (B, H)
        """
        # Clamp to avoid log(0)
        p     = attn_matrix.clamp(min=1e-9)
        logp  = torch.log(p)
        ent   = -(p * logp).sum(dim=-1)   # (B, H, T)
        return ent.mean(dim=-1)            # (B, H) — mean over query positions

    @staticmethod
    def diagonal_ratio(attn_matrix: torch.Tensor) -> torch.Tensor:
        """
        Fraction of attention mass on diagonal (self-attention tendency).
        attn_matrix: (B, H, T, T)
        Returns (B, H).
        """
        B, H, T, _ = attn_matrix.size()
        diag   = torch.diagonal(attn_matrix, dim1=-2, dim2=-1)  # (B, H, T)
        return diag.mean(dim=-1)   # (B, H)

    @staticmethod
    def cls_attention(attn_matrix: torch.Tensor) -> torch.Tensor:
        """
        How much does [CLS] (row 0) attend to each token?
        attn_matrix: (B, H, T, T)
        Returns (B, H, T).
        """
        return attn_matrix[:, :, 0, :]   # attention FROM [CLS] TO all tokens

    def full_analysis(
        self,
        attn_weights_per_layer: List[torch.Tensor],   # each (B, H, T, T) or (B, T, T)
    ) -> Dict[str, torch.Tensor]:
        """
        Runs all head-level metrics across all layers.
        Returns dict of {metric_name: tensor}.
        """
        results = {}
        for l_idx, attn in enumerate(attn_weights_per_layer):
            if attn.dim() == 3:
                attn = attn.unsqueeze(1)   # treat as 1 head if already merged
            results[f'layer{l_idx}_entropy']   = self.head_entropy(attn)
            results[f'layer{l_idx}_diag_ratio']= self.diagonal_ratio(attn)
        return results


# ===========================================================================
# 5. HTML Token Attribution Heatmap
# ===========================================================================
def render_token_heatmap(
    tokens      : List[str],
    scores      : np.ndarray,
    title       : str = "Token Attribution",
    save_path   : Optional[str] = None,
) -> str:
    """
    Renders an HTML heatmap of token attributions.
    Colors range from white (low) to deep coral (high).
    Returns HTML string; optionally saves to file.
    """
    # Normalize scores to [0, 1]
    scores = np.array(scores, dtype=float)
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        scores_norm = (scores - s_min) / (s_max - s_min)
    else:
        scores_norm = np.zeros_like(scores)

    def score_to_color(s: float) -> str:
        # White (1,1,1) → Coral (#D85A30 = 216,90,48)
        r = int(255 - (255 - 216) * s)
        g = int(255 - (255 - 90)  * s)
        b = int(255 - (255 - 48)  * s)
        return f'rgb({r},{g},{b})'

    def text_color(s: float) -> str:
        return '#4A1B0C' if s > 0.5 else '#2C2C2A'

    spans = []
    for tok, sc, sc_n in zip(tokens, scores, scores_norm):
        bg  = score_to_color(sc_n)
        fg  = text_color(sc_n)
        tip = f'{sc:.4f}'
        spans.append(
            f'<span style="background:{bg};color:{fg};padding:2px 4px;'
            f'margin:2px;border-radius:3px;font-family:monospace;'
            f'font-size:13px;" title="{tip}">{tok}</span>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{title}</title></head>
<body style="font-family:sans-serif;padding:2rem;max-width:900px;margin:auto;">
  <h2>{title}</h2>
  <div style="line-height:2.5;margin:1rem 0;">
    {''.join(spans)}
  </div>
  <div style="margin-top:1rem;font-size:12px;color:#888;">
    Scores range: [{s_min:.4f}, {s_max:.4f}]
  </div>
</body>
</html>"""

    if save_path:
        with open(save_path, 'w') as f:
            f.write(html)
        print(f"Heatmap saved to {save_path}")

    return html


# ===========================================================================
# 6. Reliability Diagram (Calibration Visualization)
# ===========================================================================
def plot_reliability_diagram(
    bin_data    : List[Dict],   # from expected_calibration_error
    ece         : float,
    save_path   : Optional[str] = None,
):
    """
    Plots the calibration reliability diagram.
    Perfect calibration: all bars on the diagonal.
    Gap between bar and diagonal = miscalibration.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not installed. Skipping reliability diagram.")
        return

    confs = [b['conf'] for b in bin_data if b['n'] > 0]
    accs  = [b['acc']  for b in bin_data if b['n'] > 0]

    fig, ax = plt.subplots(figsize=(6, 6))

    bar_width = 1.0 / (len(bin_data) + 1)
    for b in bin_data:
        if b['n'] == 0:
            continue
        c = b['conf']
        a = b['acc']
        color = '#D85A30' if a < c else '#1D9E75'
        ax.bar(c, a, width=bar_width * 0.9, color=color, alpha=0.7, align='center')
        ax.bar(c, c, width=bar_width * 0.9, color='none',
               edgecolor='#378ADD', linewidth=1.5, align='center', linestyle='--')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration')
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy',   fontsize=12)
    ax.set_title(f'Reliability Diagram  (ECE = {ece:.4f})', fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)

    # Legend for gap colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D85A30', alpha=0.7, label='Overconfident'),
        Patch(facecolor='#1D9E75', alpha=0.7, label='Underconfident'),
    ]
    ax.legend(handles=legend_elements + ax.get_lines(), loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Reliability diagram saved to {save_path}")
    plt.show()
