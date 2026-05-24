"""
Ablation Study Runner
======================
Systematically ablates each architectural modification to isolate its
contribution to overall performance. This is required for any research
publication claiming novel architectural improvements.

Ablation Configurations (8 modifications → 9 runs including full model):
  Config 0 — Baseline:  Sinusoidal PE + ReLU FFN + dense attn + CLS-only head
  Config 1 — +RoPE:     Add Rotary PE (M1)
  Config 2 — +SwiGLU:   Add SwiGLU FFN (M5)
  Config 3 — +PreLN:    Add Pre-Layer Normalization (M7)
  Config 4 — +Sparse:   Add HSR sparse attention in upper layers (M2+M3)
  Config 5 — +MoE:      Add Mixture-of-Experts in upper layers (M6)
  Config 6 — +CSCG:     Add Cross-Sentence Consistency Gate (M4)
  Config 7 — +MultiPool:Add multi-pool classification head (M8)
  Config 8 — FULL:      All modifications (our proposed model)

Each config is trained for a shorter schedule (ablation_epochs) for speed.
Full training uses config.epochs from TrainingConfig.
"""

import sys
import json
import copy
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer.embeddings import (HybridEmbedding, SinusoidalPositionalEncoding,
                                         TokenEmbedding, LearnablePositionalEncoding)
from src.transformer.attention  import MultiHeadAttention
from src.transformer.feedforward import PositionWiseFFN, GeLU_FFN, SwiGLU_FFN, MoE_FFN
from src.transformer.encoder    import FakeNewsTransformer
from src.training.trainer       import TrainingConfig, train, MetricLogger
from src.evaluation.metrics     import compute_all_metrics, AblationTracker, print_results_table
from src.training.trainer       import evaluate, LabelSmoothingCrossEntropy


# ===========================================================================
# Ablation Model Variants (override specific components)
# ===========================================================================

class BaselineTransformer(nn.Module):
    """
    Config 0 — Pure baseline:
      - Standard sinusoidal absolute PE (added in embedding)
      - ReLU feed-forward (no SwiGLU, no MoE)
      - Dense attention (no sparse, no RoPE)
      - Post-LN normalization
      - CLS-only classification head
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, dropout=0.1, max_len=512, n_classes=2, padding_idx=0):
        super().__init__()
        import math
        from src.transformer.embeddings import TokenEmbedding, SinusoidalPositionalEncoding

        self.token_emb = TokenEmbedding(vocab_size, d_model, padding_idx)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)

        # Vanilla Post-LN encoder layers
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=4*d_model, dropout=dropout,
            activation='relu', batch_first=True, norm_first=False  # Post-LN
        )
        self.encoder   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm      = nn.LayerNorm(d_model)

        # CLS-only head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            nn.Linear(d_model, n_classes),
        )
        self.padding_idx = padding_idx

    def forward(self, token_ids, segment_ids=None, mask=None):
        import math
        x = self.token_emb(token_ids)
        x = self.pos_enc(x)

        # Build src_key_padding_mask for PyTorch TransformerEncoder
        pad_mask = (token_ids == self.padding_idx)   # (B, T) True where pad

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)

        cls_out = x[:, 0, :]
        logits  = self.classifier(cls_out)

        return {
            'logits'      : logits,
            'cls_hidden'  : cls_out,
            'attn_weights': [],
            'moe_aux_loss': torch.tensor(0.0),
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        return {'total': total, 'trainable': total}


def build_ablation_model(config_name: str, vocab_size: int,
                          base_cfg: TrainingConfig) -> nn.Module:
    """
    Factory: returns the appropriate model variant for each ablation config.
    Configs 1-7 use FakeNewsTransformer with selective modifications disabled
    by patching the constructor arguments and/or replacing submodules.
    """
    d  = base_cfg.d_model
    h  = base_cfg.num_heads
    L  = base_cfg.num_layers
    dr = base_cfg.dropout
    ml = base_cfg.max_len
    nc = base_cfg.n_classes

    if config_name == 'baseline':
        return BaselineTransformer(vocab_size, d, h, L, dr, ml, nc)

    # For all other configs, we use FakeNewsTransformer but patch
    # specific submodules after construction.

    model = FakeNewsTransformer(
        vocab_size=vocab_size, d_model=d, num_heads=h, num_layers=L,
        dropout=dr, max_len=ml, n_classes=nc,
        n_experts=base_cfg.n_experts,
        classifier_dropout=base_cfg.classifier_dropout,
    )

    if config_name == 'no_rope':
        # Replace RoPE attention with standard attention (learnable PE instead)
        _replace_rope_with_learnable_pe(model, d, h, L, dr, ml)

    elif config_name == 'no_swiglu':
        # Replace SwiGLU FFNs with GeLU FFNs
        _replace_ffn(model, 'gelu', d, dr, L)

    elif config_name == 'no_preln':
        # Replace Pre-LN layers with Post-LN
        _replace_preln_with_postln(model, d, h, L, dr, ml)

    elif config_name == 'no_sparse':
        # Make all layers dense (disable sparse attention)
        _disable_sparse_attention(model)

    elif config_name == 'no_moe':
        # Replace MoE FFNs with single SwiGLU
        _disable_moe(model, d, dr, L)

    elif config_name == 'no_cscg':
        # Replace CSCG with identity
        model.encoder.cscg = nn.Identity()

    elif config_name == 'no_multipool':
        # Replace multi-pool head with CLS-only
        _replace_multipool_with_cls(model, d, nc, base_cfg.classifier_dropout)

    elif config_name == 'full':
        pass   # Already built with all modifications

    return model


# ---------------------------------------------------------------------------
# Patching helpers (used by build_ablation_model)
# ---------------------------------------------------------------------------

def _replace_rope_with_learnable_pe(model, d_model, num_heads, num_layers, dropout, max_len):
    """Remove RoPE from all attention layers, add learnable PE in embedding."""
    from src.transformer.embeddings import LearnablePositionalEncoding

    # Add learnable PE on top of embedding output
    lear_pe = LearnablePositionalEncoding(d_model, max_len, dropout)
    original_emb_forward = model.encoder.embedding.forward

    def patched_emb_forward(token_ids, segment_ids=None):
        x = original_emb_forward(token_ids, segment_ids)
        return lear_pe(x)

    model.encoder.embedding.forward = patched_emb_forward

    # Disable RoPE in all attention layers by making apply_rope a no-op
    for layer in model.encoder.layers:
        attn = layer.self_attn
        if hasattr(attn, 'rope'):
            attn.rope.apply_rope = lambda x: x
            attn.rope.forward    = lambda q, k: (q, k)


def _replace_ffn(model, ffn_type: str, d_model, dropout, num_layers):
    """Replace all FFN submodules with specified type."""
    d_ff = 4 * d_model
    for layer in model.encoder.layers:
        if ffn_type == 'relu':
            layer.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        elif ffn_type == 'gelu':
            layer.ffn = GeLU_FFN(d_model, d_ff, dropout)
        layer.use_moe = False


def _replace_preln_with_postln(model, d_model, num_heads, num_layers, dropout, max_len):
    """Swap Pre-LN order to Post-LN in all encoder layers."""
    # Monkey-patch forward method of each layer to use Post-LN order
    import types

    def postln_forward(self, x, mask=None):
        # Post-LN: SubLayer first, then residual + LN
        attn_out, attn_w = self.self_attn(x, mask)
        x = self.norm1(x + self.drop(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x, attn_w

    for layer in model.encoder.layers:
        layer.forward = types.MethodType(postln_forward, layer)


def _disable_sparse_attention(model):
    """Force all layers to use dense attention (sparse_k=None)."""
    for layer in model.encoder.layers:
        attn = layer.self_attn
        if hasattr(attn, 'sparse_k'):
            attn.sparse_k = None
        if hasattr(attn, 'sparse_k_ratio'):
            # Override forward to always pass sparse_k=None
            import types
            def dense_forward(self, x, mask=None):
                self.sparse_k = None
                return super(type(self), self).forward(x, mask)
            attn.forward = types.MethodType(dense_forward, attn)


def _disable_moe(model, d_model, dropout, num_layers):
    """Replace MoE FFNs with single SwiGLU."""
    for layer in model.encoder.layers:
        if layer.use_moe:
            layer.ffn     = SwiGLU_FFN(d_model, dropout=dropout)
            layer.use_moe = False


def _replace_multipool_with_cls(model, d_model, n_classes, cls_dropout):
    """Replace multi-pool (3*d_model) head with CLS-only head."""
    model.classifier = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
        nn.GELU(),
        nn.Dropout(cls_dropout),
        nn.Linear(d_model, n_classes),
    )

    # Patch model.forward to pass only cls_hidden
    import types
    original_forward = model.forward

    def patched_forward(self, token_ids, segment_ids=None, mask=None):
        enc_out  = self.encoder(token_ids, segment_ids, mask)
        logits   = self.classifier(enc_out['cls_hidden'])
        moe_loss = self.encoder.get_all_moe_aux_loss()
        return {
            'logits'      : logits,
            'cls_hidden'  : enc_out['cls_hidden'],
            'attn_weights': enc_out['attn_weights'],
            'moe_aux_loss': moe_loss,
        }

    model.forward = types.MethodType(patched_forward, model)


# ===========================================================================
# Main Ablation Runner
# ===========================================================================
ABLATION_CONFIGS = [
    ('baseline',      'Baseline (Sinusoidal PE + ReLU + PostLN + CLS-only)'),
    ('no_rope',       'Full − RoPE           (+Learnable PE instead)'),
    ('no_swiglu',     'Full − SwiGLU         (+GeLU FFN)'),
    ('no_preln',      'Full − Pre-LN         (+Post-LN)'),
    ('no_sparse',     'Full − HSR Sparse Attn(+Dense all layers)'),
    ('no_moe',        'Full − MoE FFN        (+Single SwiGLU upper)'),
    ('no_cscg',       'Full − CSCG Gate'),
    ('no_multipool',  'Full − Multi-Pool Head(+CLS-only)'),
    ('full',          'FULL MODEL (all M1–M8)'),
]


def run_ablation_study(
    train_loader,
    val_loader,
    test_loader,
    vocab_size    : int,
    base_config   : TrainingConfig,
    ablation_epochs: int = 5,
    device        : torch.device = torch.device('cpu'),
    output_dir    : str = './outputs/ablation',
) -> AblationTracker:
    """
    Runs the full ablation study. Each configuration is trained independently
    from scratch for `ablation_epochs` epochs, then evaluated on test set.

    Returns AblationTracker with results for all configurations.
    """
    from pathlib import Path
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tracker    = AblationTracker()
    criterion  = LabelSmoothingCrossEntropy(base_config.n_classes, base_config.label_smoothing)

    for config_name, config_desc in ABLATION_CONFIGS:
        print(f"\n{'='*70}")
        print(f"  ABLATION: {config_desc}")
        print(f"{'='*70}")

        # Short training config for ablation
        abl_cfg          = copy.deepcopy(base_config)
        abl_cfg.epochs   = ablation_epochs
        abl_cfg.output_dir = f"{output_dir}/{config_name}"
        abl_cfg.patience = ablation_epochs   # no early stopping in ablation

        # Build model
        model = build_ablation_model(config_name, vocab_size, abl_cfg)
        print(f"  Parameters: {model.count_parameters()['trainable']:,}")

        # Train
        logger = MetricLogger(use_wandb=False)
        trained_model = train(model, train_loader, val_loader, abl_cfg, device, logger)

        # Test evaluation
        test_raw     = evaluate(trained_model, test_loader, criterion, device)
        test_metrics = compute_all_metrics(
            test_raw['logits'], test_raw['labels'],
            prefix='test', n_classes=base_config.n_classes,
            n_bootstrap=500,
        )
        test_metrics['test_loss'] = test_raw['loss']
        test_metrics['n_params']  = model.count_parameters()['trainable']

        tracker.add(config_desc, test_metrics)
        print_results_table(test_metrics, title=f"Results: {config_desc}")

        # Save per-config results
        with open(f"{output_dir}/{config_name}_results.json", 'w') as f:
            json.dump({k: float(v) if hasattr(v, 'item') else v
                       for k, v in test_metrics.items()}, f, indent=2)

    # Print comparison table
    tracker.print_comparison(
        ['f1_macro', 'f1_fake', 'f1_real', 'roc_auc', 'mcc', 'ece'],
        prefix='test_'
    )

    # LaTeX table for paper
    latex = tracker.to_latex_table(
        ['f1_macro', 'f1_fake', 'f1_real', 'roc_auc', 'mcc', 'kappa', 'ece'],
        prefix='test_'
    )
    latex_path = f"{output_dir}/ablation_table.tex"
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX ablation table saved to {latex_path}")

    return tracker


if __name__ == '__main__':
    print("Import this module and call run_ablation_study() from your main script.")
    print("See scripts/run_ablation.py for the complete entry point.")
