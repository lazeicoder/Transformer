"""
Notebook 03 — Explainability & Error Analysis
===============================================
Run: python notebooks/03_explainability.py [path/to/WELFake_Dataset.csv]

Produces:
  outputs/plots/attention_rollout_sample_N.html   (token heatmaps)
  outputs/plots/head_entropy_heatmap.png
  outputs/plots/reliability_diagram.png
  outputs/plots/error_analysis.png
  outputs/plots/attention_head_specialization.png
"""

import sys, json
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.pipeline          import SimpleBPETokenizer, WELFakeDataset, FakeNewsCollator
from src.transformer.encoder    import FakeNewsTransformer
from src.training.trainer       import TrainingConfig, LabelSmoothingCrossEntropy, evaluate
from src.evaluation.metrics     import (compute_all_metrics, expected_calibration_error,
                                         confusion_matrix, print_results_table)
from src.explainability.attribution import (AttentionRollout, IntegratedGradients,
                                             AttentionHeadAnalyzer,
                                             render_token_heatmap, plot_reliability_diagram)

PLOTS_DIR = ROOT / 'outputs' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Load trained model
# =============================================================================
def load_model(device: torch.device) -> FakeNewsTransformer:
    ckpt_path = ROOT / 'outputs' / 'best_model.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {ckpt_path}. Run notebooks/02_training.py first."
        )
    ckpt   = torch.load(ckpt_path, map_location=device)
    cfg    = ckpt.get('config', {})

    # Re-load tokenizer
    tok_path = ROOT / 'outputs' / 'tokenizer.json'
    tokenizer = SimpleBPETokenizer.load(str(tok_path))

    model = FakeNewsTransformer(
        vocab_size         = tokenizer.vocab_size(),
        d_model            = cfg.get('d_model', 256),
        num_heads          = cfg.get('num_heads', 8),
        num_layers         = cfg.get('num_layers', 6),
        dropout            = cfg.get('dropout', 0.1),
        max_len            = cfg.get('max_len', 512),
        n_experts          = cfg.get('n_experts', 4),
        n_classes          = cfg.get('n_classes', 2),
        classifier_dropout = cfg.get('classifier_dropout', 0.3),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f"Model loaded from {ckpt_path}")
    return model, tokenizer


# =============================================================================
# Plot 1 — Attention Rollout Heatmaps
# =============================================================================
def attention_rollout_analysis(model, tokenizer, samples, device, n_samples=5):
    rollout_fn = AttentionRollout()
    print("\nGenerating attention rollout heatmaps...")

    for i, (text, true_label) in enumerate(samples[:n_samples]):
        token_ids_list = tokenizer.encode(text, max_length=128, add_special=True)
        token_ids  = torch.tensor([token_ids_list], dtype=torch.long).to(device)
        seg_ids    = torch.zeros_like(token_ids)

        with torch.no_grad():
            out = model(token_ids, seg_ids)

        # Collect attention weights from store
        attn_per_layer = out['attn_weights']   # list of (1, T, T)

        if not attn_per_layer:
            print("  No attention weights stored. Skipping.")
            continue

        attn_tensors = [a.detach().cpu() for a in attn_per_layer]
        attribution  = rollout_fn.compute(attn_tensors)[0].numpy()   # (T,)

        pred_label = out['logits'].argmax(dim=-1).item()
        pred_name  = {0: 'FAKE', 1: 'REAL'}[pred_label]
        true_name  = {0: 'FAKE', 1: 'REAL'}[true_label]

        # Decode tokens for display
        tokens = []
        for tid in token_ids_list:
            tok_str = tokenizer.inv_vocab.get(tid, f'[{tid}]')
            tok_str = tok_str.replace('</w>', '')
            tokens.append(tok_str if tok_str else '▪')

        title = f"Sample {i+1} | True: {true_name} | Pred: {pred_name}"
        html  = render_token_heatmap(
            tokens[:len(attribution)],
            attribution[:len(tokens)],
            title=title,
            save_path=str(PLOTS_DIR / f'attention_rollout_sample_{i+1}.html'),
        )
        print(f"  Sample {i+1}: True={true_name}, Pred={pred_name}")


# =============================================================================
# Plot 2 — Attention Head Entropy Heatmap
# =============================================================================
def plot_head_entropy(model, tokenizer, sample_text: str, device):
    print("\nComputing attention head entropy...")
    token_ids_list = tokenizer.encode(sample_text, max_length=256, add_special=True)
    token_ids  = torch.tensor([token_ids_list], dtype=torch.long).to(device)
    seg_ids    = torch.zeros_like(token_ids)

    from src.transformer.attention import GLOBAL_ATTN_STORE
    GLOBAL_ATTN_STORE.clear()

    with torch.no_grad():
        out = model(token_ids, seg_ids)

    n_layers  = len(model.encoder.layers)
    n_heads   = model.encoder.layers[0].self_attn.num_heads

    # Collect raw attention from store
    entropy_grid = np.zeros((n_layers, n_heads))

    for l_idx in range(n_layers):
        key  = f'layer_{l_idx}'
        attn = GLOBAL_ATTN_STORE.get(key)
        if attn is None:
            continue
        # attn: (B, H, T, T)
        if attn.dim() == 3:
            attn = attn.unsqueeze(1)

        # Per-head entropy
        p    = attn.clamp(min=1e-9)
        ent  = -(p * torch.log(p)).sum(dim=-1).mean(dim=-1)   # (B, H)
        entropy_grid[l_idx] = ent[0].numpy()

    fig, ax = plt.subplots(figsize=(max(6, n_heads), max(4, n_layers // 2 + 1)))
    im = ax.imshow(entropy_grid, cmap='YlOrRd', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Entropy (nats)')

    ax.set_xlabel('Attention Head', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title('Attention Head Entropy\n(High=Diffuse, Low=Focused)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f'H{i}' for i in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f'L{i}' for i in range(n_layers)])

    # Annotate with values
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(j, i, f'{entropy_grid[i,j]:.2f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if entropy_grid[i,j] > entropy_grid.max()*0.6 else '#2C2C2A')

    plt.tight_layout()
    path = PLOTS_DIR / 'head_entropy_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Head entropy heatmap saved: {path}")


# =============================================================================
# Plot 3 — Reliability Diagram
# =============================================================================
def plot_calibration(model, test_loader, device):
    print("\nComputing calibration metrics...")
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            token_ids   = batch['token_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            out   = model(token_ids, segment_ids)
            probs = torch.softmax(out['logits'], dim=-1)
            all_probs.append(probs.cpu())
            all_labels.append(batch['labels'])

    all_probs  = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    y_score    = all_probs[:, 1]

    ece_result = expected_calibration_error(all_labels, y_score, n_bins=15)
    print(f"  ECE = {ece_result['ece']:.4f}")
    print(f"  MCE = {ece_result['mce']:.4f}")

    plot_reliability_diagram(
        ece_result['bin_data'],
        ece_result['ece'],
        save_path=str(PLOTS_DIR / 'reliability_diagram.png'),
    )


# =============================================================================
# Plot 4 — Error Analysis
# =============================================================================
def error_analysis(model, tokenizer, df, device, n_errors=20):
    """Identify and characterize model errors."""
    print("\nRunning error analysis...")
    model.eval()
    errors = []

    sample = df.sample(min(500, len(df)), random_state=42)

    for _, row in sample.iterrows():
        text = str(row.get('title', '')) + ' ' + str(row.get('text', ''))
        text = text.strip()
        ids  = tokenizer.encode(text, max_length=512, add_special=True)
        token_ids = torch.tensor([ids], dtype=torch.long).to(device)
        seg_ids   = torch.zeros_like(token_ids)

        with torch.no_grad():
            out    = model(token_ids, seg_ids)
            probs  = torch.softmax(out['logits'], dim=-1)[0].cpu().numpy()
            pred   = out['logits'].argmax(dim=-1).item()
            true   = int(row['label'])

        if pred != true:
            confidence = probs[pred]
            errors.append({
                'text_preview': text[:120],
                'true_label'  : true,
                'pred_label'  : pred,
                'confidence'  : float(confidence),
                'text_length' : len(text.split()),
            })

    if not errors:
        print("  No errors found in sample (model may be very accurate or sample too small).")
        return

    error_confidences = [e['confidence'] for e in errors]
    error_lengths     = [e['text_length'] for e in errors]
    error_types       = [f"True={e['true_label']}, Pred={e['pred_label']}" for e in errors]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Confidence distribution of errors
    axes[0].hist(error_confidences, bins=20, color='#D85A30', alpha=0.8, edgecolor='white')
    axes[0].axvline(np.mean(error_confidences), color='black', linestyle='--',
                     label=f'Mean={np.mean(error_confidences):.3f}')
    axes[0].set_title('Error Confidence Distribution', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Model Confidence'); axes[0].set_ylabel('Count')
    axes[0].legend(); axes[0].spines[['top', 'right']].set_visible(False)

    # Text length of errors
    axes[1].hist(error_lengths, bins=20, color='#378ADD', alpha=0.8, edgecolor='white')
    axes[1].set_title('Text Length of Misclassified Samples', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Word Count'); axes[1].set_ylabel('Count')
    axes[1].spines[['top', 'right']].set_visible(False)

    # Error type counts
    from collections import Counter
    type_counts = Counter(error_types)
    keys        = list(type_counts.keys())
    vals        = [type_counts[k] for k in keys]
    colors      = ['#D85A30', '#1D9E75']
    axes[2].bar(keys, vals, color=colors[:len(keys)], edgecolor='white')
    axes[2].set_title('Error Type Breakdown', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Count')
    for spine in ['top', 'right']:
        axes[2].spines[spine].set_visible(False)

    fig.suptitle(f'Error Analysis ({len(errors)} errors in {len(sample)} samples)',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = PLOTS_DIR / 'error_analysis.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error analysis saved: {path}")

    # Print top confident errors
    top_errors = sorted(errors, key=lambda x: -x['confidence'])[:5]
    print("\n  Top-5 Most Confident Errors:")
    for e in top_errors:
        tname = {0:'Fake', 1:'Real'}[e['true_label']]
        pname = {0:'Fake', 1:'Real'}[e['pred_label']]
        print(f"    [{tname}→{pname}, conf={e['confidence']:.3f}]: {e['text_preview'][:80]}...")


# =============================================================================
# Main
# =============================================================================
def main(csv_path: str = None):
    print("\n" + "="*60)
    print("  PHASE 3 — EXPLAINABILITY & ERROR ANALYSIS")
    print("="*60 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    try:
        model, tokenizer = load_model(device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating a fresh untrained model for demonstration...")
        tokenizer = SimpleBPETokenizer()
        tokenizer.train(["fake news real story breaking"] * 200, vocab_size=2000)
        model = FakeNewsTransformer(vocab_size=tokenizer.vocab_size(),
                                     d_model=128, num_heads=4, num_layers=2)
        model.to(device).eval()

    # Sample texts for analysis
    sample_texts_and_labels = [
        ("SHOCKING: Scientists HIDE truth about vaccines! Government conspiracy exposed!!", 0),
        ("New study published in NEJM shows mRNA vaccines reduce hospitalization by 87%", 1),
        ("You won't BELIEVE what they found in the water supply — must share before deleted!", 0),
        ("Federal Reserve announces interest rate decision following inflation data review", 1),
        ("EXCLUSIVE: Deep state operatives sabotage election — whistleblower reveals all!", 0),
    ]

    # 1. Attention Rollout Heatmaps
    attention_rollout_analysis(model, tokenizer, sample_texts_and_labels, device)

    # 2. Head Entropy
    plot_head_entropy(model, tokenizer,
                      "Breaking news: Government announces major policy change on climate",
                      device)

    # 3. Calibration (needs test loader)
    try:
        from src.data.pipeline import load_welfake
        import pandas as pd
        _, _, test_loader, full_df = load_welfake(
            csv_path   = csv_path,
            tokenizer  = tokenizer,
            max_len    = 512,
            batch_size = 32,
            num_workers= 0,
        )
        plot_calibration(model, test_loader, device)
        error_analysis(model, tokenizer, full_df, device)
    except Exception as e:
        print(f"Skipping calibration/error analysis (data unavailable): {e}")

    print(f"\nAll explainability outputs saved to {PLOTS_DIR}")
    print("Explainability analysis complete.\n")


if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_file)
