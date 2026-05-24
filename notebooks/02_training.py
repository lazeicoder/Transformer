"""
Notebook 02 — Model Training
==============================
Run: python notebooks/02_training.py [path/to/WELFake_Dataset.csv]

Covers:
  1. Load tokenizer (from EDA step) or retrain
  2. Build DataLoaders
  3. Initialize FakeNewsTransformer (all 8 modifications)
  4. Train with cosine LR + warmup + label smoothing + MoE aux loss
  5. Plot training curves
  6. Save model and training history

Outputs:
  outputs/best_model.pt
  outputs/training_history.json
  outputs/plots/training_curves.png
"""

import sys, json, math
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.pipeline       import SimpleBPETokenizer, load_welfake, EDAAnalyzer
from src.transformer.encoder import FakeNewsTransformer
from src.training.trainer    import TrainingConfig, train, MetricLogger
from src.evaluation.metrics  import compute_all_metrics, print_results_table
from src.training.trainer    import evaluate, LabelSmoothingCrossEntropy

PLOTS_DIR = ROOT / 'outputs' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FAKE_COLOR = '#D85A30'
REAL_COLOR = '#1D9E75'
BLUE_COLOR = '#378ADD'


# =============================================================================
# Plot Training Curves
# =============================================================================
def plot_training_curves(history_path: str):
    with open(history_path) as f:
        history = json.load(f)

    epochs_data = [h for h in history if 'val_f1_macro' in h]
    if not epochs_data:
        print("No epoch-level data found in history.")
        return

    epochs    = list(range(1, len(epochs_data) + 1))
    train_acc = [h.get('train_accuracy', 0) for h in epochs_data]
    val_acc   = [h.get('val_accuracy',   0) for h in epochs_data]
    train_loss= [h.get('train_loss',     0) for h in epochs_data]
    val_loss  = [h.get('val_loss',       0) for h in epochs_data]
    val_f1    = [h.get('val_f1_macro',   0) for h in epochs_data]
    val_auc   = [h.get('val_roc_auc',    0) for h in epochs_data]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color=FAKE_COLOR, marker='o', markersize=4, label='Train Loss')
    ax.plot(epochs, val_loss,   color=BLUE_COLOR,  marker='s', markersize=4, label='Val Loss')
    ax.set_title('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, train_acc, color=FAKE_COLOR, marker='o', markersize=4, label='Train Acc')
    ax.plot(epochs, val_acc,   color=BLUE_COLOR,  marker='s', markersize=4, label='Val Acc')
    ax.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Macro-F1
    ax = axes[1, 0]
    ax.plot(epochs, val_f1, color=REAL_COLOR, marker='^', markersize=5, label='Val Macro-F1')
    best_f1_ep = int(np.argmax(val_f1)) + 1
    ax.axvline(best_f1_ep, color='gray', linestyle='--', alpha=0.7, label=f'Best epoch={best_f1_ep}')
    ax.set_title('Validation Macro-F1', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Macro-F1')
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # ROC-AUC
    ax = axes[1, 1]
    ax.plot(epochs, val_auc, color='#8E44AD', marker='D', markersize=4, label='Val ROC-AUC')
    ax.set_title('Validation ROC-AUC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ROC-AUC')
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Training Dynamics — FakeNews Transformer', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = PLOTS_DIR / 'training_curves.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {path}")


# =============================================================================
# Plot Confusion Matrix
# =============================================================================
def plot_confusion_matrix(cm: np.ndarray, title: str = 'Confusion Matrix'):
    """Custom confusion matrix plot from scratch — no sklearn."""
    fig, ax = plt.subplots(figsize=(5, 5))
    classes = ['Fake', 'Real']
    n = cm.shape[0]

    # Color scale based on value
    vmax = cm.max()
    for i in range(n):
        for j in range(n):
            val   = cm[i, j]
            norm  = val / vmax if vmax > 0 else 0
            r = int(255 - (255 - 53)  * norm)
            g = int(255 - (255 - 122) * norm)
            b = int(255 - (255 - 221) * norm)
            color = (r/255, g/255, b/255)
            ax.add_patch(plt.Rectangle((j, n-1-i), 1, 1, color=color))
            text_c = 'white' if norm > 0.6 else '#2C2C2A'
            ax.text(j + 0.5, n - 1 - i + 0.5, str(val),
                    ha='center', va='center', fontsize=16,
                    fontweight='bold', color=text_c)

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels([f'Pred: {c}' for c in classes], fontsize=11)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels([f'True: {c}' for c in reversed(classes)], fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    # Add border
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    path = PLOTS_DIR / 'confusion_matrix.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {path}")


# =============================================================================
# Plot ROC and PR Curves from scratch
# =============================================================================
def plot_roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray,
                        roc_auc: float, pr_auc: float):
    from src.evaluation.metrics import (roc_curve_from_scratch, pr_curve_from_scratch)

    fpr, tpr, _  = roc_curve_from_scratch(y_true, y_score)
    prec, rec    = pr_curve_from_scratch(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE_COLOR, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE_COLOR)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.spines[['top', 'right']].set_visible(False)

    # PR Curve
    ax = axes[1]
    order = np.argsort(rec)
    ax.plot(rec[order], prec[order], color=REAL_COLOR, lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.fill_between(rec[order], prec[order], alpha=0.08, color=REAL_COLOR)
    baseline = (y_true == 1).mean()
    ax.axhline(baseline, color='k', linestyle='--', lw=1.5, label=f'Baseline={baseline:.3f}')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = PLOTS_DIR / 'roc_pr_curves.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR curves saved: {path}")


# =============================================================================
# Main Training Script
# =============================================================================
def main(csv_path: str = None):
    print("\n" + "="*60)
    print("  PHASE 2 — MODEL TRAINING")
    print("="*60 + "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load or train tokenizer ──
    tok_path = ROOT / 'outputs' / 'tokenizer.json'
    if tok_path.exists():
        tokenizer = SimpleBPETokenizer.load(str(tok_path))
    else:
        print("Tokenizer not found. Run notebooks/01_eda.py first.")
        # Quick fallback: train on dummy data
        tokenizer = SimpleBPETokenizer()
        tokenizer.train(["example text for tokenizer training"] * 100, vocab_size=3000)

    # ── Training Config ──
    config = TrainingConfig(
        d_model            = 256,
        num_heads          = 8,
        num_layers         = 6,
        dropout            = 0.1,
        max_len            = 512,
        n_experts          = 4,
        n_classes          = 2,
        epochs             = 20,
        batch_size         = 16,
        lr                 = 3e-4,
        lr_min             = 1e-6,
        weight_decay       = 0.01,
        warmup_ratio       = 0.06,
        grad_clip          = 1.0,
        label_smoothing    = 0.1,
        moe_loss_weight    = 0.01,
        classifier_dropout = 0.3,
        output_dir         = str(ROOT / 'outputs'),
        seed               = 42,
        patience           = 5,
        fp16               = (device.type == 'cuda'),
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── DataLoaders ──
    train_loader, val_loader, test_loader, full_df = load_welfake(
        csv_path   = csv_path,
        tokenizer  = tokenizer,
        max_len    = config.max_len,
        batch_size = config.batch_size,
        num_workers= 0,   # set to 2+ on Linux
        seed       = config.seed,
    )

    # ── Model ──
    model = FakeNewsTransformer(
        vocab_size         = tokenizer.vocab_size(),
        d_model            = config.d_model,
        num_heads          = config.num_heads,
        num_layers         = config.num_layers,
        dropout            = config.dropout,
        max_len            = config.max_len,
        n_experts          = config.n_experts,
        n_classes          = config.n_classes,
        classifier_dropout = config.classifier_dropout,
    )

    params = model.count_parameters()
    print(f"\nModel Architecture:")
    print(f"  Total parameters   : {params['total']:,}")
    print(f"  Trainable params   : {params['trainable']:,}")
    print(f"  d_model={config.d_model}, heads={config.num_heads}, "
          f"layers={config.num_layers}, experts={config.n_experts}")
    print(f"  Modifications: RoPE + SwiGLU + Pre-LN + HSR + MoE + CSCG + MultiPool\n")

    # ── Train ──
    logger = MetricLogger(use_wandb=False)
    trained_model = train(model, train_loader, val_loader, config, device, logger)

    # ── Plot training curves ──
    history_path = str(ROOT / 'outputs' / 'training_history.json')
    if Path(history_path).exists():
        plot_training_curves(history_path)

    # ── Test Evaluation ──
    print("\n" + "="*60)
    print("  FINAL TEST SET EVALUATION")
    print("="*60)
    criterion  = LabelSmoothingCrossEntropy(config.n_classes, config.label_smoothing)
    test_raw   = evaluate(trained_model, test_loader, criterion, device)
    test_metrics = compute_all_metrics(
        test_raw['logits'], test_raw['labels'],
        prefix='test', n_classes=config.n_classes,
        n_bootstrap=1000,
    )
    test_metrics['test_loss'] = test_raw['loss']
    print_results_table(test_metrics, title="TEST SET RESULTS — FakeNews Transformer")

    # ── Additional plots ──
    y_true  = test_raw['labels'].numpy()
    probs   = torch.softmax(test_raw['logits'], dim=-1).numpy()
    y_score = probs[:, 1]
    y_pred  = test_raw['logits'].argmax(dim=-1).numpy()

    from src.evaluation.metrics import confusion_matrix as cm_fn
    cm = cm_fn(y_true, y_pred, n_classes=2)
    plot_confusion_matrix(cm, title='Confusion Matrix — Test Set')
    plot_roc_pr_curves(
        y_true, y_score,
        roc_auc=test_metrics.get('test_roc_auc', 0),
        pr_auc =test_metrics.get('test_pr_auc', 0),
    )

    # ── Save test results ──
    results_path = ROOT / 'outputs' / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump({k: float(v) if hasattr(v, 'item') else str(v)
                   for k, v in test_metrics.items()}, f, indent=2)
    print(f"\nTest results saved to {results_path}")

    return trained_model, test_metrics


if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_file)
