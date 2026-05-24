"""
Notebook 01 — Exploratory Data Analysis
=========================================
Run this as a script: python notebooks/01_eda.py
Or open in Jupyter: jupyter notebook notebooks/01_eda.ipynb

Produces:
  outputs/eda/class_distribution.png
  outputs/eda/text_length_dist.png
  outputs/eda/title_length_dist.png
  outputs/eda/top_bigrams_fake.png
  outputs/eda/top_bigrams_real.png
  outputs/eda/token_length_after_bpe.png
  outputs/eda/ttr_comparison.png
  outputs/eda/eda_summary.json

All plots are publication-ready (300 DPI, tight layout, no seaborn dependency).
"""

import sys, os, json, math, collections
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
OUT_DIR  = ROOT / 'outputs' / 'eda'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from src.data.pipeline import EDAAnalyzer, SimpleBPETokenizer

# ─── Style constants ──────────────────────────────────────────────────────────
FAKE_COLOR = '#D85A30'
REAL_COLOR = '#1D9E75'
FONT       = {'family': 'DejaVu Sans', 'size': 11}
matplotlib.rc('font', **FONT)


# =============================================================================
# Load Data
# =============================================================================
def load_data(csv_path: str = None) -> pd.DataFrame:
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        df['title'] = df['title'].fillna('')
        df['text']  = df['text'].fillna('')
        df['label'] = df['label'].astype(int)
        print(f"Loaded {len(df):,} rows from {csv_path}")
        return df

    # ── Synthetic demo data if WELFake not downloaded ──
    print("WELFake CSV not found — generating synthetic demo dataset.")
    np.random.seed(42)
    n = 2000

    fake_headlines = [
        "SHOCKING: Government hides truth about {}", "You won't believe what {} did",
        "BREAKING: {} exposed as fraud", "Secret agenda behind {}",
        "They don't want you to know about {}", "Exclusive: {} under investigation",
    ]
    real_headlines = [
        "Study finds {} linked to improved outcomes", "New report on {}",
        "Researchers discover {} may affect health", "Officials announce {} policy",
        "Analysis: {} shows promising results", "Report: {} data released",
    ]
    subjects = ["vaccines", "climate", "elections", "economy", "AI", "health", "energy"]

    import random
    random.seed(42)

    rows = []
    for i in range(n):
        label = i % 2
        subj  = random.choice(subjects)
        if label == 0:  # Fake
            title = random.choice(fake_headlines).format(subj.upper())
            text  = (f"SHOCKING revelations about {subj}!! "
                     + "This is what they don't want you to know. " * 15
                     + f"Share this before it's deleted! #Truth #{subj}")
        else:           # Real
            title = random.choice(real_headlines).format(subj)
            text  = (f"A new study published in the Journal of {subj.capitalize()} "
                     + f"found that {subj} policies have significant effects. "
                     + "Researchers analyzed data from multiple cohorts. " * 10
                     + "The findings suggest further research is needed.")
        rows.append({'title': title, 'text': text, 'label': label})

    return pd.DataFrame(rows)


# =============================================================================
# Plot 1 — Class Distribution
# =============================================================================
def plot_class_distribution(analyzer: EDAAnalyzer):
    dist = analyzer.class_distribution()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart
    bars = axes[0].bar(dist['class'], dist['count'],
                        color=[FAKE_COLOR, REAL_COLOR], edgecolor='white', linewidth=1.5)
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Sample Count')
    for bar, pct in zip(bars, dist['pct']):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Pie chart
    axes[1].pie(dist['count'], labels=dist['class'],
                colors=[FAKE_COLOR, REAL_COLOR],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[1].set_title('Class Proportion', fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = OUT_DIR / 'class_distribution.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 2 — Text Length Distribution
# =============================================================================
def plot_text_lengths(analyzer: EDAAnalyzer):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title in zip(axes,
                               ['text_len', 'title_len'],
                               ['Article Body (words)', 'Headline (words)']):
        for label, name, color in [(0, 'Fake', FAKE_COLOR), (1, 'Real', REAL_COLOR)]:
            data = analyzer.df[analyzer.df['label'] == label][col]
            ax.hist(data, bins=50, alpha=0.6, color=color, label=name, density=True)
            ax.axvline(data.median(), color=color, linestyle='--', linewidth=1.5,
                       alpha=0.9, label=f'{name} median={data.median():.0f}')

        ax.set_title(f'Length Distribution: {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = OUT_DIR / 'text_length_dist.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 3 — Top Bigrams per Class
# =============================================================================
def plot_top_ngrams(analyzer: EDAAnalyzer):
    bigrams = analyzer.top_ngrams(n=2, top_k=15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (cls_name, color) in zip(axes, [('Fake', FAKE_COLOR), ('Real', REAL_COLOR)]):
        pairs = bigrams[cls_name]
        labels_list = [' '.join(p[0]) for p in pairs][::-1]
        counts = [p[1] for p in pairs][::-1]
        bars = ax.barh(labels_list, counts, color=color, alpha=0.8, edgecolor='white')
        ax.set_title(f'Top Bigrams — {cls_name} News', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency')
        ax.spines[['top', 'right']].set_visible(False)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(cnt), va='center', fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / 'top_bigrams.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 4 — Vocabulary Richness (TTR)
# =============================================================================
def plot_ttr(analyzer: EDAAnalyzer):
    richness = analyzer.vocabulary_richness()
    classes  = list(richness.keys())
    ttrs     = [richness[c]['TTR'] for c in classes]
    uniques  = [richness[c]['unique_tokens'] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = [FAKE_COLOR, REAL_COLOR]
    axes[0].bar(classes, ttrs, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Type-Token Ratio (TTR)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('TTR')
    axes[0].set_ylim(0, max(ttrs) * 1.2)
    for i, (cls, ttr) in enumerate(zip(classes, ttrs)):
        axes[0].text(i, ttr + 0.002, f'{ttr:.4f}', ha='center', fontsize=10, fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    axes[1].bar(classes, uniques, color=colors, edgecolor='white', linewidth=1.5)
    axes[1].set_title('Unique Token Count', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    for i, (cls, u) in enumerate(zip(classes, uniques)):
        axes[1].text(i, u + 100, f'{u:,}', ha='center', fontsize=10, fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = OUT_DIR / 'ttr_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Plot 5 — BPE Token Length Distribution
# =============================================================================
def plot_bpe_lengths(df: pd.DataFrame, tokenizer: SimpleBPETokenizer, sample_n: int = 500):
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    lengths_fake, lengths_real = [], []

    for _, row in sample.iterrows():
        text = str(row.get('title', '')) + ' ' + str(row.get('text', ''))
        ids  = tokenizer.encode(text, max_length=1024, add_special=True)
        if row['label'] == 0:
            lengths_fake.append(len(ids))
        else:
            lengths_real.append(len(ids))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(lengths_fake, bins=40, alpha=0.6, color=FAKE_COLOR, label='Fake', density=True)
    ax.hist(lengths_real, bins=40, alpha=0.6, color=REAL_COLOR, label='Real', density=True)
    ax.axvline(512, color='#2C2C2A', linestyle='--', linewidth=2, label='Max len = 512')
    ax.set_title('BPE Token Length Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Token Count (after BPE)')
    ax.set_ylabel('Density')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = OUT_DIR / 'token_length_after_bpe.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# =============================================================================
# Main
# =============================================================================
def main(csv_path: str = None):
    print("\n" + "="*60)
    print("  PHASE 1 — EXPLORATORY DATA ANALYSIS")
    print("="*60 + "\n")

    # Load
    df = load_data(csv_path)

    # Analyzer
    analyzer = EDAAnalyzer(df)
    analyzer.print_summary()

    # Tokenizer (train on small sample for BPE length analysis)
    print("\nTraining BPE tokenizer for token-length analysis...")
    sample_texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).tolist()
    tok = SimpleBPETokenizer()
    tok.train(sample_texts[:min(5000, len(sample_texts))], vocab_size=8000)

    # Plots
    print("\nGenerating EDA plots...")
    plot_class_distribution(analyzer)
    plot_text_lengths(analyzer)
    plot_top_ngrams(analyzer)
    plot_ttr(analyzer)
    plot_bpe_lengths(df, tok, sample_n=500)

    # Summary JSON
    summary = {
        'n_total'       : len(df),
        'n_fake'        : int((df['label'] == 0).sum()),
        'n_real'        : int((df['label'] == 1).sum()),
        'vocab_size_bpe': tok.vocab_size(),
        'text_len_stats': {str(k): v for k, v in analyzer.text_length_stats().to_dict().items()},
        'ttr'           : analyzer.vocabulary_richness(),
    }
    with open(OUT_DIR / 'eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nEDA summary saved to {OUT_DIR / 'eda_summary.json'}")

    # Save tokenizer for reuse in training
    tok_path = str(ROOT / 'outputs' / 'tokenizer.json')
    Path(tok_path).parent.mkdir(parents=True, exist_ok=True)
    tok.save(tok_path)

    print(f"\nAll EDA outputs saved to {OUT_DIR}")
    print("EDA complete.\n")
    return df, tok


if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_file)
