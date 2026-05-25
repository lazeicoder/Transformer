"""
FakeNews Transformer — Research-Paper Architectural Diagram
Saves to: outputs/architecture_diagram.png  (300 dpi)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("outputs", exist_ok=True)

# ── Canvas ──────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 22, 30
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("#FAFAFA")

# ── Palette ─────────────────────────────────────────────────────────────────
C = {
    "input":    "#E8F4FD",
    "embed":    "#D6EAF8",
    "dense":    "#D5F5E3",
    "sparse":   "#FCF3CF",
    "cscg":     "#FADBD8",
    "pool":     "#E8DAEF",
    "cls":      "#F9EBEA",
    "out":      "#FDEBD0",
    "border":   "#2C3E50",
    "arrow":    "#2C3E50",
    "mod":      "#E74C3C",
    "novel":    "#8E44AD",
    "title_bg": "#2C3E50",
    "attn_bg":  "#EBF5FB",
    "ffn_bg":   "#EAFAF1",
    "sub_border":"#7F8C8D",
}

# ── Helper functions ─────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, color, fontsize=10, bold=False,
        border=C["border"], radius=0.3, label2=None, label2_color="#555555",
        label2_size=8):
    """Draw a rounded rectangle with centred label."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        linewidth=1.4, edgecolor=border, facecolor=color, zorder=3
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    dy = 0.12 if label2 else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#1A1A1A", zorder=4)
    if label2:
        ax.text(x, y - 0.22, label2, ha="center", va="center",
                fontsize=label2_size, color=label2_color, zorder=4,
                style="italic")


def arrow(ax, x1, y1, x2, y2, color=C["arrow"], lw=1.6, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=5)


def badge(ax, x, y, text, color=C["mod"], size=7.5):
    """Small coloured pill badge for modification labels."""
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            color="white", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color,
                      edgecolor="none"))


def section_bg(ax, x, y, w, h, color, label="", label_size=8):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        linewidth=1.0, edgecolor=C["sub_border"],
        facecolor=color, zorder=1, alpha=0.55
    )
    ax.add_patch(rect)
    if label:
        ax.text(x + 0.18, y + h - 0.22, label, ha="left", va="top",
                fontsize=label_size, color="#555", style="italic", zorder=2)

# ════════════════════════════════════════════════════════════════════════════
# TITLE
# ════════════════════════════════════════════════════════════════════════════
title_rect = FancyBboxPatch((0.3, 28.8), 21.4, 1.0,
    boxstyle="round,pad=0.1", linewidth=0,
    facecolor=C["title_bg"], zorder=3)
ax.add_patch(title_rect)
ax.text(11, 29.35, "FakeNews Transformer — Full Architecture",
        ha="center", va="center", fontsize=16, fontweight="bold",
        color="white", zorder=4)
ax.text(11, 29.0, "Hierarchical Sparse Routing Transformer with Cross-Sentence Consistency Gate  |  M1–M8",
        ha="center", va="center", fontsize=9, color="#BDC3C7", zorder=4)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INPUT
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 26.8, 21.0, 1.7, C["input"], "① Input")

# Three input streams
for xi, lbl, sub in [
    (4.5,  "Headline Tokens",  "segment_id = 0"),
    (11.0, "Body Tokens",      "segment_id = 1"),
    (17.5, "Metadata Tokens",  "segment_id = 2  (source / date)"),
]:
    box(ax, xi, 27.35, 4.2, 0.7, lbl, C["input"], fontsize=9,
        label2=sub, label2_size=7.5)

# Merge arrow
arrow(ax, 4.5,  26.98, 11.0, 26.55)
arrow(ax, 11.0, 26.98, 11.0, 26.55)
arrow(ax, 17.5, 26.98, 11.0, 26.55)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EMBEDDING
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 24.5, 21.0, 1.9, C["embed"], "② Embedding Layer")

box(ax, 5.5, 25.45, 3.8, 0.75, "Token Embedding",
    C["embed"], fontsize=9, label2="vocab_size × d_model", label2_size=7.5)
box(ax, 11.0, 25.45, 3.8, 0.75, "Segment Embedding",
    C["embed"], fontsize=9, label2="3 segments × d_model", label2_size=7.5)
box(ax, 16.5, 25.45, 3.8, 0.75, "LayerNorm + Dropout",
    C["embed"], fontsize=9, label2="ε = 1e-6", label2_size=7.5)

# ⊕ add symbol
ax.text(8.55, 25.45, "⊕", ha="center", va="center", fontsize=14,
        color=C["border"], zorder=5)
arrow(ax, 5.5+1.9, 25.45, 8.3, 25.45, style="-")
arrow(ax, 11.0-1.9, 25.45, 8.8, 25.45, style="-")
arrow(ax, 8.55, 25.45, 14.6, 25.45)

# Note: no absolute PE — RoPE applied inside attention
ax.text(11.0, 24.72, "★  No absolute positional encoding — RoPE applied inside each attention head (M1)",
        ha="center", va="center", fontsize=7.8, color="#555", style="italic", zorder=4)

arrow(ax, 11.0, 26.55, 11.0, 26.38)   # input → embed section
arrow(ax, 16.5+1.9, 25.45, 19.5, 25.45, style="-")
arrow(ax, 19.5, 25.45, 19.5, 24.5)
arrow(ax, 19.5, 24.5, 11.0, 24.5)
arrow(ax, 11.0, 24.5, 11.0, 24.3)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DENSE ENCODER LAYERS (lower N/2 layers)
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 19.5, 21.0, 4.6, C["dense"], "③ Dense Encoder Layers  [Layers 0 … N/2−1]")

ax.text(11.0, 23.85, "Repeated × (N/2) layers  —  local syntactic feature extraction",
        ha="center", va="center", fontsize=8.5, color="#1A6B3C", style="italic", zorder=4)

# ── Sub-layer A: Pre-LN + Dense MHA ─────────────────────────────────────
section_bg(ax, 1.0, 21.55, 20.0, 2.1, C["attn_bg"], "Self-Attention sub-layer")

box(ax, 4.0, 22.6, 3.5, 0.7, "Pre-LayerNorm", C["dense"],
    fontsize=9, label2="M7 — applied before sub-layer", label2_size=7.5)
badge(ax, 5.55, 23.05, "M7")

box(ax, 9.5, 22.6, 4.2, 0.7, "Multi-Head Attention\n(Dense)", C["dense"],
    fontsize=9, bold=True)
badge(ax, 11.5, 23.05, "M1 RoPE", color="#1A5276")

# RoPE detail inside MHA
ax.text(9.5, 22.15, "Q,K rotated by RoPE  |  V unchanged  |  8 heads  |  head_dim = d/H",
        ha="center", va="center", fontsize=7, color="#1A5276", zorder=4)

box(ax, 15.5, 22.6, 3.2, 0.7, "Dropout + Residual", C["dense"], fontsize=9)

arrow(ax, 4.0+1.75, 22.6, 9.5-2.1, 22.6)
arrow(ax, 9.5+2.1, 22.6, 15.5-1.6, 22.6)
arrow(ax, 15.5+1.6, 22.6, 18.5, 22.6, style="-")
arrow(ax, 18.5, 22.6, 18.5, 21.55)

# ── Sub-layer B: Pre-LN + SwiGLU FFN ────────────────────────────────────
section_bg(ax, 1.0, 19.7, 20.0, 1.7, C["ffn_bg"], "FFN sub-layer")

box(ax, 4.0, 20.55, 3.5, 0.7, "Pre-LayerNorm", C["dense"],
    fontsize=9, label2="M7", label2_size=7.5)
badge(ax, 5.55, 21.0, "M7")

box(ax, 9.5, 20.55, 4.2, 0.7, "SwiGLU FFN", C["dense"],
    fontsize=9, bold=True,
    label2="(xW₁ ⊙ SiLU(xW₃))W₂   |   d_ff = ⌈2/3 · 4d⌉", label2_size=7.5)
badge(ax, 11.5, 21.0, "M5 SwiGLU", color="#117A65")

box(ax, 15.5, 20.55, 3.2, 0.7, "Dropout + Residual", C["dense"], fontsize=9)

arrow(ax, 4.0+1.75, 20.55, 9.5-2.1, 20.55)
arrow(ax, 9.5+2.1, 20.55, 15.5-1.6, 20.55)

# Residual skip line for full layer
ax.annotate("", xy=(1.5, 20.55), xytext=(1.5, 22.6),
            arrowprops=dict(arrowstyle="-", color="#888", lw=1.2,
                            connectionstyle="arc3,rad=0.0"), zorder=2)
ax.text(1.2, 21.6, "skip", ha="center", va="center", fontsize=7,
        color="#888", rotation=90, zorder=4)

arrow(ax, 11.0, 24.3, 11.0, 24.1)
arrow(ax, 11.0, 24.1, 4.0-1.75, 22.6+0.35, style="-")
arrow(ax, 4.0-1.75, 22.6+0.35, 4.0-1.75, 22.6)
arrow(ax, 4.0-1.75, 22.6, 4.0-1.75, 22.6, style="-")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SPARSE ENCODER LAYERS (upper N/2 layers)
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 14.5, 21.0, 4.7, C["sparse"], "④ Sparse Encoder Layers  [Layers N/2 … N−1]  — HSR (M3)")

ax.text(11.0, 18.95, "Repeated × (N/2) layers  —  global semantic claim detection  |  M2 + M3 + M5 + M6 + M7",
        ha="center", va="center", fontsize=8.5, color="#7D6608", style="italic", zorder=4)

# Transition arrow
arrow(ax, 11.0, 19.5, 11.0, 19.3)
ax.text(13.2, 19.4, "HSR boundary: dense → sparse", ha="left", va="center",
        fontsize=7.5, color=C["mod"], style="italic", zorder=4)

# ── Sub-layer A: Pre-LN + Sparse MHA ────────────────────────────────────
section_bg(ax, 1.0, 16.55, 20.0, 2.1, C["attn_bg"], "Sparse Self-Attention sub-layer")

box(ax, 4.0, 17.6, 3.5, 0.7, "Pre-LayerNorm", C["sparse"],
    fontsize=9, label2="M7", label2_size=7.5)
badge(ax, 5.55, 18.05, "M7")

box(ax, 9.5, 17.6, 4.2, 0.7, "Sparse MHA\n(Top-K)", C["sparse"],
    fontsize=9, bold=True)
badge(ax, 8.2, 18.05, "M2", color="#884EA0")
badge(ax, 11.5, 18.05, "M1 RoPE", color="#1A5276")

ax.text(9.5, 17.15, "k = max(8, T×0.125)  |  attend to top 12.5% keys  |  RoPE on Q,K",
        ha="center", va="center", fontsize=7, color="#6C3483", zorder=4)

box(ax, 15.5, 17.6, 3.2, 0.7, "Dropout + Residual", C["sparse"], fontsize=9)

arrow(ax, 4.0+1.75, 17.6, 9.5-2.1, 17.6)
arrow(ax, 9.5+2.1, 17.6, 15.5-1.6, 17.6)

# ── Sub-layer B: Pre-LN + MoE-SwiGLU FFN ───────────────────────────────
section_bg(ax, 1.0, 14.7, 20.0, 1.7, C["ffn_bg"], "MoE-SwiGLU FFN sub-layer")

box(ax, 4.0, 15.55, 3.5, 0.7, "Pre-LayerNorm", C["sparse"],
    fontsize=9, label2="M7", label2_size=7.5)
badge(ax, 5.55, 16.0, "M7")

box(ax, 9.5, 15.55, 4.8, 0.7, "MoE-SwiGLU FFN", C["sparse"],
    fontsize=9, bold=True,
    label2="Router → softmax weights → Σ expert_e(x)  |  n_experts=4", label2_size=7.5)
badge(ax, 8.0, 16.0, "M6 MoE", color="#BA4A00")
badge(ax, 11.8, 16.0, "M5 SwiGLU", color="#117A65")

box(ax, 16.0, 15.55, 3.2, 0.7, "Dropout + Residual\n+ MoE Aux Loss", C["sparse"], fontsize=8.5)

arrow(ax, 4.0+1.75, 15.55, 9.5-2.4, 15.55)
arrow(ax, 9.5+2.4, 15.55, 16.0-1.6, 15.55)

# Residual skip for sparse layer
ax.annotate("", xy=(1.5, 15.55), xytext=(1.5, 17.6),
            arrowprops=dict(arrowstyle="-", color="#888", lw=1.2), zorder=2)
ax.text(1.2, 16.6, "skip", ha="center", va="center", fontsize=7,
        color="#888", rotation=90, zorder=4)

# Connect dense output → sparse input
arrow(ax, 11.0, 19.3, 4.0-1.75, 17.6+0.35, style="-")
arrow(ax, 4.0-1.75, 17.6+0.35, 4.0-1.75, 17.6)

# Final norm after last layer
arrow(ax, 11.0, 14.5, 11.0, 14.2)
box(ax, 11.0, 13.85, 5.0, 0.65, "Final LayerNorm", C["sparse"],
    fontsize=9, label2="Post-stack normalisation", label2_size=7.5)
arrow(ax, 11.0, 13.52, 11.0, 13.3)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CSCG  (M4)
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 11.5, 21.0, 1.6, C["cscg"], "⑤ Cross-Sentence Consistency Gate  (M4 — Novel)")

box(ax, 11.0, 12.3, 10.0, 0.85, "Cross-Sentence Consistency Gate  (CSCG)", C["cscg"],
    fontsize=10, bold=True,
    label2="gate_i = σ(W_g·[CLS ; xᵢ])   →   x_i' = gate_i⊙xᵢ + (1−gate_i)⊙xᵢ.detach()   →   LayerNorm",
    label2_size=7.8)
badge(ax, 6.5, 12.55, "M4  NOVEL", color=C["novel"])

ax.text(11.0, 11.68, "Suppresses tokens inconsistent with the global [CLS] claim  |  Ablation: +0.8–1.2% F1",
        ha="center", va="center", fontsize=7.8, color="#6C3483", style="italic", zorder=4)

arrow(ax, 11.0, 13.3, 11.0, 13.1)
arrow(ax, 11.0, 11.5, 11.0, 11.3)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MULTI-POOL HEAD  (M8)
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 8.8, 21.0, 2.3, C["pool"], "⑥ Multi-Pool Classification Head  (M8 — Novel)")

# Three pool streams
for xi, lbl, sub in [
    (4.5,  "[CLS] Token",  "global claim repr."),
    (11.0, "Mean Pool",    "avg over non-pad tokens"),
    (17.5, "Max Pool",     "strongest feature / dim"),
]:
    box(ax, xi, 10.35, 3.8, 0.7, lbl, C["pool"],
        fontsize=9, label2=sub, label2_size=7.5)
    badge(ax, xi+1.7, 10.75, "M8", color=C["novel"])

# Concat
ax.text(11.0, 9.65, "concat  [ CLS ; mean ; max ]  →  3 × d_model", ha="center",
        va="center", fontsize=8.5, color="#4A235A", fontweight="bold", zorder=4)

arrow(ax, 4.5,  10.0, 11.0, 9.85, style="-")
arrow(ax, 11.0, 10.0, 11.0, 9.85)
arrow(ax, 17.5, 10.0, 11.0, 9.85, style="-")

# Classifier MLP
box(ax, 11.0, 9.1, 9.0, 0.65,
    "Linear(3d→d)  →  LayerNorm  →  GELU  →  Dropout(0.3)  →  Linear(d→2)",
    C["pool"], fontsize=8.5)
arrow(ax, 11.0, 9.45, 11.0, 9.43)
arrow(ax, 11.0, 8.8, 11.0, 8.6)

# Distribute from encoder output to three pools
arrow(ax, 11.0, 11.3, 4.5,  10.7)
arrow(ax, 11.0, 11.3, 11.0, 10.7)
arrow(ax, 11.0, 11.3, 17.5, 10.7)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OUTPUT
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 7.0, 21.0, 1.4, C["out"], "⑦ Output")

box(ax, 7.5, 7.7, 5.5, 0.75, "Logits  (B × 2)", C["out"],
    fontsize=10, bold=True, label2="raw scores — REAL / FAKE", label2_size=8)
box(ax, 15.5, 7.7, 5.0, 0.75, "MoE Aux Loss  (scalar)", C["out"],
    fontsize=9, label2="load-balancing term added to CE loss", label2_size=7.5)

arrow(ax, 8.6, 8.6, 7.5, 8.08)
arrow(ax, 8.6, 8.6, 15.5, 8.08)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAINING OBJECTIVE (compact)
# ════════════════════════════════════════════════════════════════════════════
section_bg(ax, 0.5, 5.3, 21.0, 1.5, "#EBF5FB", "⑧ Training Objective")
ax.text(11.0, 6.1,
        "L = LabelSmoothing-CE(logits, y, ε=0.1)  +  λ · L_MoE",
        ha="center", va="center", fontsize=10, color="#1A1A1A",
        fontweight="bold", zorder=4)
ax.text(11.0, 5.65,
        "Optimizer: AdamW  |  Schedule: CosineAnnealing + Linear Warmup  |  Gradient Clip: 1.0",
        ha="center", va="center", fontsize=8.5, color="#555", zorder=4)

# ════════════════════════════════════════════════════════════════════════════
# LEGEND
# ════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C["dense"],  "Dense layers (lower N/2)"),
    (C["sparse"], "Sparse layers (upper N/2)  — HSR"),
    (C["cscg"],   "CSCG gate  (M4, Novel)"),
    (C["pool"],   "Multi-pool head  (M8, Novel)"),
    (C["mod"],    "Modification badge  (M1–M8)"),
    (C["novel"],  "Novel contribution"),
]
lx, ly = 1.0, 4.6
ax.text(lx, ly, "Legend", fontsize=9, fontweight="bold", color=C["border"], zorder=4)
for i, (color, label) in enumerate(legend_items):
    col = lx + (i % 3) * 7.0
    row = ly - 0.5 - (i // 3) * 0.5
    rect = FancyBboxPatch((col, row - 0.15), 0.45, 0.3,
                          boxstyle="round,pad=0.03", linewidth=0.8,
                          edgecolor=C["border"], facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(col + 0.6, row, label, va="center", fontsize=8,
            color="#1A1A1A", zorder=4)

# ════════════════════════════════════════════════════════════════════════════
# MODIFICATION SUMMARY TABLE (bottom)
# ════════════════════════════════════════════════════════════════════════════
mods = [
    ("M1", "RoPE",    "Rotary Positional Encoding on Q,K",          "Su et al., 2021"),
    ("M2", "Sparse",  "Top-K Sparse Attention (k=12.5% of T)",      "BigBird / Longformer"),
    ("M3", "HSR",     "Hierarchical Sparse Routing (dense→sparse)",  "Novel"),
    ("M4", "CSCG",    "Cross-Sentence Consistency Gate",             "Novel"),
    ("M5", "SwiGLU",  "Gated FFN: (xW₁⊙SiLU(xW₃))W₂",             "Shazeer, 2020"),
    ("M6", "MoE",     "Soft Mixture-of-Experts FFN (n=4)",           "Fedus et al., 2022"),
    ("M7", "Pre-LN",  "Pre-Layer Normalisation",                     "Xiong et al., 2020"),
    ("M8", "MultiPool","CLS + Mean + Max pooling head",              "Novel"),
]
tx, ty = 0.8, 3.5
ax.text(tx, ty, "Architectural Modifications Summary", fontsize=9,
        fontweight="bold", color=C["border"], zorder=4)
headers = ["ID", "Name", "Description", "Reference"]
col_x   = [tx, tx+1.2, tx+3.5, tx+14.5]
ax.text(col_x[0], ty-0.35, headers[0], fontsize=8, fontweight="bold", color="#555", zorder=4)
ax.text(col_x[1], ty-0.35, headers[1], fontsize=8, fontweight="bold", color="#555", zorder=4)
ax.text(col_x[2], ty-0.35, headers[2], fontsize=8, fontweight="bold", color="#555", zorder=4)
ax.text(col_x[3], ty-0.35, headers[3], fontsize=8, fontweight="bold", color="#555", zorder=4)
ax.axhline(ty-0.5, xmin=0.035, xmax=0.965, color="#BDC3C7", lw=0.8, zorder=3)

for i, (mid, name, desc, ref) in enumerate(mods):
    row_y = ty - 0.75 - i * 0.32
    bg = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
    ax.add_patch(FancyBboxPatch((tx-0.2, row_y-0.13), 20.6, 0.28,
                 boxstyle="square,pad=0", linewidth=0,
                 facecolor=bg, zorder=2))
    novel = ref == "Novel"
    badge_c = C["novel"] if novel else C["mod"]
    badge(ax, col_x[0]+0.25, row_y, mid, color=badge_c, size=7)
    ax.text(col_x[1], row_y, name, fontsize=8, va="center",
            fontweight="bold", color="#1A1A1A", zorder=4)
    ax.text(col_x[2], row_y, desc, fontsize=8, va="center",
            color="#1A1A1A", zorder=4)
    ref_c = C["novel"] if novel else "#555"
    ax.text(col_x[3], row_y, ref, fontsize=8, va="center",
            color=ref_c, fontweight="bold" if novel else "normal", zorder=4)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out_path = "outputs/architecture_diagram.png"
plt.tight_layout(pad=0)
plt.savefig(out_path, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out_path}")
