"""
app.py — Gradio Web Application
=================================
Interactive demo for the FakeNews Transformer.

Run: python app.py
Then open: http://localhost:7860

Features:
  - Paste any article/headline → get Fake/Real prediction with confidence
  - Token attribution heatmap (Attention Rollout)
  - Probability gauge chart
  - Model stats panel
"""

import sys, json
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data.pipeline          import SimpleBPETokenizer
from src.transformer.encoder    import FakeNewsTransformer
from src.explainability.attribution import AttentionRollout, render_token_heatmap


# =============================================================================
# Load Model & Tokenizer
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL  = None
TOK    = None
ROLLOUT= AttentionRollout()

def load_assets():
    global MODEL, TOK
    tok_path  = ROOT / 'outputs' / 'tokenizer.json'
    ckpt_path = ROOT / 'outputs' / 'best_model.pt'

    if not tok_path.exists() or not ckpt_path.exists():
        # Demo mode: untrained model
        print("WARNING: Trained model not found. Running in demo mode (random predictions).")
        TOK = SimpleBPETokenizer()
        TOK.train(["fake news real breaking government health"] * 500, vocab_size=3000)
        MODEL = FakeNewsTransformer(vocab_size=TOK.vocab_size(),
                                     d_model=128, num_heads=4, num_layers=2)
    else:
        TOK   = SimpleBPETokenizer.load(str(tok_path))
        ckpt  = torch.load(ckpt_path, map_location=DEVICE)
        cfg   = ckpt.get('config', {})
        MODEL = FakeNewsTransformer(
            vocab_size         = TOK.vocab_size(),
            d_model            = cfg.get('d_model', 256),
            num_heads          = cfg.get('num_heads', 8),
            num_layers         = cfg.get('num_layers', 6),
            dropout            = 0.0,
            max_len            = cfg.get('max_len', 512),
            n_experts          = cfg.get('n_experts', 4),
            n_classes          = cfg.get('n_classes', 2),
            classifier_dropout = 0.0,
        )
        MODEL.load_state_dict(ckpt['model_state'])
        print(f"Model loaded. Parameters: {sum(p.numel() for p in MODEL.parameters()):,}")

    MODEL.to(DEVICE).eval()


# =============================================================================
# Prediction Function
# =============================================================================
def predict(text: str):
    if not text.strip():
        return "Please enter some text.", "", "", ""

    text = text.strip()
    ids  = TOK.encode(text, max_length=512, add_special=True)
    token_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    seg_ids   = torch.zeros_like(token_ids)

    with torch.no_grad():
        out    = MODEL(token_ids, seg_ids)
        probs  = torch.softmax(out['logits'], dim=-1)[0].cpu().numpy()
        pred   = out['logits'].argmax(dim=-1).item()
        attn_layers = [a.detach().cpu() for a in out['attn_weights']]

    fake_p = float(probs[0])
    real_p = float(probs[1])
    label  = 'FAKE NEWS 🚨' if pred == 0 else 'REAL NEWS ✅'

    # Confidence meter text
    conf      = max(fake_p, real_p)
    conf_desc = (
        "Very High Confidence" if conf > 0.90 else
        "High Confidence"      if conf > 0.75 else
        "Moderate Confidence"  if conf > 0.60 else
        "Low Confidence (uncertain)"
    )

    verdict = f"""
## Verdict: {label}

| | Probability |
|---|---|
| 🔴 Fake | {fake_p:.1%} |
| 🟢 Real | {real_p:.1%} |

**{conf_desc}** ({conf:.1%})
    """.strip()

    # Attention Rollout Heatmap HTML
    heatmap_html = ""
    if attn_layers:
        try:
            attribution = ROLLOUT.compute(attn_layers)[0].numpy()
            tokens = []
            for tid in ids:
                tok_str = TOK.inv_vocab.get(tid, f'[{tid}]').replace('</w>', '')
                tokens.append(tok_str or '▪')
            tokens = tokens[:len(attribution)]
            attribution = attribution[:len(tokens)]
            heatmap_html = render_token_heatmap(
                tokens, attribution,
                title=f"Token Importance: {label}"
            )
            # Extract just the body content for Gradio HTML component
            import re
            body_match = re.search(r'<body[^>]*>(.*?)</body>', heatmap_html, re.DOTALL)
            if body_match:
                heatmap_html = f'<div style="padding:1rem;">{body_match.group(1)}</div>'
        except Exception as e:
            heatmap_html = f"<p>Heatmap unavailable: {e}</p>"

    # Model confidence bar (HTML)
    bar_color  = '#D85A30' if pred == 0 else '#1D9E75'
    bar_width  = int(conf * 100)
    gauge_html = f"""
<div style="padding:1rem;font-family:sans-serif;">
  <div style="font-size:13px;color:#888;margin-bottom:6px;">Confidence</div>
  <div style="background:#eee;border-radius:8px;height:20px;overflow:hidden;">
    <div style="background:{bar_color};width:{bar_width}%;height:100%;
                border-radius:8px;transition:width 0.4s;"></div>
  </div>
  <div style="font-size:13px;margin-top:4px;color:{bar_color};font-weight:bold;">
    {conf:.1%} — {conf_desc}
  </div>
</div>
    """.strip()

    return verdict, heatmap_html, gauge_html


# =============================================================================
# Gradio Interface
# =============================================================================
def build_interface():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return None

    load_assets()

    examples = [
        ["SHOCKING: Government HIDES vaccine ingredients — whistleblower exposes everything!! SHARE before deleted!"],
        ["New peer-reviewed study in The Lancet finds that mRNA vaccines reduce severe COVID-19 by 93% in adults over 60."],
        ["You won't BELIEVE what scientists found — they've been lying to us about climate change all along!!"],
        ["The Federal Reserve raised interest rates by 25 basis points, citing persistent core inflation data."],
        ["EXCLUSIVE: Secret elite society controlling world governments — leaked documents PROVE it!"],
        ["NASA confirms the discovery of organic molecules on Mars, strengthening the case for ancient microbial life."],
    ]

    with gr.Blocks(
        title="FakeNews Transformer",
        theme=gr.themes.Default(
            primary_hue="orange",
            secondary_hue="green",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css="""
            .verdict-box { border-radius: 12px; padding: 1rem; }
            .header { text-align: center; margin-bottom: 1.5rem; }
        """,
    ) as demo:

        gr.HTML("""
            <div class="header">
                <h1>🔍 Fake News Transformer</h1>
                <p style="color:#666;max-width:600px;margin:auto;">
                    Research-grade fake news detector built with a custom Transformer
                    featuring RoPE, SwiGLU, HSR Sparse Attention, MoE FFN, CSCG Gate,
                    and Multi-Pool Classification Head.
                </p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Article Text or Headline",
                    placeholder="Paste a news article, headline, or claim here...",
                    lines=6,
                )
                submit_btn = gr.Button("Analyze", variant="primary", size="lg")

                gr.Examples(
                    examples=examples,
                    inputs=text_input,
                    label="Example Inputs",
                )

            with gr.Column(scale=1):
                verdict_out = gr.Markdown(label="Verdict")
                gauge_out   = gr.HTML(label="Confidence")

        with gr.Row():
            heatmap_out = gr.HTML(label="Token Attribution Heatmap (Attention Rollout)")

        with gr.Accordion("ℹ️ Model Architecture & Modifications", open=False):
            gr.Markdown("""
### Architectural Modifications (M1–M8)

| ID | Modification | Contribution |
|----|---|---|
| M1 | **Rotary Positional Encoding (RoPE)** | Relative position awareness, better on variable-length articles |
| M2 | **Sparse Top-K Attention** | Hard attention gate, focuses on informative tokens |
| M3 | **Hierarchical Sparse Routing (HSR)** | Dense lower layers (syntax) + Sparse upper layers (semantics) |
| M4 | **Cross-Sentence Consistency Gate (CSCG)** | Detects headline/body contradictions |
| M5 | **SwiGLU Activation** | Smoother gradients, gated feature selection |
| M6 | **Mixture-of-Experts FFN (MoE)** | Specialised sub-networks for different linguistic patterns |
| M7 | **Pre-Layer Normalization** | More stable training, no warmup required |
| M8 | **Multi-Pool Classification Head** | CLS + Mean + Max pooling for richer representations |

### Evaluation Metrics
Accuracy · Macro-F1 · ROC-AUC · PR-AUC · MCC · Cohen's Kappa ·
ECE · Brier Score · Sensitivity · Specificity · G-mean · McNemar's Test
+ 95% Bootstrap Confidence Intervals
            """)

        submit_btn.click(
            fn=predict,
            inputs=text_input,
            outputs=[verdict_out, heatmap_out, gauge_out],
        )
        text_input.submit(
            fn=predict,
            inputs=text_input,
            outputs=[verdict_out, heatmap_out, gauge_out],
        )

    return demo


if __name__ == '__main__':
    demo = build_interface()
    if demo:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
        )
