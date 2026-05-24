# FakeNews Transformer
---

## Architectural Modifications (M1–M8)

| ID | Modification | Component | Research Reference |
|----|---|---|---|
| **M1** | Rotary Positional Encoding (RoPE) | Attention | Su et al., 2021 |
| **M2** | Sparse Top-K Attention | Attention | Bigbird, Longformer |
| **M3** | Hierarchical Sparse Routing (HSR) | Architecture | **Novel** |
| **M4** | Cross-Sentence Consistency Gate (CSCG) | Post-encoder | **Novel** |
| **M5** | SwiGLU Activation | FFN | Shazeer, 2020 |
| **M6** | Mixture-of-Experts FFN (soft routing) | FFN | Fedus et al., 2022 |
| **M7** | Pre-Layer Normalization | Normalization | Xiong et al., 2020 |
| **M8** | Multi-Pool Classification Head | Classifier | **Novel** |

---

## Project Structure

```
fakenews_transformer/
│
├── src/
│   ├── transformer/
│   │   ├── embeddings.py    ← TokenEmb, RoPE, SegmentEmb, HybridEmb
│   │   ├── attention.py     ← MHA + RoPE, SparseAttn, CSCG
│   │   ├── feedforward.py   ← ReLU/GeLU/SwiGLU FFN, MoE FFN
│   │   └── encoder.py       ← TransformerEncoderLayer, TransformerEncoder,
│   │                           MultiPoolHead, FakeNewsTransformer
│   │
│   ├── data/
│   │   └── pipeline.py      ← BPE Tokenizer, SegmentLabeler, WELFakeDataset,
│   │                           DataCollator, EDAAnalyzer
│   │
│   ├── training/
│   │   ├── trainer.py       ← LabelSmoothingCE, CosineAnnealingWithWarmup,
│   │   │                       TrainingConfig, CheckpointManager, train()
│   │   └── ablation.py      ← 9-config ablation study runner
│   │
│   ├── evaluation/
│   │   └── metrics.py       ← 12 metrics from scratch + Bootstrap CI +
│   │                           McNemar's test + AblationTracker
│   │
│   └── explainability/
│       └── attribution.py   ← AttentionRollout, GradientSaliency,
│                               IntegratedGradients, HeadAnalyzer
│
├── notebooks/
│   ├── 01_eda.py            ← EDA plots (class dist, length, bigrams, TTR, BPE)
│   ├── 02_training.py       ← Training + curves + confusion matrix + ROC/PR
│   └── 03_explainability.py ← Rollout heatmaps + head entropy + calibration
│
├── scripts/
│   └── run_all.py           ← One-command full pipeline
│
├── tests/
│   └── test_all.py          ← 35+ unit tests across all modules
│
├── configs/
│   └── model_config.json    ← All hyperparameters
│
├── app.py                   ← Gradio web demo
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone / download the project
cd fakenews_transformer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate.bat     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

No dataset download required. The dataset ([WELFake](https://huggingface.co/datasets/davanstrien/WELFake), ~150 MB) is downloaded automatically from HuggingFace on first run and cached locally.

---

## Usage

### Run full pipeline (recommended)
```bash
python scripts/run_all.py --epochs 20
```

### Run with ablation study (for paper)
```bash
python scripts/run_all.py --epochs 20 --ablation
```

### Run with synthetic data only (no download)
```bash
python scripts/run_all.py --demo
```

### Use a local CSV instead of auto-download
```bash
python scripts/run_all.py --csv data/WELFake_Dataset.csv --epochs 20
```

### Run individual phases
```bash
# Phase 1: EDA
python notebooks/01_eda.py

# Phase 2: Training
python notebooks/02_training.py

# Phase 3: Explainability
python notebooks/03_explainability.py
```

### Run tests
```bash
python -m pytest tests/test_all.py -v
```

### Launch web demo
```bash
python app.py
# Open: http://localhost:7860
```

---

## Research Paper Outline

**Title:** *Hierarchical Sparse Routing Transformer with Cross-Sentence Consistency Gate for Fake News Detection*

**Abstract:** We propose a modified Transformer encoder with 8 architectural improvements...

### Sections:
1. **Introduction** — Fake news impact, limitations of BERT-based approaches
2. **Related Work** — Survey of DL-based fake news detection (2019–2024)
3. **Methodology**
   - 3.1 Dataset & Preprocessing (WELFake, BPE, segment labeling)
   - 3.2 Architecture Overview (diagram of full model)
   - 3.3 Modifications M1–M8 (each with equation + rationale)
4. **Experiments**
   - 4.1 Training Setup (hardware, hyperparameters)
   - 4.2 Main Results (Table with all 12 metrics + 95% CI)
   - 4.3 Ablation Study (Table showing contribution of each M)
   - 4.4 McNemar's Test (statistical significance vs BERT baseline)
5. **Analysis**
   - 5.1 Attention Visualization (rollout heatmaps)
   - 5.2 Error Analysis
   - 5.3 Calibration (Reliability Diagram)
6. **Conclusion**

---

## Key Equations (for paper Section 3)

**RoPE** (M1):
```
q_m = R_θ_m · q,  k_n = R_θ_n · k
<q_m, k_n> = <q, R_θ(n-m) · k>    ← depends only on relative position
```

**SwiGLU** (M5):
```
FFN(x) = (x·W₁ ⊙ SiLU(x·W₃)) · W₂
SiLU(x) = x · σ(x)
```

**CSCG** (M4 — novel):
```
gate_i = σ(W_gate · [x_CLS ; x_i])
x_i'   = gate_i ⊙ x_i + (1-gate_i) ⊙ x_i
```

**Multi-Pool** (M8 — novel):
```
h = [x_CLS ; mean({x_i}) ; max({x_i})]    → Linear → GELU → Linear
```
