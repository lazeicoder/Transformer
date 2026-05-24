"""
scripts/run_all.py — Complete Pipeline Entry Point
=====================================================
Runs the full pipeline end-to-end:
  1. EDA
  2. Training
  3. Explainability analysis
  4. Ablation study

Usage:
  python scripts/run_all.py --csv data/WELFake_Dataset.csv --ablation

Flags:
  --csv       path to WELFake_Dataset.csv (required for real data)
  --ablation  run ablation study (adds ~5x training time)
  --epochs    number of training epochs (default: 20)
  --d_model   model dimension (default: 256; use 512 for stronger results)
  --demo      skip training, run with a small synthetic dataset
"""

import sys, argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="FakeNews Transformer — Full Pipeline")
    parser.add_argument('--csv',       type=str,  default=None,  help='Path to WELFake CSV')
    parser.add_argument('--ablation',  action='store_true',       help='Run ablation study')
    parser.add_argument('--epochs',    type=int,  default=20,    help='Training epochs')
    parser.add_argument('--d_model',   type=int,  default=256,   help='Model dimension')
    parser.add_argument('--num_layers',type=int,  default=6,     help='Encoder layers')
    parser.add_argument('--num_heads', type=int,  default=8,     help='Attention heads')
    parser.add_argument('--demo',      action='store_true',       help='Demo mode (synthetic data)')
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nFakeNews Transformer Pipeline")
    print(f"Device: {device}")
    print(f"Config: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)

    import importlib.util

    csv_path = None if args.demo else args.csv

    # ── Phase 1: EDA ──────────────────────────────────────────────
    print("\n[1/4] Running EDA...")
    eda_spec  = importlib.util.spec_from_file_location("eda",  ROOT / "notebooks/01_eda.py")
    eda_mod   = importlib.util.module_from_spec(eda_spec)
    eda_spec.loader.exec_module(eda_mod)
    df, tokenizer = eda_mod.main(csv_path)

    # ── Phase 2: Training ─────────────────────────────────────────
    print("\n[2/4] Training model...")
    train_spec = importlib.util.spec_from_file_location("train", ROOT / "notebooks/02_training.py")
    train_mod  = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_mod)

    # Override config before training
    from src.training.trainer import TrainingConfig
    cfg = TrainingConfig(
        d_model    = args.d_model,
        num_heads  = args.num_heads,
        num_layers = args.num_layers,
        epochs     = args.epochs,
        output_dir = str(ROOT / 'outputs'),
        fp16       = (device.type == 'cuda'),
    )

    trained_model, test_metrics = train_mod.main(csv_path)

    # ── Phase 3: Explainability ───────────────────────────────────
    print("\n[3/4] Running explainability analysis...")
    expl_spec = importlib.util.spec_from_file_location("expl", ROOT / "notebooks/03_explainability.py")
    expl_mod  = importlib.util.module_from_spec(expl_spec)
    expl_spec.loader.exec_module(expl_mod)
    expl_mod.main(csv_path)

    # ── Phase 4: Ablation ─────────────────────────────────────────
    if args.ablation:
        print("\n[4/4] Running ablation study...")
        from src.data.pipeline       import load_welfake
        from src.training.ablation   import run_ablation_study
        from src.training.trainer    import TrainingConfig

        abl_cfg = TrainingConfig(
            d_model    = args.d_model,
            num_heads  = args.num_heads,
            num_layers = args.num_layers,
            output_dir = str(ROOT / 'outputs'),
        )

        train_loader, val_loader, test_loader, _ = load_welfake(
            csv_path  = csv_path,
            tokenizer = tokenizer,
            max_len   = 512, batch_size=16, num_workers=0,
        )

        tracker = run_ablation_study(
            train_loader, val_loader, test_loader,
            vocab_size    = tokenizer.vocab_size(),
            base_config   = abl_cfg,
            ablation_epochs = 5,
            device        = device,
            output_dir    = str(ROOT / 'outputs' / 'ablation'),
        )
    else:
        print("\n[4/4] Ablation study skipped. Use --ablation to run.")

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Outputs in: {ROOT / 'outputs'}")
    print(f"  Best model: {ROOT / 'outputs' / 'best_model.pt'}")
    print(f"  Run the web app: python app.py")


if __name__ == '__main__':
    main()
