"""
Training Loop — From Scratch
==============================
Implements:
  1. LabelSmoothingCrossEntropy loss (regularization) [from scratch]
  2. CosineAnnealingWithWarmup scheduler [from scratch]
  3. Gradient clipping + gradient norm tracking
  4. Mixed-precision training (torch.cuda.amp)
  5. Checkpoint manager (save best by val F1, not accuracy)
  6. WandB-compatible metric logging (console fallback)
  7. Early stopping with patience
  8. MoE auxiliary loss integration

Research decisions documented inline.
"""

import os
import math
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..transformer.encoder import FakeNewsTransformer


# ===========================================================================
# 1. Label Smoothing Cross-Entropy [from scratch]
# ===========================================================================
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing replaces the hard 0/1 targets with:
        y_smooth = (1 - ε) * y_one_hot + ε / K
    where K = n_classes.

    Research rationale:
      Fake news labels in WELFake were crowd-sourced and may contain noise.
      Label smoothing acts as a regularizer that prevents the model from
      becoming overconfident (high logit magnitude), improving calibration
      and generalization. Typically ε = 0.1 for NLP tasks.

    Reference:
      Müller et al. (2019). When Does Label Smoothing Help? NeurIPS.
    """
    def __init__(self, n_classes: int = 2, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing  = smoothing
        self.n_classes  = n_classes
        self.reduction  = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, n_classes) — raw logits
        targets : (B,)           — class indices
        """
        log_probs = F.log_softmax(logits, dim=-1)             # (B, K)

        # Hard target component
        nll_loss  = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        # Smooth target component: -1/K * Σ log_probs
        smooth_loss = -log_probs.mean(dim=-1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ===========================================================================
# 2. Cosine Annealing Scheduler with Linear Warmup [from scratch]
# ===========================================================================
class CosineAnnealingWithWarmup:
    """
    Learning rate schedule:
      Phase 1 (0 → warmup_steps)  : linear ramp from 0 to lr_max
      Phase 2 (warmup → total)    : cosine decay from lr_max to lr_min

    Formula (phase 2):
      lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(π*(t-warmup)/decay_steps))

    Research rationale:
      Warmup prevents large gradient updates at the start when the model
      weights are random. Cosine decay is smoother than step decay and
      avoids the loss spike at step boundaries. This is the standard
      schedule in modern Transformer training (Chinchilla, LLaMA etc.)
    """
    def __init__(
        self,
        optimizer      : torch.optim.Optimizer,
        lr_max         : float,
        total_steps    : int,
        warmup_steps   : int  = 0,
        lr_min         : float = 0.0,
    ):
        self.optimizer     = optimizer
        self.lr_max        = lr_max
        self.lr_min        = lr_min
        self.total_steps   = total_steps
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _get_lr(self) -> float:
        t = self._step
        if t <= self.warmup_steps:
            # Linear warmup
            return self.lr_max * t / max(1, self.warmup_steps)
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            progress    = (t - self.warmup_steps) / max(1, decay_steps)
            progress    = min(progress, 1.0)
            cosine      = 0.5 * (1 + math.cos(math.pi * progress))
            return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def get_last_lr(self) -> float:
        return self._get_lr()


# ===========================================================================
# 3. Training Configuration
# ===========================================================================
@dataclass
class TrainingConfig:
    # Model
    d_model    : int   = 256
    num_heads  : int   = 8
    num_layers : int   = 6
    dropout    : float = 0.1
    max_len    : int   = 512
    n_experts  : int   = 4
    n_classes  : int   = 2

    # Training
    epochs          : int   = 20
    batch_size      : int   = 16
    lr              : float = 3e-4
    lr_min          : float = 1e-6
    weight_decay    : float = 0.01
    warmup_ratio    : float = 0.06    # 6% of total steps
    grad_clip       : float = 1.0
    label_smoothing : float = 0.1
    moe_loss_weight : float = 0.01

    # Regularization
    classifier_dropout: float = 0.3

    # I/O
    output_dir  : str = './outputs'
    seed        : int = 42

    # Training dynamics
    patience    : int   = 5     # early stopping
    eval_steps  : int   = 500   # evaluate every N steps (0=epoch-only)
    fp16        : bool  = True  # mixed precision

    def to_dict(self) -> Dict:
        return asdict(self)


# ===========================================================================
# 4. Metric Logger (WandB-compatible / console fallback)
# ===========================================================================
class MetricLogger:
    def __init__(self, use_wandb: bool = False, project: str = "fakenews-transformer"):
        self.use_wandb = use_wandb
        self.history   = []
        if use_wandb:
            try:
                import wandb
                wandb.init(project=project)
                print("WandB initialized.")
            except ImportError:
                print("WandB not installed; using console logging.")
                self.use_wandb = False

    def log(self, metrics: Dict, step: int):
        self.history.append({'step': step, **metrics})
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def save_history(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


# ===========================================================================
# 5. Checkpoint Manager
# ===========================================================================
class CheckpointManager:
    """
    Saves model + optimizer + scheduler state.
    Keeps the single best checkpoint by validation macro-F1.
    """
    def __init__(self, output_dir: str, metric_name: str = 'val_f1'):
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.best_metric = -float('inf')
        self.best_path   = None

    def save(
        self,
        model     : nn.Module,
        optimizer : torch.optim.Optimizer,
        scheduler : CosineAnnealingWithWarmup,
        epoch     : int,
        step      : int,
        metrics   : Dict,
        config    : TrainingConfig,
    ) -> bool:
        """Returns True if this is the new best checkpoint."""
        current = metrics.get(self.metric_name, -float('inf'))
        is_best = current > self.best_metric

        ckpt = {
            'epoch'         : epoch,
            'step'          : step,
            'model_state'   : model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_step': scheduler._step,
            'metrics'       : metrics,
            'config'        : config.to_dict(),
        }

        # Always save latest
        latest_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(ckpt, latest_path)

        if is_best:
            self.best_metric = current
            best_path = self.output_dir / 'best_model.pt'
            shutil.copy(latest_path, best_path)
            self.best_path = best_path
            print(f"  [BEST] {self.metric_name}={current:.4f} — saved to {best_path}")

        return is_best

    def load_best(self, model: nn.Module, device: torch.device) -> Dict:
        if self.best_path is None or not self.best_path.exists():
            raise FileNotFoundError("No best checkpoint found.")
        ckpt = torch.load(self.best_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best model (epoch={ckpt['epoch']}, "
              f"{self.metric_name}={ckpt['metrics'].get(self.metric_name, 'N/A'):.4f})")
        return ckpt['metrics']


# ===========================================================================
# 6. Single Training Step
# ===========================================================================
def train_step(
    model     : FakeNewsTransformer,
    batch     : Dict[str, torch.Tensor],
    criterion : LabelSmoothingCrossEntropy,
    optimizer : torch.optim.Optimizer,
    scheduler : CosineAnnealingWithWarmup,
    scaler    : Optional[GradScaler],
    config    : TrainingConfig,
    device    : torch.device,
) -> Dict[str, float]:
    """Single forward + backward + optimizer step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    token_ids   = batch['token_ids'].to(device)
    segment_ids = batch['segment_ids'].to(device)
    labels      = batch['labels'].to(device)

    if scaler is not None:
        with autocast():
            out      = model(token_ids, segment_ids)
            cls_loss = criterion(out['logits'], labels)
            moe_loss = out['moe_aux_loss'].to(device) * config.moe_loss_weight
            loss     = cls_loss + moe_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        out      = model(token_ids, segment_ids)
        cls_loss = criterion(out['logits'], labels)
        moe_loss = out['moe_aux_loss'].to(device) * config.moe_loss_weight
        loss     = cls_loss + moe_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

    scheduler.step()

    preds    = out['logits'].argmax(dim=-1)
    acc      = (preds == labels).float().mean().item()

    return {
        'loss'     : loss.item(),
        'cls_loss' : cls_loss.item(),
        'moe_loss' : moe_loss.item() if torch.is_tensor(moe_loss) else float(moe_loss),
        'accuracy' : acc,
        'grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm),
        'lr'       : scheduler.get_last_lr(),
    }


# ===========================================================================
# 7. Evaluation Step (no gradient)
# ===========================================================================
@torch.no_grad()
def evaluate(
    model     : FakeNewsTransformer,
    loader    : DataLoader,
    criterion : LabelSmoothingCrossEntropy,
    device    : torch.device,
) -> Dict:
    """
    Returns per-batch aggregated metrics.
    Full metrics (F1, AUC, etc.) are computed in evaluation/metrics.py.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_logits, all_labels = [], []

    for batch in loader:
        token_ids   = batch['token_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        labels      = batch['labels'].to(device)

        out      = model(token_ids, segment_ids)
        cls_loss = criterion(out['logits'], labels)

        preds    = out['logits'].argmax(dim=-1)
        correct  = (preds == labels).sum().item()

        total_loss    += cls_loss.item() * labels.size(0)
        total_correct += correct
        total_samples += labels.size(0)

        all_logits.append(out['logits'].cpu())
        all_labels.append(labels.cpu())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return {
        'loss'     : avg_loss,
        'accuracy' : accuracy,
        'logits'   : all_logits,
        'labels'   : all_labels,
    }


# ===========================================================================
# 8. Main Training Loop
# ===========================================================================
def train(
    model       : FakeNewsTransformer,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    config      : TrainingConfig,
    device      : torch.device,
    logger      : Optional[MetricLogger] = None,
) -> FakeNewsTransformer:
    """
    Full training loop with:
      - Label smoothing loss
      - Cosine LR with warmup
      - Mixed precision (if config.fp16 and CUDA available)
      - Gradient norm logging
      - Early stopping on val macro-F1
      - Best checkpoint saving
    """
    # Delayed import to avoid circular dependency
    from ..evaluation.metrics import compute_all_metrics

    model.to(device)
    criterion = LabelSmoothingCrossEntropy(config.n_classes, config.label_smoothing)
    criterion.to(device)

    # Optimizer: separate LR for classifier (10x higher — standard practice)
    encoder_params    = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())
    optimizer = AdamW(
        [
            {'params': encoder_params,    'lr': config.lr,       'weight_decay': config.weight_decay},
            {'params': classifier_params, 'lr': config.lr * 10,  'weight_decay': config.weight_decay},
        ],
        betas=(0.9, 0.999), eps=1e-8
    )

    total_steps  = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler    = CosineAnnealingWithWarmup(optimizer, config.lr, total_steps, warmup_steps, config.lr_min)

    scaler  = GradScaler() if (config.fp16 and device.type == 'cuda') else None
    ckpt_mgr= CheckpointManager(config.output_dir, metric_name='val_f1_macro')
    logger  = logger or MetricLogger()

    # Early stopping state
    patience_counter = 0
    global_step      = 0

    print(f"\n{'='*60}")
    print(f"  TRAINING: {config.epochs} epochs, {total_steps} steps")
    print(f"  Warmup: {warmup_steps} steps ({config.warmup_ratio*100:.0f}%)")
    print(f"  Device: {device}  |  FP16: {scaler is not None}")
    print(f"  Parameters: {model.count_parameters()['trainable']:,}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        running     = {'loss': 0.0, 'accuracy': 0.0, 'grad_norm': 0.0}

        for step, batch in enumerate(train_loader):
            step_metrics = train_step(
                model, batch, criterion, optimizer, scheduler, scaler, config, device
            )

            for k in running:
                running[k] += step_metrics.get(k, 0.0)

            global_step += 1

            if (step + 1) % 50 == 0:
                n = step + 1
                print(
                    f"  Epoch {epoch:02d} | Step {step+1:04d}/{len(train_loader)} "
                    f"| loss={running['loss']/n:.4f} "
                    f"| acc={running['accuracy']/n:.4f} "
                    f"| grad={running['grad_norm']/n:.3f} "
                    f"| lr={step_metrics['lr']:.2e}"
                )
                logger.log({f'train/{k}': v/n for k, v in running.items()}, global_step)

        # ── Epoch-level evaluation ──
        val_raw = evaluate(model, val_loader, criterion, device)
        val_metrics = compute_all_metrics(
            val_raw['logits'], val_raw['labels'],
            prefix='val', n_classes=config.n_classes
        )
        val_metrics['val_loss'] = val_raw['loss']

        # Combine train metrics for logging
        n_steps = len(train_loader)
        train_summary = {f'train_{k}': v/n_steps for k, v in running.items()}
        all_metrics   = {**train_summary, **val_metrics}

        elapsed = time.time() - epoch_start
        print(f"\nEpoch {epoch:02d} [{elapsed:.0f}s]")
        print(f"  Val  loss={val_metrics['val_loss']:.4f} | "
              f"acc={val_metrics['val_accuracy']:.4f} | "
              f"F1={val_metrics['val_f1_macro']:.4f} | "
              f"AUC={val_metrics.get('val_roc_auc', 0):.4f}")

        logger.log(all_metrics, global_step)

        is_best = ckpt_mgr.save(model, optimizer, scheduler, epoch, global_step,
                                 val_metrics, config)

        if not is_best:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.patience}).")
                break
        else:
            patience_counter = 0

    # Load best weights
    print("\nLoading best model...")
    ckpt_mgr.load_best(model, device)
    logger.save_history(str(Path(config.output_dir) / 'training_history.json'))

    return model
