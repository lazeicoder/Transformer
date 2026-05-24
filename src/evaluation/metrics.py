"""
Research-Grade Evaluation Metrics — From Scratch
===================================================
Implements every metric from scratch (no sklearn for the core math):
  1.  Confusion Matrix
  2.  Precision, Recall, F1 (macro, micro, weighted, per-class)
  3.  ROC-AUC (trapezoidal rule)
  4.  PR-AUC (Average Precision)
  5.  Matthews Correlation Coefficient (MCC)
  6.  Cohen's Kappa
  7.  Calibration (ECE — Expected Calibration Error)
  8.  Brier Score
  9.  Specificity / Sensitivity / G-mean
  10. McNemar's Test (for comparing two models)
  11. Bootstrapped Confidence Intervals (95% CI for all metrics)
  12. Ablation table generation

Why these metrics for a research paper:
  • F1 macro + MCC: robust to class imbalance (WELFake is ~50/50 but
    real-world fake news is heavily imbalanced).
  • ROC-AUC + PR-AUC: threshold-independent; PR-AUC is preferred when
    one class is much smaller (mirrors deployment scenarios).
  • ECE + Brier: measure model calibration — a miscalibrated model that
    assigns 99% confidence to wrong predictions is dangerous in deployment.
  • Cohen's Kappa: accounts for chance agreement (important for crowd-
    sourced label sets like WELFake).
  • McNemar: formal hypothesis test for claiming one architecture is
    significantly better than another (required in NLP papers post-2020).
  • Bootstrap CI: statistical rigor — single-point estimates without CIs
    are now widely rejected at top venues (ACL, EMNLP, AAAI).
"""

import math
import random
import warnings
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict

import torch
import numpy as np


# ===========================================================================
# Helper: safe division
# ===========================================================================
def safe_div(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom > 0 else default


# ===========================================================================
# 1. Confusion Matrix (from scratch)
# ===========================================================================
def confusion_matrix(
    y_true   : Union[List[int], np.ndarray, torch.Tensor],
    y_pred   : Union[List[int], np.ndarray, torch.Tensor],
    n_classes: int = 2,
) -> np.ndarray:
    """
    Returns (n_classes, n_classes) confusion matrix.
    cm[i][j] = number of samples where true=i, predicted=j.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    assert len(y_true) == len(y_pred)

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    return cm


# ===========================================================================
# 2. Precision, Recall, F1 (from scratch)
# ===========================================================================
def precision_recall_f1_per_class(
    cm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-class precision, recall, F1 from confusion matrix.
    Returns arrays of shape (n_classes,).
    """
    n = cm.shape[0]
    precision = np.zeros(n)
    recall    = np.zeros(n)
    f1        = np.zeros(n)

    for c in range(n):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        precision[c] = safe_div(tp, tp + fp)
        recall[c]    = safe_div(tp, tp + fn)
        f1[c]        = safe_div(2 * precision[c] * recall[c], precision[c] + recall[c])

    return precision, recall, f1


def macro_f1(cm: np.ndarray) -> float:
    _, _, f1s = precision_recall_f1_per_class(cm)
    return f1s.mean()


def weighted_f1(cm: np.ndarray) -> float:
    _, _, f1s = precision_recall_f1_per_class(cm)
    support   = cm.sum(axis=1)          # samples per class
    return safe_div((f1s * support).sum(), support.sum())


def micro_f1(cm: np.ndarray) -> float:
    """Micro-F1 = accuracy for multi-class single-label."""
    tp_sum = np.diag(cm).sum()
    total  = cm.sum()
    return safe_div(tp_sum, total)


# ===========================================================================
# 3. ROC-AUC (trapezoidal rule, from scratch)
# ===========================================================================
def roc_curve_from_scratch(
    y_true  : np.ndarray,
    y_score : np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR, thresholds) from scratch.
    y_score: probability of the positive class.
    """
    thresholds = np.unique(y_score)[::-1]   # descending
    fpr_list, tpr_list = [0.0], [0.0]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tp     = ((y_pred == 1) & (y_true == 1)).sum()
        fp     = ((y_pred == 1) & (y_true == 0)).sum()
        tpr    = safe_div(tp, n_pos)
        fpr    = safe_div(fp, n_neg)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    fpr_list.append(1.0); tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    """Area under curve using trapezoidal rule."""
    # Sort by x
    order = np.argsort(x)
    x, y  = x[order], y[order]
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz"); return float(trapz(y, x))


def roc_auc_from_scratch(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve_from_scratch(y_true, y_score)
    return auc_trapezoidal(fpr, tpr)


# ===========================================================================
# 4. PR-AUC (Average Precision, from scratch)
# ===========================================================================
def pr_curve_from_scratch(
    y_true  : np.ndarray,
    y_score : np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precision-Recall curve."""
    thresholds = np.unique(y_score)[::-1]
    prec_list, rec_list = [], []
    n_pos = (y_true == 1).sum()

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tp     = ((y_pred == 1) & (y_true == 1)).sum()
        fp     = ((y_pred == 1) & (y_true == 0)).sum()
        fn     = ((y_pred == 0) & (y_true == 1)).sum()
        prec_list.append(safe_div(tp, tp + fp))
        rec_list.append(safe_div(tp, tp + fn))

    return np.array(prec_list), np.array(rec_list)


def pr_auc_from_scratch(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (area under PR curve)."""
    prec, rec = pr_curve_from_scratch(y_true, y_score)
    # Average precision: Σ (R_n - R_{n-1}) * P_n
    order  = np.argsort(rec)
    rec, prec = rec[order], prec[order]
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz"); return float(trapz(prec, rec))


# ===========================================================================
# 5. Matthews Correlation Coefficient (from scratch)
# ===========================================================================
def mcc_from_scratch(cm: np.ndarray) -> float:
    """
    Binary MCC: (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    Extended to multi-class via the general formula.
    MCC ∈ [-1, +1]; +1=perfect, 0=random, -1=perfectly wrong.
    """
    n   = cm.shape[0]
    N   = cm.sum()

    # Generalized MCC numerator: N·(Σ cm[k,k]) - Σ_k (row_k·col_k)
    t_k = cm.sum(axis=1)   # row sums (true per class)
    p_k = cm.sum(axis=0)   # col sums (pred per class)
    correct = np.diag(cm).sum()

    numerator   = N * correct - (t_k * p_k).sum()
    denominator = math.sqrt(
        (N**2 - (p_k**2).sum()) * (N**2 - (t_k**2).sum())
    )
    return safe_div(float(numerator), float(denominator))


# ===========================================================================
# 6. Cohen's Kappa (from scratch)
# ===========================================================================
def cohen_kappa_from_scratch(cm: np.ndarray) -> float:
    """
    κ = (P_o - P_e) / (1 - P_e)
    P_o = observed agreement (accuracy)
    P_e = expected agreement by chance
    κ > 0.8 = strong agreement; > 0.6 = moderate.
    """
    N    = cm.sum()
    P_o  = safe_div(np.diag(cm).sum(), N)
    t_k  = cm.sum(axis=1) / N
    p_k  = cm.sum(axis=0) / N
    P_e  = (t_k * p_k).sum()
    return safe_div(P_o - P_e, 1.0 - P_e)


# ===========================================================================
# 7. Calibration: ECE (Expected Calibration Error, from scratch)
# ===========================================================================
def expected_calibration_error(
    y_true   : np.ndarray,
    y_probs  : np.ndarray,   # probabilities for positive class
    n_bins   : int = 15,
) -> Dict[str, float]:
    """
    ECE = Σ_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    A well-calibrated model has ECE ≈ 0.
    Overconfident models: conf >> acc → high ECE.
    ECE is increasingly required in ML papers as it reflects
    deployment reliability.

    Returns dict with 'ece', 'mce' (max calibration error),
    and per-bin data for reliability diagram plotting.
    """
    N    = len(y_true)
    bins = np.linspace(0, 1, n_bins + 1)

    ece_total = 0.0
    mce       = 0.0
    bin_data  = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        in_bin = (y_probs >= lo) & (y_probs < hi)
        if hi == 1.0:
            in_bin = (y_probs >= lo) & (y_probs <= hi)

        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bin_data.append({'conf': (lo+hi)/2, 'acc': 0, 'n': 0})
            continue

        conf = y_probs[in_bin].mean()
        acc  = y_true[in_bin].mean()
        diff = abs(acc - conf)

        ece_total += (n_in_bin / N) * diff
        mce        = max(mce, diff)
        bin_data.append({'conf': conf, 'acc': acc, 'n': int(n_in_bin)})

    return {'ece': ece_total, 'mce': mce, 'bin_data': bin_data}


# ===========================================================================
# 8. Brier Score (from scratch)
# ===========================================================================
def brier_score(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    BS = (1/N) Σ (p_i - y_i)²
    Measures probabilistic accuracy. BS=0 is perfect; BS=1 is worst.
    """
    return float(np.mean((y_probs - y_true) ** 2))


# ===========================================================================
# 9. Specificity / Sensitivity / G-mean
# ===========================================================================
def sensitivity_specificity_gmean(cm: np.ndarray) -> Dict[str, float]:
    """
    For binary classification (Fake=0, Real=1):
      Sensitivity (Recall for Fake) = TP_fake / (TP_fake + FN_fake)
      Specificity (Recall for Real) = TP_real / (TP_real + FN_real)
      G-mean = sqrt(sensitivity * specificity)
    """
    tp_fake = cm[0, 0]; fn_fake = cm[0, 1]
    tp_real = cm[1, 1]; fn_real = cm[1, 0]

    sensitivity = safe_div(tp_fake, tp_fake + fn_fake)
    specificity = safe_div(tp_real, tp_real + fn_real)
    g_mean      = math.sqrt(sensitivity * specificity)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'g_mean'     : g_mean,
    }


# ===========================================================================
# 10. McNemar's Test (from scratch)
# ===========================================================================
def mcnemar_test(
    y_true   : np.ndarray,
    y_pred_a : np.ndarray,   # predictions of model A
    y_pred_b : np.ndarray,   # predictions of model B
) -> Dict[str, float]:
    """
    McNemar's test for statistical significance between two classifiers.
    H₀: Both classifiers have the same error rate.

    Contingency table:
        b = A wrong, B right
        c = A right, B wrong
    χ² = (|b - c| - 1)² / (b + c)   [with continuity correction]
    p-value derived from χ²(df=1) distribution.

    If p < 0.05: the models are significantly different.

    Reference:
      Dietterich, T.G. (1998). Approximate Statistical Tests for
      Comparing Supervised Classification Learning Algorithms.
      Neural Computation.
    """
    y_true   = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = ((~correct_a) & correct_b).sum()   # A wrong, B right
    c = (correct_a & (~correct_b)).sum()   # A right, B wrong

    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'significant': False}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared(df=1) CDF approximation
    # Using the Wilson-Hilferty approximation for chi-squared CDF
    p_value = _chi2_sf(statistic, df=1)

    return {
        'b'          : int(b),
        'c'          : int(c),
        'statistic'  : float(statistic),
        'p_value'    : float(p_value),
        'significant': p_value < 0.05,
    }


def _chi2_sf(x: float, df: int) -> float:
    """
    Approximate survival function P(χ²(df) > x) using
    the incomplete gamma function approximation.
    """
    if x <= 0:
        return 1.0
    # For df=1: χ²(1) ~ (N(0,1))^2
    # P(χ²(1) > x) = 2 * P(N(0,1) > sqrt(x)) = erfc(sqrt(x/2) / sqrt(2))
    return math.erfc(math.sqrt(x / 2) / math.sqrt(2))


# ===========================================================================
# 11. Bootstrap Confidence Intervals (from scratch)
# ===========================================================================
def bootstrap_ci(
    y_true     : np.ndarray,
    y_pred     : np.ndarray,
    y_probs    : np.ndarray,
    metric_fn  ,           # callable(y_true, y_pred, y_probs) → float
    n_bootstrap: int  = 1000,
    ci         : float = 0.95,
    seed       : int  = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.
    Returns (point_estimate, lower_bound, upper_bound).

    Algorithm:
      1. Draw n_bootstrap samples of size N WITH replacement.
      2. Compute metric on each bootstrap sample.
      3. CI = [α/2 percentile, 1-α/2 percentile] of bootstrap distribution.

    This is the standard approach for reporting uncertainty in NLP results.
    Reference: Berg-Kirkpatrick et al. (2012). An Empirical Investigation
    of Statistical Significance in NLP.
    """
    rng    = np.random.default_rng(seed)
    N      = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx     = rng.integers(0, N, size=N)
        bt_true = y_true[idx]
        bt_pred = y_pred[idx]
        bt_prob = y_probs[idx]
        try:
            score = metric_fn(bt_true, bt_pred, bt_prob)
            scores.append(score)
        except Exception:
            pass

    scores = np.array(scores)
    alpha  = 1.0 - ci
    lower  = float(np.percentile(scores, 100 * alpha / 2))
    upper  = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    point  = float(metric_fn(y_true, y_pred, y_probs))
    return point, lower, upper


# ===========================================================================
# 12. Master Evaluation Function
# ===========================================================================
def compute_all_metrics(
    logits    : torch.Tensor,   # (N, n_classes)
    labels    : torch.Tensor,   # (N,)
    prefix    : str  = '',
    n_classes : int  = 2,
    n_bootstrap: int = 200,     # set to 1000 for final test eval
) -> Dict[str, float]:
    """
    Computes the complete research-grade metric suite.
    Returns a flat dict suitable for logging/tables.

    Prefix: 'val_' or 'test_' prepended to all keys.
    """
    p = f"{prefix}_" if prefix else ""

    probs_all = torch.softmax(logits, dim=-1).numpy()   # (N, K)
    preds     = logits.argmax(dim=-1).numpy()
    y_true    = labels.numpy()
    # Positive class probability (class 1 = Real)
    y_score   = probs_all[:, 1]

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, preds, n_classes)

    # --- Per-class P/R/F1 ---
    prec_pc, rec_pc, f1_pc = precision_recall_f1_per_class(cm)

    metrics = {}

    # Scalar metrics
    metrics[f"{p}accuracy"]    = float(micro_f1(cm))
    metrics[f"{p}f1_macro"]    = float(macro_f1(cm))
    metrics[f"{p}f1_weighted"] = float(weighted_f1(cm))
    metrics[f"{p}f1_micro"]    = float(micro_f1(cm))
    metrics[f"{p}mcc"]         = float(mcc_from_scratch(cm))
    metrics[f"{p}kappa"]       = float(cohen_kappa_from_scratch(cm))
    metrics[f"{p}brier"]       = float(brier_score(y_true, y_score))

    # Per-class F1
    for c in range(n_classes):
        cls_name = {0: 'fake', 1: 'real'}.get(c, str(c))
        metrics[f"{p}f1_{cls_name}"]        = float(f1_pc[c])
        metrics[f"{p}precision_{cls_name}"] = float(prec_pc[c])
        metrics[f"{p}recall_{cls_name}"]    = float(rec_pc[c])

    # ROC-AUC
    try:
        metrics[f"{p}roc_auc"] = float(roc_auc_from_scratch(y_true, y_score))
        metrics[f"{p}pr_auc"]  = float(pr_auc_from_scratch(y_true, y_score))
    except Exception as e:
        warnings.warn(f"AUC computation failed: {e}")
        metrics[f"{p}roc_auc"] = 0.0
        metrics[f"{p}pr_auc"]  = 0.0

    # Calibration
    ece_result = expected_calibration_error(y_true, y_score)
    metrics[f"{p}ece"] = ece_result['ece']
    metrics[f"{p}mce"] = ece_result['mce']

    # Sensitivity / Specificity / G-mean (binary only)
    if n_classes == 2:
        sens_spec = sensitivity_specificity_gmean(cm)
        metrics[f"{p}sensitivity"] = sens_spec['sensitivity']
        metrics[f"{p}specificity"] = sens_spec['specificity']
        metrics[f"{p}g_mean"]      = sens_spec['g_mean']

    # Bootstrap CI for macro-F1
    def macro_f1_fn(yt, yp, ys):
        cm_ = confusion_matrix(yt, yp, n_classes)
        return macro_f1(cm_)

    point, lo, hi = bootstrap_ci(y_true, preds, y_score, macro_f1_fn, n_bootstrap)
    metrics[f"{p}f1_macro_ci_low"]  = lo
    metrics[f"{p}f1_macro_ci_high"] = hi

    return metrics


# ===========================================================================
# 13. Pretty Print for Paper Table
# ===========================================================================
def print_results_table(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Prints results in a format suitable for copy-paste into a LaTeX table.
    """
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    key_order = [
        'accuracy', 'f1_macro', 'f1_weighted', 'f1_micro',
        'f1_fake', 'f1_real',
        'precision_fake', 'precision_real',
        'recall_fake', 'recall_real',
        'roc_auc', 'pr_auc',
        'mcc', 'kappa',
        'sensitivity', 'specificity', 'g_mean',
        'ece', 'mce', 'brier',
    ]
    # Detect prefix
    prefix = ''
    for k in metrics:
        if '_' in k:
            candidate = k.split('_')[0] + '_'
            if all(kk.startswith(candidate) for kk in metrics if not kk.endswith('_ci_low') and not kk.endswith('_ci_high')):
                prefix = candidate
                break

    for key in key_order:
        full_key = prefix + key
        if full_key in metrics:
            ci_lo = metrics.get(prefix + key.replace('f1_macro', 'f1_macro_ci_low'), None)
            ci_hi = metrics.get(prefix + key.replace('f1_macro', 'f1_macro_ci_high'), None)
            val   = metrics[full_key]
            if ci_lo is not None and ci_hi is not None and 'f1_macro' in key:
                print(f"  {key:<25} {val:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
            else:
                print(f"  {key:<25} {val:.4f}")
    print()


# ===========================================================================
# 14. Ablation Study Helper
# ===========================================================================
class AblationTracker:
    """
    Records metrics for each ablated configuration.
    Generates a comparison table for the paper.
    """
    def __init__(self):
        self.results: List[Dict] = []

    def add(self, config_name: str, metrics: Dict[str, float]):
        self.results.append({'config': config_name, **metrics})

    def to_latex_table(
        self,
        metric_keys: List[str],
        prefix: str = 'test_',
    ) -> str:
        """
        Generates LaTeX table rows for ablation section.
        Best value per column is bolded.
        """
        cols   = [prefix + k for k in metric_keys]
        header = ' & '.join(['Configuration'] + metric_keys) + r' \\'
        lines  = [r'\begin{tabular}{l' + 'c' * len(metric_keys) + r'}',
                  r'\toprule', header, r'\midrule']

        # Find best per column
        best = {}
        for col in cols:
            vals    = [r.get(col, 0) for r in self.results]
            best[col] = max(vals)

        for row in self.results:
            cells = [row['config']]
            for col in cols:
                val = row.get(col, 0)
                if val == best[col]:
                    cells.append(r'\textbf{' + f'{val:.4f}' + '}')
                else:
                    cells.append(f'{val:.4f}')
            lines.append(' & '.join(cells) + r' \\')

        lines += [r'\bottomrule', r'\end{tabular}']
        return '\n'.join(lines)

    def print_comparison(self, metric_keys: List[str], prefix: str = 'test_'):
        """Console-friendly comparison table."""
        print(f"\n{'='*80}")
        print("ABLATION STUDY RESULTS")
        print(f"{'='*80}")
        header = f"{'Configuration':<35}" + ''.join(f"{k:>12}" for k in metric_keys)
        print(header)
        print('-' * 80)
        for row in self.results:
            vals = f"{'Config':<35}"
            vals = f"{row['config']:<35}"
            for k in metric_keys:
                col = prefix + k
                vals += f"{row.get(col, 0):>12.4f}"
            print(vals)
