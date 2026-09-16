"""
csa_utils.py — shared helpers for the Calibrated Selective Attribution project.
Imported by Phase 4-7 notebooks so conformal math and plotting are defined once.

Author: Monirul I. Mahmud | Supervisor: Dr. Justin Zhan
"""
import numpy as np
import pandas as pd

# =====================================================================
# Plotting style (DPI 800, no titles) — matches Phase 0-3 figures
# =====================================================================
def apply_plot_style(plt):
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 800,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 12, "axes.linewidth": 0.8,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.15,
    })

# shared palette
COLORS = {
    "correct": "#2D7D46", "incorrect": "#C44E52",
    "primary": "#2B6CB0", "secondary": "#D4A843", "accent": "#7B4FAF",
    "gray": "#8C8C8C",
    "attacked": "#2B6CB0", "clean": "#2D7D46", "adaptive": "#C44E52",
    "llama3.2-3b": "#2B6CB0", "qwen2.5-3b": "#D97706",
    "attntrace": "#2B6CB0", "ragorigin": "#D4A843",
}

# =====================================================================
# Calibration metrics
# =====================================================================
def expected_calibration_error(confidences, correct, n_bins=10):
    """
    ECE: average |accuracy - confidence| over equal-width confidence bins.
    confidences: array in [0,1]; correct: boolean array.
    Returns (ece, mce, bin_table).
    """
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    # scale confidences to [0,1] if not already
    lo, hi = confidences.min(), confidences.max()
    if hi > lo:
        conf01 = (confidences - lo) / (hi - lo)
    else:
        conf01 = np.zeros_like(confidences)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    rows = []
    n = len(conf01)
    for i in range(n_bins):
        m = (conf01 >= bins[i]) & (conf01 < bins[i+1] if i < n_bins-1 else conf01 <= bins[i+1])
        if m.sum() == 0:
            rows.append((bins[i], bins[i+1], 0, np.nan, np.nan)); continue
        acc = correct[m].mean()
        avg_conf = conf01[m].mean()
        gap = abs(acc - avg_conf)
        ece += (m.sum() / n) * gap
        mce = max(mce, gap)
        rows.append((bins[i], bins[i+1], int(m.sum()), acc, avg_conf))
    bin_table = pd.DataFrame(rows, columns=["bin_lo","bin_hi","count","accuracy","avg_conf"])
    return ece, mce, bin_table

# =====================================================================
# Split-conformal selective prediction (Algorithm 1 in the proposal)
# =====================================================================
def fit_gap_threshold(cal_gaps_correct, alpha):
    """
    Fit q_hat on the CORRECT calibration cases' gaps.
    Nonconformity score is s = -gap, so we answer when gap >= q_hat.
    Uses the finite-sample correction floor(alpha*(n+1))/n (Algorithm 1, line 11).
    """
    g = np.asarray(cal_gaps_correct, dtype=float)
    n = len(g)
    if n == 0:
        return -np.inf
    level = np.floor(alpha * (n + 1)) / n
    level = min(max(level, 0.0), 1.0)
    # answer when gap is in the top (1-level) of correct-case gaps
    return float(np.quantile(g, level, method="lower"))

def selective_decision(gaps, q_hat):
    """Return boolean array: True = answer, False = abstain."""
    return np.asarray(gaps, dtype=float) >= q_hat

def coverage_risk_curve(gaps, correct, n_points=200):
    """
    Sweep a gap threshold and return (coverage, risk) arrays.
    coverage = fraction answered; risk = error rate among answered.
    """
    gaps = np.asarray(gaps, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    ts = np.linspace(gaps.min() - 1e-9, gaps.max() + 1e-9, n_points)
    covs, risks = [], []
    for t in ts:
        ans = gaps >= t
        n = ans.sum()
        if n == 0:
            covs.append(0.0); risks.append(0.0); continue
        covs.append(n / len(gaps))
        risks.append((~correct & ans).sum() / n)
    return np.array(covs), np.array(risks)

def evaluate_alpha(cal_df, test_df, alpha, gap_col="gap", correct_col="correct"):
    """
    Full split-conformal evaluation at one alpha.
    cal_df / test_df: DataFrames with gap and correct columns (attacked cases only).
    Returns dict with q_hat, coverage, selective_risk, bound, bound_holds.
    """
    cal_correct_gaps = cal_df.loc[cal_df[correct_col] == True, gap_col].values
    q_hat = fit_gap_threshold(cal_correct_gaps, alpha)
    ans = selective_decision(test_df[gap_col].values, q_hat)
    n_ans = int(ans.sum())
    if n_ans == 0:
        return dict(alpha=alpha, q_hat=q_hat, coverage=0.0, selective_risk=0.0,
                    bound=np.inf, bound_holds=True, n_answered=0)
    correct = test_df[correct_col].values.astype(bool)
    risk = float((~correct & ans).sum() / n_ans)
    cov = float(n_ans / len(test_df))
    bound = alpha / cov if cov > 0 else np.inf
    return dict(alpha=alpha, q_hat=float(q_hat), coverage=cov, selective_risk=risk,
                bound=float(bound), bound_holds=bool(risk <= bound + 1e-9), n_answered=n_ans)

def bootstrap_ci(values, n_boot=1000, ci=95, seed=0):
    """Percentile bootstrap CI for the mean of `values`."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return (float(v.mean()), float(lo), float(hi))

def auroc(signal, correct):
    """AUROC of a confidence signal against correctness. 0.5 = useless."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(correct, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, np.asarray(signal, dtype=float)))
