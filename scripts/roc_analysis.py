"""Compute ROC, EER, and AUC from impostor and genuine similarity CSVs.

Inputs:
  --imp_csv: CSV with impostor similarities (if it contains 'type', rows with type=='impostor' are used; otherwise all rows)
  --gen_csv: CSV with genuine similarities (if it contains 'type', rows with type=='genuine' are used; otherwise all rows)
  --out_dir: output directory for stats, curve CSV, and plots

Outputs:
  - roc_points.csv  (threshold, FPR, TPR, FAR, FRR)
  - roc_summary.csv (AUC, EER, EER_threshold)
  - roc_curve.png
  - hist_overlay.png (optional quick visual)
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_sims(path: str, want_type: str | None) -> np.ndarray:
    df = pd.read_csv(path)
    # find similarity column
    sim_col = None
    for cand in ["similarity","cosine","corr","correlation","sim"]:
        if cand in df.columns:
            sim_col = cand
            break
    if sim_col is None:
        raise ValueError(f"No similarity column found in {path}")

    if want_type and "type" in df.columns:
        df = df[df["type"] == want_type]

    sims = df[sim_col].astype(float).values
    return sims


def _roc_from_scores(imps: np.ndarray, gens: np.ndarray, num_thresh: int = 512):
    # Scores: higher = more genuine. We'll sweep thresholds from min to max.
    all_scores = np.concatenate([imps, gens], axis=0)
    lo, hi = float(np.min(all_scores)), float(np.max(all_scores))
    # Use quantiles to place thresholds more densely where data is
    qs = np.linspace(0.0, 1.0, num_thresh)
    ths = np.quantile(all_scores, qs)

    # Pre-sort for efficient vectorization
    imps_sorted = np.sort(imps)
    gens_sorted = np.sort(gens)

    # FAR = FPR = P(impostor >= t) ; FRR = 1-TPR = P(genuine < t)
    # We can compute counts via binary search (np.searchsorted)
    n_imp = len(imps_sorted)
    n_gen = len(gens_sorted)
    FAR = []
    FRR = []
    for t in ths:
        # impostors >= t
        i_ge = n_imp - np.searchsorted(imps_sorted, t, side="left")
        far = i_ge / max(n_imp, 1)

        # genuine < t
        g_lt = np.searchsorted(gens_sorted, t, side="left")
        frr = g_lt / max(n_gen, 1)

        FAR.append(far)
        FRR.append(frr)

    FAR = np.asarray(FAR)
    FRR = np.asarray(FRR)
    TPR = 1.0 - FRR
    FPR = FAR

    # AUC via trapezoid on FPR-TPR curve (sorted by FPR)
    order = np.argsort(FPR)
    auc = float(np.trapz(TPR[order], FPR[order]))

    # EER: point where |FAR-FRR| minimized
    diff = np.abs(FAR - FRR)
    k = int(np.argmin(diff))
    eer = float((FAR[k] + FRR[k]) / 2.0)
    eer_th = float(ths[k])

    return ths, FPR, TPR, FAR, FRR, auc, eer, eer_th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imp_csv", required=True, help="path to impostor CSV")
    ap.add_argument("--gen_csv", required=True, help="path to genuine CSV")
    ap.add_argument("--out_dir", default="output/verification_analysis")
    ap.add_argument("--num_thresholds", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    imps = _get_sims(args.imp_csv, want_type="impostor")
    gens = _get_sims(args.gen_csv, want_type="genuine")
    if len(imps) == 0 or len(gens) == 0:
        raise SystemExit("Empty impostor or genuine arrays — check inputs.")

    ths, FPR, TPR, FAR, FRR, auc, eer, eer_th = _roc_from_scores(imps, gens, num_thresh=args.num_thresholds)

    # save points
    pts = pd.DataFrame({
        "threshold": ths,
        "FPR": FPR,
        "TPR": TPR,
        "FAR": FAR,
        "FRR": FRR,
    })
    pts.to_csv(os.path.join(args.out_dir, "roc_points.csv"), index=False)

    # summary
    pd.DataFrame([{ "AUC": auc, "EER": eer, "EER_threshold": eer_th }]).to_csv(
        os.path.join(args.out_dir, "roc_summary.csv"), index=False
    )

    # plots
    plt.figure(figsize=(6,6))
    plt.plot(FPR, TPR, label=f"ROC (AUC={auc:.4f})")
    plt.plot([0,1],[0,1], "k--", alpha=0.4)
    plt.scatter([FPR[np.argmin(np.abs(FAR-FRR))]], [TPR[np.argmin(np.abs(FAR-FRR))]], s=20, c='r', label=f"EER≈{eer:.4e}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve")
    plt.legend(); plt.grid(True, ls=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "roc_curve.png"), dpi=180)
    plt.close()

    # histogram overlay for quick sanity
    plt.figure(figsize=(9,6))
    plt.hist(imps, bins=40, alpha=0.6, label="impostor")
    plt.hist(gens, bins=40, alpha=0.6, label="genuine")
    plt.legend(); plt.xlabel("similarity"); plt.ylabel("count"); plt.title("Similarity distributions (genuine vs impostor)")
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "hist_overlay.png"), dpi=180)
    plt.close()

    print(f"[OK] Wrote ROC/EER/AUC analysis to {args.out_dir}")


if __name__ == "__main__":
    main()

