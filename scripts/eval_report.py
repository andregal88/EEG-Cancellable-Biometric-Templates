# scripts/eval_report.py
import os, argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, auc

def _pick_thr_at_far(y_true, y_scores, far_target):
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    idx = int(np.argmin(np.abs(fpr - far_target)))
    thr_star = float(thr[idx if idx < len(thr) else -1])
    return thr_star, fpr, tpr, thr

def _eer_from_curve(fpr, tpr, thr):
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr_eer = float(thr[idx]) if idx < len(thr) else float(thr[-1])
    return eer, thr_eer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="test scores.npz")
    ap.add_argument("--val_scores", default=None, help="validation scores.npz (optional)")
    ap.add_argument("--far_targets", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    T = np.load(args.scores, allow_pickle=True)
    y_true_t = T["y_true"]; y_score_t = T["y_scores"]
    mode = str(T["mode"]) if "mode" in T.files else "N/A"
    distance = str(T["distance"]) if "distance" in T.files else "N/A"
    roc_auc_t = float(T["roc_auc"]) if "roc_auc" in T.files else None
    eer_t = float(T["eer"]) if "eer" in T.files else None

    rows = []
    for ft in args.far_targets:
        if args.val_scores:
            V = np.load(args.val_scores, allow_pickle=True)
            thr_star, _, _, _ = _pick_thr_at_far(V["y_true"], V["y_scores"], ft)
        else:
            thr_star, fpr_tmp, tpr_tmp, thr_tmp = _pick_thr_at_far(y_true_t, y_score_t, ft)
        # apply on test
        y_hat = (y_score_t >= thr_star).astype(int)
        far = float(((y_hat == 1) & (y_true_t == 0)).mean())
        frr = float(((y_hat == 0) & (y_true_t == 1)).mean())
        acc = float((y_hat == (y_true_t == 1)).mean())
        # recompute for display (AUC/EER)
        fpr, tpr, thr = roc_curve(y_true_t, y_score_t)
        auc_t = float(auc(fpr, tpr))
        eer_disp, _ = _eer_from_curve(fpr, tpr, thr)
        rows.append({
            "mode": mode, "distance": distance,
            "far_target": ft, "threshold": thr_star,
            "FAR": far, "FRR": frr, "ACC": acc,
            "AUC": auc_t, "EER": eer_disp,
            "n_pairs": int(len(y_true_t)),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()