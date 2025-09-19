# scripts/fuse_scores.py
# Fuse plain and real-cancellable cosine scores by probe/template alignment with normalization and safer EER computation.

import os, argparse, numpy as np
from sklearn.metrics import roc_curve, auc

def min_max_norm(x):
    x = np.asarray(x)
    mn, mx = np.min(x), np.max(x)
    if mx > mn:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)

def compute_eer(fpr, tpr, thresholds):
    fnr = 1 - tpr
    diff = fnr - fpr
    idxs = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(idxs) > 0:
        i = idxs[0]
        x0, x1 = fpr[i], fpr[i+1]
        y0, y1 = fnr[i], fnr[i+1]
        t0, t1 = thresholds[i], thresholds[i+1]
        denom = (x1 - x0) - (y1 - y0)
        if denom != 0:
            alpha = (y0 - x0) / denom
            eer = float(x0 + (x1 - x0) * alpha)
            thr_eer = float(t0 + (t1 - t0) * alpha)
        else:
            eer = float((x0 + y0) / 2.0)
            thr_eer = float(t0)
    else:
        idx_eer = int(np.argmin(np.abs(fnr - fpr)))
        eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
        thr_eer = float(thresholds[idx_eer]) if idx_eer < len(thresholds) else float(thresholds[-1])
    return eer, thr_eer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="path to plain scores.npz")
    ap.add_argument("--realcanc", required=True, help="path to real-cancellable scores.npz")
    ap.add_argument("--alpha", type=float, default=0.7, help="weight for plain")
    ap.add_argument("--out", required=True, help="output fused npz path")
    args = ap.parse_args()

    A = np.load(args.plain, allow_pickle=True)
    B = np.load(args.realcanc, allow_pickle=True)

    need = {"probe_ids", "templ_subjects", "y_true", "y_scores"}
    for name, pack in (("plain", A), ("realcanc", B)):
        miss = need - set(pack.files)
        if miss:
            raise ValueError(f"{name} scores missing keys: {sorted(miss)}")

    keyA = np.char.add(A["probe_ids"].astype(str), np.char.add("|", A["templ_subjects"].astype(str)))
    keyB = np.char.add(B["probe_ids"].astype(str), np.char.add("|", B["templ_subjects"].astype(str)))

    idxA = {k: i for i, k in enumerate(keyA)}
    rows = [(i, j) for j, k in enumerate(keyB) if (i := idxA.get(k)) is not None]
    if not rows:
        raise ValueError("No overlapping pairs between plain and real-cancellable scores.")

    iA, iB = zip(*rows)
    y_true_A = A["y_true"][list(iA)]
    y_true_B = B["y_true"][list(iB)]
    if not np.array_equal(y_true_A, y_true_B):
        raise ValueError("y_true arrays do not match between datasets.")
    y_true = y_true_A

    s_plain = min_max_norm(A["y_scores"][list(iA)])
    s_realc = min_max_norm(B["y_scores"][list(iB)])
    s_fused = args.alpha * s_plain + (1 - args.alpha) * s_realc

    fpr, tpr, thr = roc_curve(y_true, s_fused)
    roc_auc = float(auc(fpr, tpr))
    eer, thr_eer = compute_eer(fpr, tpr, thr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        y_true=y_true, y_scores=s_fused, fpr=fpr, tpr=tpr, thresholds=thr,
        roc_auc=roc_auc, eer=eer, thr_eer=thr_eer,
        probe_ids=A["probe_ids"][list(iA)], templ_subjects=A["templ_subjects"][list(iA)],
        alpha=args.alpha, mode="fused", distance="cosine"
    )
    print(f"[OK] Fused saved → {args.out} | AUC={roc_auc:.3f} EER={eer:.3f}")

if __name__ == "__main__":
    main()