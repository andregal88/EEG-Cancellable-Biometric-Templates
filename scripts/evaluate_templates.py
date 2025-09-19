# In-memory evaluation for EEG biometric templates (plain vs cancellable)
from __future__ import annotations
import os, glob, json, argparse
from typing import Dict, List
import numpy as np
import pandas as pd
from pathlib import Path
from zlib import crc32

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, det_curve

from .biohashing_utils import cancellable_transform

# ---------------- helpers ----------------

def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _subject_from_template_name(name: str) -> str:
    return Path(name).name.split("_")[0]

def _load_templates_plain(templates_dir: str) -> Dict[str, np.ndarray]:
    templates = {}
    for f in sorted(glob.glob(os.path.join(templates_dir, "S*_template.npy"))):
        s = _subject_from_template_name(f)
        templates[s] = np.load(f).astype(np.float32).ravel()
    if not templates:
        raise FileNotFoundError(f"No templates found in {templates_dir}")
    return templates

def _load_templates_cancellable(canc_dir: str, real: bool = False) -> Dict[str, np.ndarray]:
    templates = {}
    pat = "S*_ct_real.npy" if real else "S*_ct_gray.npy"
    for f in sorted(glob.glob(os.path.join(canc_dir, pat))):
        s = _subject_from_template_name(f)
        arr = np.load(f)
        templates[s] = arr.astype(np.float32).ravel() if real else arr.astype(np.uint8).ravel()
    if not templates:
        raise FileNotFoundError(f"No cancellable templates found in {canc_dir} ({pat})")
    return templates

def _iter_feature_csvs(features_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(features_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No feature CSVs found in {features_dir}")
    return files

def _read_features(files: List[str]) -> pd.DataFrame:
    frames = []
    needed = {"subject", "run", "trial_idx"}
    for fp in files:
        df = pd.read_csv(fp)
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{fp} missing columns: {sorted(missing)}")
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)

def _deterministic_split_mask(n: int, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(round(n * (1.0 - ratio)))
    mask = np.zeros(n, dtype=bool)
    mask[idx[:cut]] = True
    return mask

def _train_test_split_per_subject(df: pd.DataFrame, ratio: float = 0.2, seed: int = 42) -> pd.Series:
    if "split" in df.columns:
        return (df["split"].astype(str) == "train")
    mask = np.zeros(len(df), dtype=bool)
    for s, g in df.groupby("subject", sort=False):
        sd = abs(hash((s, seed))) % (2**31)
        m = _deterministic_split_mask(len(g), ratio, seed=sd)
        mask[g.index] = m
    return pd.Series(mask, index=df.index)

def _l2norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(x))
    if n < eps: return x
    return (x / n).astype(x.dtype)

def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    # με L2 σταθερά ON, αυτό ισοδυναμεί με dot
    a = _l2norm(a, eps); b = _l2norm(b, eps)
    return float(np.dot(a, b))

def _hamming_similarity(a_bits: np.ndarray, b_bits: np.ndarray) -> float:
    a = a_bits.ravel(); b = b_bits.ravel()
    if a.dtype == np.uint8 and ((a == 0) | (a == 1)).all() and ((b == 0) | (b == 1)).all():
        dist = (a != b).mean(); return 1.0 - float(dist)
    if a.dtype != b.dtype:
        raise ValueError("Bit vectors dtype mismatch")
    max_bits = 16 if a.dtype == np.uint16 else 8
    xor = np.bitwise_xor(a, b).astype(np.uint32)
    diffs = np.unpackbits(xor.view(np.uint8)).reshape(len(a), -1)[:, -max_bits:].sum(axis=1)
    dist = diffs.mean() / max_bits
    return 1.0 - float(dist)

# -------------- Z-score loader --------------

def _load_norm_stats(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = data["features"]; mu = np.array(data["mean"], dtype=np.float32); sd = np.array(data["std"], dtype=np.float32)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return feats, mu, sd

def _apply_zscore(vec: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((vec - mu) / sd).astype(np.float32)

# ---------------- main eval ----------------

def evaluate(
    mode: str,
    templates_dir: str,
    cancellable_dir: str,
    features_dir: str,
    output_dir: str,
    distance: str,
    bits: int,
    dim_ratio: float,
    rp_scheme: str,
    seed: int,
    real_cancellable: bool,
    user_specific: bool,
    val_ratio: float,
    neg_per_probe: int | None,
    limit_probes: int | None,
    apply_norm: bool,
    norm_stats: str | None,
    quant_mode: str,
) -> None:
    _ensure_dir(output_dir)
    scores_path = os.path.join(output_dir, "scores.npz")
    metrics_csv = os.path.join(output_dir, "metrics.csv")

    # Load templates
    if mode == "plain":
        templates = _load_templates_plain(templates_dir)
    elif mode == "cancellable":
        templates = _load_templates_cancellable(cancellable_dir, real=real_cancellable)
    else:
        raise ValueError("mode must be 'plain' or 'cancellable'")
    subjects = sorted(templates.keys())

    rng = np.random.RandomState(seed)

    # Load features & split
    files = _iter_feature_csvs(features_dir)
    df = _read_features(files)
    train_mask = _train_test_split_per_subject(df, ratio=val_ratio, seed=seed)
    test_df = df.loc[~train_mask].copy()

    # Z-score stats (optional)
    mu = sd = None
    feature_cols_all = [c for c in df.columns if c not in ["subject","run","trial_idx"]]
    if apply_norm:
        path = norm_stats or "output/templates/feat_norm.json"
        feat_order, mu, sd = _load_norm_stats(path)
        # βεβαιώσου ότι η σειρά ταιριάζει:
        if feature_cols_all != feat_order:
            # αναδιάταξη
            df = df[["subject","run","trial_idx"] + feat_order]
            test_df = df.loc[~train_mask].copy()
            feature_cols_all = feat_order

    if limit_probes is not None and limit_probes > 0:
        test_df = test_df.iloc[:int(limit_probes)].copy()

    meta_cols = ["subject", "run", "trial_idx"]
    feature_cols = [c for c in test_df.columns if c not in meta_cols]

    # L2-normalize templates for cosine
    if distance == "cosine":
        for k in list(templates.keys()):
            templates[k] = _l2norm(templates[k].astype(np.float32))

    # scorer
    if distance == "cosine":
        def score_fn(probe_vec: np.ndarray, templ_vec: np.ndarray) -> float:
            return _cosine_similarity(probe_vec, templ_vec)
    elif distance == "hamming":
        def score_fn(probe_vec: np.ndarray, templ_vec: np.ndarray) -> float:
            return _hamming_similarity(probe_vec, templ_vec)
    else:
        raise ValueError("distance must be 'cosine' or 'hamming'")

    def transform_probe_with_key(row_vec: np.ndarray, key: int) -> np.ndarray:
        # Z-score πριν από cancellable pipeline
        x = row_vec
        if apply_norm and (mu is not None):
            x = _apply_zscore(x, mu, sd)
        if real_cancellable:
            y01 = cancellable_transform(x, key, dim_ratio=dim_ratio, rp_scheme=rp_scheme, bits=bits, return_real=True, quant_mode=quant_mode)
            return y01.astype(np.float32)
        else:
            g = cancellable_transform(x, key, dim_ratio=dim_ratio, rp_scheme=rp_scheme, bits=bits, return_real=False, quant_mode=quant_mode)
            return g.astype(np.uint8)

    _tx_cache: dict[tuple[str, int], np.ndarray] = {}

    # compute scores
    y_true: List[int] = []
    y_scores: List[float] = []
    probe_subjects: List[str] = []
    templ_subjects: List[str] = []
    probe_ids: List[str] = []

    probe_counter = 0
    for _, row in test_df.iterrows():
        subj_probe = str(row["subject"])
        raw_vec = row[feature_cols].values.astype(np.float32)
        if apply_norm and (mu is not None) and mode == "plain":
            raw_vec = _apply_zscore(raw_vec, mu, sd)

        # candidate templates
        cand_subjects = subjects
        probe_counter += 1
        if probe_counter % 200 == 0:
            print(f"[INFO] Probes processed: {probe_counter}")

        for s in cand_subjects:
            if mode == "plain":
                probe_vec = _l2norm(raw_vec) if distance == "cosine" else raw_vec
            else:
                key_s = crc32(s.encode("utf-8")) & 0xFFFFFFFF
                pid = f"{row['subject']}_{row['run']}_{row['trial_idx']}"
                cache_key = (pid, key_s)
                if cache_key in _tx_cache:
                    probe_vec = _tx_cache[cache_key]
                else:
                    probe_vec = transform_probe_with_key(raw_vec, key_s)
                    _tx_cache[cache_key] = probe_vec
                if distance == "cosine" and probe_vec.dtype != np.uint8:
                    probe_vec = _l2norm(probe_vec)

            templ = templates[s]
            score = score_fn(probe_vec, templ)
            y_scores.append(float(score))
            y_true.append(1 if s == subj_probe else 0)
            probe_subjects.append(subj_probe)
            templ_subjects.append(s)
            probe_ids.append(f"{row['subject']}_{row['run']}_{row['trial_idx']}")

    y_true = np.array(y_true, dtype=np.int8)
    y_scores = np.array(y_scores, dtype=np.float32)

    fpr, tpr, thr = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1 - tpr
    idx_eer = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx_eer] + fnr[idx_eer]) / 2.0)
    thr_eer = float(thr[idx_eer]) if idx_eer < len(thr) else float(thr[-1])

    np.savez_compressed(
        scores_path,
        y_true=y_true, y_scores=y_scores,
        fpr=fpr, tpr=tpr, thresholds=thr,
        roc_auc=roc_auc, eer=eer, thr_eer=thr_eer,
        probe_subjects=np.array(probe_subjects, dtype=object),
        templ_subjects=np.array(templ_subjects, dtype=object),
        probe_ids=np.array(probe_ids, dtype=object),
        mode=mode, distance=distance,
        bits=bits, dim_ratio=dim_ratio,
        real_cancellable=real_cancellable,
        apply_norm=apply_norm, quant_mode=quant_mode,
    )

    pd.DataFrame([{
        "mode": mode, "distance": distance,
        "bits": bits, "dim_ratio": dim_ratio,
        "roc_auc": roc_auc, "eer": eer,
        "thr_eer": thr_eer, "n_pairs": int(len(y_true)),
    }]).to_csv(metrics_csv, index=False)

    roc_png = os.path.join(output_dir, "roc.png")
    det_png = os.path.join(output_dir, "det.png")

    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {mode} ({distance})")
    plt.legend(loc="lower right"); plt.savefig(roc_png, dpi=180, bbox_inches="tight"); plt.close()

    det_fpr, det_fnr, _ = det_curve(y_true, y_scores)
    plt.figure(); plt.plot(det_fpr, det_fnr)
    plt.xlabel("FPR"); plt.ylabel("FNR"); plt.title(f"DET — {mode} ({distance})")
    plt.savefig(det_png, dpi=180, bbox_inches="tight"); plt.close()

    print(f"[OK] Saved: {scores_path}, {metrics_csv}, {roc_png}, {det_png}")
    print(f"[INFO] EER={eer:.4f} @ thr≈{thr_eer:.4f} | AUC={roc_auc:.3f} | pairs={len(y_true)}")

def main():
    p = argparse.ArgumentParser(description="In-memory evaluation for EEG biometric templates.")
    p.add_argument("--mode", choices=["plain", "cancellable"], default="plain")
    p.add_argument("--templates_dir", default="output/templates")
    p.add_argument("--cancellable_dir", default="output/cancellable_templates")
    p.add_argument("--features_dir", default="output/features")
    p.add_argument("--output_dir", default="output/eval")
    p.add_argument("--distance", choices=["cosine", "hamming"], default=None)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--dim_ratio", type=float, default=0.5)
    p.add_argument("--rp_scheme", choices=["gaussian", "rademacher"], default="gaussian")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real_cancellable", action="store_true")
    p.add_argument("--user_specific", action="store_true")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--neg_per_probe", type=int, default=None)
    p.add_argument("--limit_probes", type=int, default=None)
    # NEW:
    p.add_argument("--apply_norm", action="store_true", help="Apply z-score using stats file")
    p.add_argument("--norm_stats", default=None, help="Path to feat_norm.json (from fit_feature_norm.py)")
    p.add_argument("--quant_mode", default="uniform", help="uniform | median-binary | balanced (for cancellable)")
    args = p.parse_args()

    if args.distance is None:
        args.distance = "cosine" if (args.mode == "plain" or args.real_cancellable) else "hamming"

    evaluate(
        mode=args.mode,
        templates_dir=args.templates_dir,
        cancellable_dir=args.cancellable_dir,
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        distance=args.distance,
        bits=args.bits,
        dim_ratio=args.dim_ratio,
        rp_scheme=args.rp_scheme,
        seed=args.seed,
        real_cancellable=args.real_cancellable,
        user_specific=args.user_specific,
        val_ratio=args.val_ratio,
        neg_per_probe=args.neg_per_probe,
        limit_probes=args.limit_probes,
        apply_norm=args.apply_norm,
        norm_stats=args.norm_stats,
        quant_mode=args.quant_mode,
    )

if __name__ == "__main__":
    main()