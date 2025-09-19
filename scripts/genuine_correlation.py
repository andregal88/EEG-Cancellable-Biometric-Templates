"""Compute genuine (within-user, cross-trial) similarity distribution.

For each user template file (Sxxx_template.npy) under --templates_dir, this script:
- Applies the same cancellable transform as other scripts per trial using a fixed key
- Computes cosine similarity for trial pairs within the same user
- Optionally subsamples pairs per user via --max_pairs_per_user
- Writes a CSV with columns: user_a, user_b, type, similarity, trial_i, trial_j

Usage:
  python3 -m scripts.genuine_correlation \
    --templates_dir output/templates \
    --out_csv output/genuine_correlation.csv \
    [--max_pairs_per_user 2000]
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from zlib import crc32

from scripts.biohashing_utils import cancellable_transform


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_dir", default="output/templates", help="Path to plain templates")
    ap.add_argument("--out_csv", default="output/genuine_correlation.csv")
    ap.add_argument("--max_pairs_per_user", type=int, default=2000,
                    help="Subsample at most this many trial pairs per user (<=0 to use all)")
    ap.add_argument("--dim_ratio", type=float, default=0.5)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--rp_scheme", default="gaussian")
    ap.add_argument("--quant_mode", default="uniform")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    files = [f for f in os.listdir(args.templates_dir) if f.endswith("_template.npy")]
    rows = []

    rng = np.random.default_rng(42)
    print(f"[INFO] Computing within-user genuine similarities for {len(files)} users...")

    for f in tqdm(sorted(files), desc="Users"):
        user = f.split("_")[0]
        # Try loading per-trial data from template NPY first
        arr = np.load(os.path.join(args.templates_dir, f))
        if not (arr.ndim == 2 and arr.shape[0] >= 2):
            # Fallback: attempt to load train/test trials CSVs
            train_csv = os.path.join(args.templates_dir, f.replace("_template.npy", "_train_trials.csv"))
            test_csv = os.path.join(args.templates_dir, f.replace("_template.npy", "_test_trials.csv"))
            frames = []
            for p in [train_csv, test_csv]:
                if os.path.exists(p):
                    try:
                        df_trials = pd.read_csv(p)
                        # keep only numeric columns
                        df_trials = df_trials.select_dtypes(include=["number"])
                        if len(df_trials) > 0 and df_trials.shape[1] > 0:
                            frames.append(df_trials)
                    except Exception:
                        pass
            if frames:
                df_all = pd.concat(frames, axis=0, ignore_index=True)
                if len(df_all) >= 2:
                    arr = df_all.values.astype(np.float32)
                else:
                    continue
            else:
                continue

        key = crc32(user.encode("utf-8")) & 0xFFFFFFFF

        # transform each trial independently; return real-valued [0,1]
        trials = []
        for i in range(arr.shape[0]):
            trials.append(
                cancellable_transform(
                    arr[i], key,
                    dim_ratio=args.dim_ratio, bits=args.bits,
                    rp_scheme=args.rp_scheme,
                    return_real=True, quant_mode=args.quant_mode
                )
            )
        trials = np.asarray(trials, dtype=np.float32)

        # generate trial pairs
        n = trials.shape[0]
        iu, ju = np.triu_indices(n, k=1)
        idx = np.column_stack((iu, ju))

        if args.max_pairs_per_user and args.max_pairs_per_user > 0 and idx.shape[0] > args.max_pairs_per_user:
            sel = rng.choice(idx.shape[0], size=args.max_pairs_per_user, replace=False)
            idx = idx[sel]

        for i, j in idx:
            sim = cosine(trials[i], trials[j])
            rows.append({
                "user_a": user,
                "user_b": user,
                "type": "genuine",
                "trial_i": int(i),
                "trial_j": int(j),
                "similarity": float(sim),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["user_a", "trial_i", "trial_j", "similarity"], inplace=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved genuine similarities → {args.out_csv}")


if __name__ == "__main__":
    main()
