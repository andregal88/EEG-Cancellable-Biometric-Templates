# Fit per-feature mean/std on TRAIN split and save to JSON.
import os, glob, json, argparse
import numpy as np, pandas as pd

def _iter_feature_csvs(features_dir: str):
    files = sorted(glob.glob(os.path.join(features_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No feature CSVs in {features_dir}")
    return files

def _deterministic_split_mask(n: int, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(round(n * (1.0 - ratio)))
    mask = np.zeros(n, dtype=bool)
    mask[idx[:cut]] = True  # train=True
    return mask

def _train_mask_per_subject(df: pd.DataFrame, ratio: float = 0.2, seed: int = 42) -> pd.Series:
    if "split" in df.columns:
        return (df["split"].astype(str) == "train")
    mask = np.zeros(len(df), dtype=bool)
    for s, g in df.groupby("subject", sort=False):
        sd = abs(hash((s, seed))) % (2**31)
        m = _deterministic_split_mask(len(g), ratio, seed=sd)
        mask[g.index] = m
    return pd.Series(mask, index=df.index)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", default="output/features")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--out", default="output/templates/feat_norm.json")
    args = ap.parse_args()

    files = _iter_feature_csvs(args.features_dir)
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, axis=0, ignore_index=True)

    meta = ["subject", "run", "trial_idx"]
    feats = [c for c in df.columns if c not in meta]
    train_mask = _train_mask_per_subject(df, ratio=args.val_ratio, seed=42)
    X = df.loc[train_mask, feats].to_numpy(np.float64)

    mu = X.mean(axis=0).tolist()
    sd = X.std(axis=0).tolist()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"features": feats, "mean": mu, "std": sd}, f)
    print(f"[OK] Wrote stats → {args.out} (features={len(feats)})")

if __name__ == "__main__":
    main()