import os, argparse, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to impostor_correlation.csv")
    ap.add_argument("--out_dir", default="output/bruteforce_analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    if "type" not in df.columns or "similarity" not in df.columns:
        raise ValueError("CSV must include 'type' and 'similarity' columns.")

    imp = df[df["type"]=="impostor"]["similarity"].astype(float).values
    gen = df[df["type"]=="genuine"]["similarity"].astype(float).values

    # richer stats
    rows = []
    for label, arr in tqdm([("impostor", imp), ("genuine", gen)], desc="Compute stats"):
        if len(arr) == 0:
            continue
        rows.append({
            "type": label,
            "n": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p05": float(np.percentile(arr, 5)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "bruteforce_stats.csv"), index=False)

    # per-type histograms
    for label, arr in tqdm([("impostor", imp), ("genuine", gen)], desc="Per-type plots"):
        if len(arr) == 0:
            continue
        plt.figure(figsize=(8,5))
        plt.hist(arr, bins=30)
        plt.xlabel("similarity"); plt.ylabel("count")
        plt.title(f"{label.capitalize()} similarity distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{label}_hist.png"), dpi=160)
        plt.close()

    # overlay
    if len(imp) and len(gen):
        plt.figure(figsize=(9,6))
        plt.hist(imp, bins=30, alpha=0.6, label="impostor")
        plt.hist(gen, bins=30, alpha=0.6, label="genuine")
        plt.legend(); plt.xlabel("similarity"); plt.ylabel("count")
        plt.title("Genuine vs Impostor Similarity Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "bruteforce_overlay.png"), dpi=180)
        plt.close()

    print(f"[OK] Wrote brute-force proxy results to {args.out_dir}")

if __name__ == "__main__":
    main()
