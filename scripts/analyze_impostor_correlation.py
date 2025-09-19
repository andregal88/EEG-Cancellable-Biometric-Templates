import os, argparse, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to impostor_correlation.csv")
    ap.add_argument("--out_dir", default="output/impostor_analysis")
    ap.add_argument("--per_user_max", type=int, default=16, help="how many per-user hist plots to save")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # find similarity column
    sim_col = None
    for cand in ["cosine","corr","correlation","similarity","sim"]:
        if cand in df.columns:
            sim_col = cand
            break
    if sim_col is None:
        raise ValueError("No similarity column found in CSV.")

    user_col = "user_a" if "user_a" in df.columns else "user"

    # --- overall stats
    x = df[sim_col].values.astype(float)
    overall = {
        "n": int(len(x)), "mean": float(np.mean(x)), "std": float(np.std(x)),
        "min": float(np.min(x)), "p05": float(np.percentile(x,5)),
        "median": float(np.median(x)), "p95": float(np.percentile(x,95)), "max": float(np.max(x)),
    }
    pd.DataFrame([overall]).to_csv(os.path.join(args.out_dir, "overall_stats.csv"), index=False)

    # overall hist
    plt.figure(figsize=(9,6))
    plt.hist(x, bins=30)
    plt.xlabel("similarity"); plt.ylabel("count")
    plt.title("Similarity distribution (overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "overall_hist.png"), dpi=180)
    plt.close()

    # If 'type' column exists, produce separate + overlay stats/plots
    if 'type' in df.columns:
        out = []
        for t in ['impostor','genuine']:
            xt = df[df['type']==t][sim_col].astype(float).values
            if len(xt) == 0:
                continue
            out.append({
                'type': t,
                'n': int(len(xt)), 'mean': float(np.mean(xt)), 'std': float(np.std(xt)),
                'min': float(np.min(xt)), 'p05': float(np.percentile(xt,5)),
                'median': float(np.median(xt)), 'p95': float(np.percentile(xt,95)), 'max': float(np.max(xt)),
            })
        if out:
            pd.DataFrame(out).to_csv(os.path.join(args.out_dir, 'by_type_stats.csv'), index=False)

        imp = df[df['type']=='impostor'][sim_col].astype(float).values
        gen = df[df['type']=='genuine'][sim_col].astype(float).values
        if len(imp) and len(gen):
            plt.figure(figsize=(9,6))
            plt.hist(imp, bins=30, alpha=0.6, label='impostor')
            plt.hist(gen, bins=30, alpha=0.6, label='genuine')
            plt.legend(); plt.xlabel('similarity'); plt.ylabel('count')
            plt.title('Genuine vs Impostor Similarity Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, 'overlay_hist.png'), dpi=180)
            plt.close()

    # per-user stats
    stats = []
    per_users = df.groupby(user_col)
    # collect per-user stats
    for u, gsub in tqdm(per_users, desc="Per-user stats"):
        gx = gsub[sim_col].astype(float).values
        stats.append({
            "user": u, "n": int(len(gx)), "mean": float(np.mean(gx)),
            "std": float(np.std(gx)), "median": float(np.median(gx)),
            "min": float(np.min(gx)), "max": float(np.max(gx))
        })
    per_df = pd.DataFrame(stats)
    per_df.to_csv(os.path.join(args.out_dir, "per_user_stats.csv"), index=False)

    # per-user histograms (limited)
    to_plot = per_df.sort_values("mean")["user"].head(args.per_user_max).tolist()
    for u in tqdm(to_plot, desc="Per-user plots"):
        gx = df[df[user_col]==u][sim_col].astype(float).values
        plt.figure(figsize=(7,5))
        plt.hist(gx, bins=20)
        plt.xlabel("similarity"); plt.ylabel("count")
        plt.title(f"Similarity per user: {u}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"user_{u}.png"), dpi=150)
        plt.close()

    print(f"[OK] Wrote analysis to {args.out_dir}")

if __name__ == "__main__":
    main()
