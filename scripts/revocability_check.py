import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from zlib import crc32
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.biohashing_utils import cancellable_transform


def load_template(path: str, retries: int = 3, delay: float = 1.0):
    """Load template data with retries to absorb intermittent IO timeouts."""
    for attempt in range(retries):
        try:
            return np.load(path)
        except TimeoutError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_dir", default="output/templates", help="Path to plain templates")
    ap.add_argument("--n_keys", type=int, default=50, help="How many random keys per user")
    ap.add_argument("--out_csv", default="output/revocability_metrics.csv")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    template_files = [f for f in os.listdir(args.templates_dir) if f.endswith("_template.npy")]

    print(f"[INFO] Found {len(template_files)} templates. Generating {args.n_keys} rotated keys each...")

    for f in tqdm(template_files, desc="Users"):
        user = f.split("_")[0]
        vec = load_template(os.path.join(args.templates_dir, f))

        # use mean if multiple trials
        if vec.ndim == 2:
            vec = vec.mean(axis=0)

        # base key (deterministic)
        base_key = crc32(user.encode("utf-8")) & 0xFFFFFFFF
        base_ct = cancellable_transform(vec, base_key, return_real=True)

        # rotated keys
        for i in range(args.n_keys):
            rand_key = np.random.randint(0, 2**32)
            new_ct = cancellable_transform(vec, rand_key, return_real=True)

            cos = np.dot(base_ct, new_ct) / (np.linalg.norm(base_ct) * np.linalg.norm(new_ct) + 1e-8)
            rows.append({"user": user, "rot_id": i, "similarity": cos})

    # Row-level metrics
    df = pd.DataFrame(rows)
    df.sort_values(["user", "rot_id", "similarity"], inplace=True)
    df.to_csv(args.out_csv, index=False)

    # Overall stats
    sims = df["similarity"].astype(float).values
    overall = {
        "n": int(len(sims)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "p05": float(np.percentile(sims, 5)),
        "median": float(np.median(sims)),
        "p95": float(np.percentile(sims, 95)),
        "max": float(np.max(sims)),
    }
    pd.DataFrame([overall]).to_csv(os.path.join(out_dir, "revocability_overall_stats.csv"), index=False)

    # Per-user stats
    g = df.groupby("user")["similarity"].agg(["count","mean","std","min","median","max"]).reset_index()
    g.rename(columns={"count":"n"}, inplace=True)
    g.to_csv(os.path.join(out_dir, "revocability_per_user_stats.csv"), index=False)

    # Overall histogram
    plt.figure(figsize=(9,6))
    plt.hist(sims, bins=30)
    plt.xlabel("similarity"); plt.ylabel("count")
    plt.title("Revocability similarity distribution (base vs random keys)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "revocability_overall_hist.png"), dpi=180)
    plt.close()

    print(f"[OK] Saved rows → {args.out_csv}")
    print(f"[OK] Stats and overall hist → {out_dir}")

if __name__ == "__main__":
    main()
