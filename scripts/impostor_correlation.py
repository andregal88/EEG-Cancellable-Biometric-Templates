import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates_dir", default="output/cancellable_templates", help="Path to cancellable templates")
    ap.add_argument("--out_csv", default="output/impostor_correlation.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    files = [f for f in os.listdir(args.templates_dir) if f.endswith("_ct_real.npy")]
    users = {f.split("_")[0]: np.load(os.path.join(args.templates_dir, f)) for f in files}

    rows = []
    user_list = list(users.keys())

    print(f"[INFO] Comparing {len(user_list)} users for impostor correlation...")

    for i, u in enumerate(tqdm(user_list, desc="Users")):
        u_vec = users[u].astype(np.float32).ravel()
        # genuine self-similarity
        rows.append({"user_a": u, "user_b": u, "type": "genuine", "similarity": cosine(u_vec, u_vec)})

        # impostor comparisons
        for v in user_list:
            if u == v: continue
            v_vec = users[v].astype(np.float32).ravel()
            sim = cosine(u_vec, v_vec)
            rows.append({"user_a": u, "user_b": v, "type": "impostor", "similarity": sim})

    df = pd.DataFrame(rows)
    # sort outputs in ascending order for readability
    df.sort_values(["user_a", "user_b", "type", "similarity"], inplace=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved → {args.out_csv}")

if __name__ == "__main__":
    main()
