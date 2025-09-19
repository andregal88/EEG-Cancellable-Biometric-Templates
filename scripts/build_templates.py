import os
import glob
import pandas as pd
import numpy as np

FEATURES_DIR = "./output/features"
TEMPLATES_DIR = "./output/templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# === User names from the files ===
feature_files = glob.glob(os.path.join(FEATURES_DIR, "*_features.csv"))
users = sorted(set([os.path.basename(f).split("_")[0] for f in feature_files]))

for user in users:
    user_files = sorted([f for f in feature_files if os.path.basename(f).startswith(user + "_")])
    # Collect all trials
    dfs = [pd.read_csv(f) for f in user_files]
    df_all = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in df_all.columns if c not in ['subject', 'run', 'trial_idx']]
    # Shuffle for randomness
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    n_train = int(len(df_all) * 0.8)
    df_train = df_all.iloc[:n_train]
    df_test = df_all.iloc[n_train:]
    # Compute template (mean vector)
    template_vec = df_train[feature_cols].mean(axis=0).to_numpy()
    np.save(os.path.join(TEMPLATES_DIR, f"{user}_template.npy"), template_vec)
    # Save training/test index for later
    df_train.to_csv(os.path.join(TEMPLATES_DIR, f"{user}_train_trials.csv"), index=False)
    df_test.to_csv(os.path.join(TEMPLATES_DIR, f"{user}_test_trials.csv"), index=False)
    print(f"[✔] Template for {user} saved ({len(df_train)} train, {len(df_test)} test trials)")

print("Templates created for all users.")