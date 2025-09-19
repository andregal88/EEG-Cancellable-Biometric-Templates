import os
import glob
import pandas as pd
from scripts.features import extract_all_features  # Make sure this import works!

# Increase maximum CSV field size (για πολύ μεγάλα αρχεία)
import sys
csv_field_size_limit = sys.maxsize
while True:
    # On some systems maxsize may be too large, so we try reducing if needed
    try:
        pd.options.display.max_rows = 200000  # Increase display limit (for debugging)
        pd.options.display.max_columns = 200  # Adjust as needed
        pd.set_option('display.max_colwidth', 200)
        break
    except OverflowError:
        csv_field_size_limit = int(csv_field_size_limit / 10)

# Set paths
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../output/preprocessing")
FEATURES_DIR = os.path.join(os.path.dirname(__file__), "../output/features")
os.makedirs(FEATURES_DIR, exist_ok=True)
csv_files = glob.glob(os.path.join(PREPROCESSED_DIR, "*.csv"))

def process_csv(csv_path, sf=128):
    filename = os.path.basename(csv_path)
    subj, run = filename.split("_")[:2]
    features_file = os.path.join(FEATURES_DIR, f"{subj}_{run}_features.csv")

    # Skip if features already exist (and non-empty)
    if os.path.exists(features_file) and os.path.getsize(features_file) > 0:
        print(f"[✓] Features already extracted for {filename}, skipping.")
        return

    try:
        print(f"[•] Reading file: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False, engine='c', encoding_errors='replace')
    except Exception as e:
        print(f"[!] 'c' engine failed, trying 'python' engine: {e}")
        try:
            df = pd.read_csv(csv_path, low_memory=False, engine='python', encoding_errors='replace')
        except Exception as e2:
            print(f"[!] Skipping {filename} -- Could not read CSV with any engine: {e2}")
            return

    channels = [col for col in df.columns if col not in ['label', 'subject', 'run', 'trial_idx']]
    results = []

    # Group by trial
    for (subject, run, trial_idx), group in df.groupby(['subject', 'run', 'trial_idx']):
        X = group[channels].to_numpy().T  # shape: (n_channels, n_samples)
        try:
            features = extract_all_features(X, sf=sf)
        except Exception as e:
            print(f"[!] Failed to extract features for {subject} {run} trial {trial_idx}: {e}")
            continue

        row = {
            'subject': subject,
            'run': run,
            'trial_idx': trial_idx
        }
        for key, value in features.items():
            # value is per-channel array
            if hasattr(value, 'shape') and value.shape[0] == len(channels):
                for ch_idx, ch_name in enumerate(channels):
                    row[f"{key}_{ch_name}"] = value[ch_idx]
            else:
                row[key] = value
        results.append(row)

    if not results:
        print(f"[!] No features extracted for {filename}")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(features_file, index=False)
    print(f"[✓] Saved extracted features to {features_file}")

if __name__ == '__main__':
    for csv_file in csv_files:
        process_csv(csv_file, sf=128)