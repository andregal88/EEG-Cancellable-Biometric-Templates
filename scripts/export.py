import pandas as pd
import os

def export_raw_to_csv(raw, subject_id, run_id, output_dir, suffix=""):
    """
    Exports raw data to CSV file with channel data and time.
    """
    data = raw.get_data()
    ch_names = raw.ch_names
    times = raw.times
    df = pd.DataFrame(data.T, columns=ch_names)
    df['time_sec'] = times
    filename = f"S{subject_id:03d}_R{run_id:02d}{suffix}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[✔] Exported to: {filepath}")