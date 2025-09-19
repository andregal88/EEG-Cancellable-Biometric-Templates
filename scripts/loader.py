# scripts/loader.py

import pandas as pd

def load_events(event_fp):
    # Try UTF-8 first, then fallback to latin1 if decoding fails
    try:
        return pd.read_csv(event_fp, delimiter='|')
    except UnicodeDecodeError:
        return pd.read_csv(event_fp, delimiter='|', encoding='latin1')


# --- EDF loader ---
import mne

def load_edf(edf_fp):
    """
    Loads an EDF file and returns the raw MNE object.
    """
    return mne.io.read_raw_edf(edf_fp, preload=True)