import os
import re
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
import sys
import time

RAW_PATH = "./data"
OUTPUT_PREPROC_PATH = "./output/preprocessing"
EDF_SUFFIX = ".edf"
EVENT_SUFFIX = ".edf.event"
SAMPLING_RATE = 160

EVENT_LABELS = {"T0": 0, "T1": 1, "T2": 2}
BASELINE_RUNS = {"R01", "R02"}
BASELINE_EPOCH_SEC = 2.0
BASELINE_OVERLAP = 0.0

ICA_METHOD = "infomax"
ICA_NCOMP = 0.99
ICA_MAX_ITER = 300
RANDOM_STATE = 42

def normalize_channel_names(raw):
    raw.rename_channels(lambda ch: ch.strip().upper().replace('.', ''))

def parse_event_file(event_fp):
    events = []
    try:
        with open(event_fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        matches = re.findall(r'(T[012])\s+duration:\s+([\d\.]+)', txt)
        for ev, dur in matches:
            events.append((ev, float(dur)))
    except Exception as e:
        print(f"[!] Failed to parse {event_fp}: {e}")
    return events

def slice_fixed_epochs(raw, sf, epoch_sec, overlap, label, subject, run):
    n_samples = raw.n_times
    win = int(epoch_sec * sf)
    step = int(win * (1 - overlap)) if overlap < 1.0 else 1
    ch_names = raw.ch_names
    trials, idx = [], 0
    for start in range(0, n_samples - win + 1, step):
        end = start + win
        data = raw.get_data(start=start, stop=end)
        df = pd.DataFrame(data.T, columns=ch_names)
        df["label"] = label
        df["subject"] = subject
        df["run"] = run
        df["trial_idx"] = idx
        trials.append(df)
        idx += 1
    return trials

def crop_by_events(raw, events_info, sf, subject, run):
    trials, curr_sample, idx = [], 0, 0
    ch_names = raw.ch_names
    for ev_type, dur in events_info:
        label = EVENT_LABELS.get(ev_type, -1)
        ns = int(round(dur * sf))
        start, end = curr_sample, curr_sample + ns
        if end > raw.n_times: end = raw.n_times
        if end <= start: curr_sample = end; continue
        data = raw.get_data(start=start, stop=end)
        if data.shape[1] == 0: curr_sample = end; continue
        df = pd.DataFrame(data.T, columns=ch_names)
        df["label"] = label
        df["subject"] = subject
        df["run"] = run
        df["trial_idx"] = idx
        trials.append(df)
        curr_sample = end
        idx += 1
    return trials

def run_ica_and_clean(raw, run):
    try:
        if ICA_METHOD == "infomax":
            ica = ICA(method="infomax", n_components=ICA_NCOMP, fit_params=dict(extended=True),
                      random_state=RANDOM_STATE, max_iter=ICA_MAX_ITER)
        else:
            ica = ICA(method=ICA_METHOD, n_components=ICA_NCOMP,
                      random_state=RANDOM_STATE, max_iter=ICA_MAX_ITER)
        ica.fit(raw)
        labels = label_components(raw, ica, method='iclabel')
        bad_idx = [i for i, lab in enumerate(labels['labels'])
                   if lab in ('eye blink', 'heart beat', 'muscle artifact', 'other', 'channel noise')]
        ica.exclude = bad_idx
        print(f"    [ICA] Removed {len(bad_idx)} components: {bad_idx}")
        return ica.apply(raw.copy())
    except Exception as e:
        print(f"[!] ICA failed ({e}). Using un-cleaned signal.")
        return raw.copy()

def preprocess_subject_run(subject, run):
    subj_dir = os.path.join(RAW_PATH, subject)
    edf_fp = os.path.join(subj_dir, f"{subject}{run}{EDF_SUFFIX}")
    event_fp = os.path.join(subj_dir, f"{subject}{run}{EVENT_SUFFIX}")
    out_fp = os.path.join(OUTPUT_PREPROC_PATH, f"{subject}_{run}_epochs.csv")
    try:
        if os.path.exists(out_fp):
            print(f"[⏭] {subject}-{run} already exists."); return True
        if not os.path.exists(edf_fp):
            print(f"[!] Missing EDF: {edf_fp}"); return False
        if not os.path.exists(event_fp) and run not in BASELINE_RUNS:
            print(f"[!] Missing EVENT file for {subject}-{run}"); return False

        # ---- STEP: Start processing
        print(f"\n[•] Processing: {edf_fp} (run={run})")
        print(f"  > File size: {os.path.getsize(edf_fp)/1e6:.2f} MB")
        t0 = time.time()

        print("  > Loading EDF... ", end="", flush=True)
        t_load = time.time()
        raw = mne.io.read_raw_edf(edf_fp, preload=True, verbose=False)
        print(f"Done ({time.time()-t_load:.1f}s)")

        normalize_channel_names(raw)
        print("  > Setting montage...", end="", flush=True)
        t_step = time.time()
        raw.set_montage('standard_1020', match_case=False, on_missing='warn')
        print(f" Done ({time.time()-t_step:.1f}s)")

        print("  > Applying average reference projection...", end="", flush=True)
        t_step = time.time()
        raw.set_eeg_reference('average', projection=True)
        raw.apply_proj()
        print(f" Done ({time.time()-t_step:.1f}s)")

        print("  > Notch filtering (50Hz)...", end="", flush=True)
        t_step = time.time()
        raw.notch_filter(50)
        print(f" Done ({time.time()-t_step:.1f}s)")

        print("  > Bandpass filtering (1-40Hz)...", end="", flush=True)
        t_step = time.time()
        raw.filter(1, 40, fir_design="firwin")
        print(f" Done ({time.time()-t_step:.1f}s)")

        print("  > ICA (artifact removal)...", end="", flush=True)
        t_step = time.time()
        raw_clean = run_ica_and_clean(raw, run)
        print(f" Done ({time.time() - t_step:.1f}s)")

        sf = float(raw.info["sfreq"])

        print("  > Epoching into trials...", end="", flush=True)
        t_step = time.time()
        if run in BASELINE_RUNS:
            trials = slice_fixed_epochs(raw_clean, sf, BASELINE_EPOCH_SEC, BASELINE_OVERLAP,
                                       label=EVENT_LABELS["T0"], subject=subject, run=run)
        else:
            events_info = parse_event_file(event_fp)
            if not events_info:
                print(f"[!] No events found for {subject}-{run}."); return False
            trials = crop_by_events(raw_clean, events_info, sf, subject, run)
        if not trials:
            print(f"[!] No valid epochs for {subject}-{run}")
            return False
        print(f" Done ({time.time()-t_step:.1f}s)")

        out_df = pd.concat(trials, ignore_index=True)
        os.makedirs(OUTPUT_PREPROC_PATH, exist_ok=True)
        out_df.to_csv(out_fp, index=False)
        print(f"[✔] Saved {out_fp} ({len(out_df)} rows, {out_df['trial_idx'].nunique()} trials)")
        print(f"  > Finished {subject}-{run} in {time.time() - t0:.1f}s\n")
        return True
    except Exception as e:
        print(f"[X] ERROR for {subject}-{run}: {e}")
        return False

def main():
    subjects = sorted([f for f in os.listdir(RAW_PATH) if re.match(r"S\d+", f)])
    runs = [f"R{str(i).zfill(2)}" for i in range(1, 15)]
    total, ok, fail = 0, 0, 0
    start_time = time.time()
    try:
        for subject in subjects:
            print(f"\n==== Subject: {subject} ====")
            for run in runs:
                print(f"--> Run: {run}")
                total += 1
                res = preprocess_subject_run(subject, run)
                if res: ok += 1
                else: fail += 1
                sys.stdout.flush()
        print(f"\nDONE: {ok}/{total} files processed successfully. {fail} failed. Time: {time.time()-start_time:.1f}s")
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Exiting early.")
        print(f"\nDONE (partial): {ok}/{total} files processed successfully. {fail} failed. Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    main()