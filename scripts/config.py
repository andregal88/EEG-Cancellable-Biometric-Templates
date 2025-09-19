import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '../data')
OUTPUT_PREPROC_PATH = os.path.join(os.path.dirname(__file__), '../output/preprocessing')
OUTPUT_FEATURES_PATH = os.path.join(os.path.dirname(__file__), '../output/features')
SAMPLING_RATE = 160

EVENT_ID_MAP = {
    "T0": 0,
    "T1": 1,
    "T2": 2
}