"""Transform raw user templates into cancellable templates and classic biohashes."""
from __future__ import annotations
import os
import numpy as np
from zlib import crc32

from scripts.biohashing_utils import (
    biohash_template,
    random_projection_template,
    cancellable_transform,
)

TEMPLATES_DIR = "output/templates"
OUT_DIR = "output/cancellable_templates"
os.makedirs(OUT_DIR, exist_ok=True)

# Defaults (tune here)
DIM_RATIO = 0.5
BITS = 8
RP_SCHEME = "gaussian"
AGGREGATE = "mean"   # mean/median
QUANT_MODE = "uniform"  # uniform | median-binary | balanced

template_files = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith("_template.npy")]
users = sorted({f.split("_")[0] for f in template_files})
if not users:
    raise SystemExit(f"No *_template.npy files under {TEMPLATES_DIR}")

for i, u in enumerate(users, 1):
    key = crc32(u.encode("utf-8")) & 0xFFFFFFFF
    path = os.path.join(TEMPLATES_DIR, f"{u}_template.npy")
    arr = np.load(path)

    vec = arr.mean(axis=0) if arr.ndim == 2 else arr

    bh = biohash_template(vec, key, rp_scheme=RP_SCHEME)
    rp = random_projection_template(vec, key, rp_scheme=RP_SCHEME)

    ct_gray = cancellable_transform(arr, key, dim_ratio=DIM_RATIO, bits=BITS,
                                    rp_scheme=RP_SCHEME, aggregate=AGGREGATE,
                                    return_real=False, quant_mode=QUANT_MODE)
    ct_real = cancellable_transform(arr, key, dim_ratio=DIM_RATIO, bits=BITS,
                                    rp_scheme=RP_SCHEME, aggregate=AGGREGATE,
                                    return_real=True, quant_mode=QUANT_MODE)

    np.save(os.path.join(OUT_DIR, f"{u}_biohash.npy"), bh)
    np.save(os.path.join(OUT_DIR, f"{u}_randproj.npy"), rp)
    np.save(os.path.join(OUT_DIR, f"{u}_ct_gray.npy"), ct_gray)
    np.save(os.path.join(OUT_DIR, f"{u}_ct_real.npy"), ct_real)

    if i % 10 == 0 or i == len(users):
        print(f"[{i}/{len(users)}] {u}: wrote cancellable + biohash/RP to {OUT_DIR} (quant_mode={QUANT_MODE})")