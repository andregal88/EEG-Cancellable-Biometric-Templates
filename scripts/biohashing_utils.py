# numpy-only cancellable/biohash utilities
# Pipeline: Permutation -> Masking -> Random Projection -> Z-score ->
#                        Gaussian CDF -> Quantization -> Gray coding

from __future__ import annotations
import numpy as np
from math import erf

# ---------- RNG helpers ----------

def _rng_from_key(key: int) -> np.random.Generator:
    return np.random.default_rng(np.uint32(key))

def _subkeys(key: int, n: int) -> list[int]:
    g = _rng_from_key(key)
    return [int(v) for v in g.integers(0, 2**32, size=n, dtype=np.uint32)]

# ---------- math helpers ----------

def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    return (x - mu) / (sd + eps)

def gaussian_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.float32(np.sqrt(2.0))))

def _quantize_uniform(x01: np.ndarray, bits: int) -> np.ndarray:
    levels = (1 << bits) - 1
    q = np.rint(np.clip(x01, 0.0, 1.0) * levels)
    if bits <= 8:
        return q.astype(np.uint8)
    elif bits <= 16:
        return q.astype(np.uint16)
    else:
        return q.astype(np.uint32)

def _gray_encode(q: np.ndarray) -> np.ndarray:
    if not np.issubdtype(q.dtype, np.unsignedinteger):
        q = q.astype(np.uint32)
    return (q ^ (q >> 1)).astype(q.dtype)

# ---------- random projection helpers ----------

def generate_random_projection_matrix(dim_in: int, dim_out: int, key: int,
                                      scheme: str = "gaussian") -> np.ndarray:
    g = _rng_from_key(key)
    if scheme == "gaussian":
        R = g.standard_normal(size=(dim_in, dim_out), dtype=np.float32)
        R *= (1.0 / np.sqrt(max(dim_out, 1)))
        return R
    elif scheme == "rademacher":
        R = g.choice([-1.0, 1.0], size=(dim_in, dim_out)).astype(np.float32)
        R *= (1.0 / np.sqrt(max(dim_out, 1)))
        return R
    else:
        raise ValueError(f"Unknown rp scheme: {scheme}")

# ---------- classic biohash / RP ----------

def biohash_template(vec: np.ndarray, key: int, dim_out: int | None = None,
                     rp_scheme: str = "gaussian") -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    d_in = v.size
    d_out = d_in if dim_out is None else int(dim_out)
    R = generate_random_projection_matrix(d_in, d_out, key, scheme=rp_scheme)
    proj = v @ R
    return (proj > 0).astype(np.uint8)

def random_projection_template(vec: np.ndarray, key: int, dim_out: int | None = None,
                               rp_scheme: str = "gaussian") -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    d_in = v.size
    d_out = d_in if dim_out is None else int(dim_out)
    R = generate_random_projection_matrix(d_in, d_out, key, scheme=rp_scheme)
    return (v @ R).astype(np.float32)

# ---------- cancellable transform (paper-like) ----------

def _permute(v: np.ndarray, key: int) -> np.ndarray:
    g = _rng_from_key(key)
    idx = g.permutation(v.size)
    return v[idx]

def _mask(v: np.ndarray, key: int, keep_ratio: float) -> np.ndarray:
    g = _rng_from_key(key)
    d = v.size
    m = max(1, int(round(d * keep_ratio)))
    idx = g.choice(d, size=m, replace=False)
    return v[idx]

def cancellable_transform_once(arr: np.ndarray, key: int,
                               dim_ratio: float = 0.5, bits: int = 8,
                               rp_scheme: str = "gaussian",
                               return_real: bool = False,
                               quant_mode: str = "uniform") -> np.ndarray:
    """
    Steps: permutation -> masking -> RP -> z-score -> Gaussian CDF -> quantize (mode) -> Gray.
    If return_real=True, returns [0,1] float (before quantization).
    """
    v = np.asarray(arr, dtype=np.float32).ravel()
    k_perm, k_mask, k_rp, k_dith = _subkeys(key, 4)

    # 1) permutation + 2) masking
    v = _permute(v, k_perm)
    v = _mask(v, k_mask, keep_ratio=dim_ratio)

    # 3) Random projection (square to masked dim)
    d_in = v.size
    R = generate_random_projection_matrix(d_in, d_in, k_rp, scheme=rp_scheme)
    y = (v @ R).astype(np.float32)

    # 4) normalize + 5) map to [0,1]
    y = _zscore(y)
    y01 = gaussian_cdf(y)

    if return_real:
        return y01.astype(np.float32)

    # 6) quantization modes
    qm = str(quant_mode or "uniform").lower()
    if qm == "median-binary":
        # 1-bit per dim around 0.5 after CDF
        g = (y01 >= 0.5).astype(np.uint8)
        return g
    elif qm == "balanced":
        # clip + small deterministic dithering + uniform quantization (default 8-bit)
        yq = np.clip(y01, 0.0, 1.0).astype(np.float32)
        rng = _rng_from_key(k_dith)
        noise = (rng.random(yq.shape, dtype=np.float32) - 0.5) * (1.0 / max(2**bits, 256))
        yq = np.clip(yq + noise, 0.0, 1.0)
        q = _quantize_uniform(yq, bits=bits)
        g = _gray_encode(q)
        return g
    else:
        # default: uniform → Gray
        q = _quantize_uniform(y01, bits=bits)
        g = _gray_encode(q)
        return g

def cancellable_transform(arr: np.ndarray, key: int,
                          dim_ratio: float = 0.5, bits: int = 8,
                          rp_scheme: str = "gaussian",
                          aggregate: str = "mean",
                          return_real: bool = False,
                          quant_mode: str = "uniform") -> np.ndarray:
    """
    Apply transform to 1D or 2D (trials x features). 2D is aggregated first.
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        v = np.median(a, axis=0) if aggregate == "median" else np.mean(a, axis=0)
    elif a.ndim == 1:
        v = a
    else:
        raise ValueError(f"Unsupported arr.ndim={a.ndim}")

    return cancellable_transform_once(
        v, key, dim_ratio=dim_ratio, bits=bits, rp_scheme=rp_scheme,
        return_real=return_real, quant_mode=quant_mode
    )