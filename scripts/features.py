import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis

def bandpower(psd, freqs, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[:, idx], axis=1)

def extract_all_features(X, sf=128):
    """
    Υπολογίζει EEG features: band power, relative power, spectral entropy, skew, kurtosis, Hjorth.
    X: np.ndarray (n_channels, n_samples)
    """
    if X.ndim != 2:
        raise ValueError(f"Input X must be 2D array, got shape {X.shape}")

    features = {}
    features['mean'] = np.mean(X, axis=1)
    features['std'] = np.std(X, axis=1)
    features['var'] = np.var(X, axis=1)
    features['skew'] = skew(X, axis=1)
    features['kurtosis'] = kurtosis(X, axis=1)

    freqs, psd = welch(X, sf, axis=-1, nperseg=min(256, X.shape[-1]))
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    total_power = np.sum(psd, axis=1, keepdims=True)
    for band_name, band_range in bands.items():
        power = bandpower(psd, freqs, band_range)
        features[f'{band_name}_power'] = power
        features[f'{band_name}_rel_power'] = power / (total_power.squeeze() + 1e-12)
    psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
    features['spectral_entropy'] = entropy(psd_norm, axis=1)

    def hjorth_params(sig):
        first_deriv = np.diff(sig)
        second_deriv = np.diff(first_deriv)
        var_zero = np.var(sig)
        var_d1 = np.var(first_deriv)
        var_d2 = np.var(second_deriv)
        mobility = np.sqrt(var_d1 / var_zero) if var_zero != 0 else 0
        complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 != 0 and mobility != 0 else 0
        return mobility, complexity

    mobility = []
    complexity = []
    for ch in X:
        m, c = hjorth_params(ch)
        mobility.append(m)
        complexity.append(c)
    features['hjorth_mobility'] = np.array(mobility)
    features['hjorth_complexity'] = np.array(complexity)

    return features