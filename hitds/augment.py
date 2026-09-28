"""Class-imbalance correction: SMOTE (baseline) and CTGAN (proposal layer 1).

Both operate on the *preprocessed* matrix (values in [0, 1]) so synthetic rows
are always in-range. CTGAN output is validated before use (proposal §29: CTGAN
can produce plausible-looking but impossible flows).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .common import get_logger

log = get_logger("hitds.augment")


def _targets(y: np.ndarray, target_min: int) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    return {c: target_min - n for c, n in zip(classes, counts) if n < target_min}


def smote(X: np.ndarray, y: np.ndarray, target_min: int, k: int = 5, seed: int = 42):
    """Plain SMOTE: new = x + u * (neighbour - x). Implemented here so it runs without imbalanced-learn."""
    rng = np.random.default_rng(seed)
    need = _targets(y, target_min)
    new_X, new_y = [X], [y]
    for cls, n_new in need.items():
        Xc = X[y == cls]
        if len(Xc) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(Xc))).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, len(Xc), n_new)
        pick = neigh[base, rng.integers(0, neigh.shape[1], n_new)]
        u = rng.random((n_new, 1))
        new_X.append(Xc[base] + u * (Xc[pick] - Xc[base]))
        new_y.append(np.full(n_new, cls, dtype=y.dtype))
        log.info("SMOTE %-14s %6d -> %6d", cls, len(Xc), len(Xc) + n_new)
    return np.vstack(new_X).astype(np.float32), np.concatenate(new_y)


def validate_synthetic(X_syn: np.ndarray, X_real: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Keep synthetic rows that stay inside the real class's per-feature range (+/- tol).

    A crude but defensible protocol-constraint check: e.g. a synthetic flow may not have
    more SYN flags or a larger packet size than any real flow of that class ever had.
    """
    lo, hi = X_real.min(0) - tol, X_real.max(0) + tol
    ok = np.all((X_syn >= lo) & (X_syn <= hi), axis=1)
    return ok


def ctgan_augment(X: np.ndarray, y: np.ndarray, target_min: int, epochs: int = 150,
                  max_rows: int = 20000, seed: int = 42):
    """Conditional Tabular GAN on minority classes only (keeps CPU cost bounded)."""
    try:
        import pandas as pd
        from ctgan import CTGAN
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install ctgan (needs torch) - or set augment.method: smote") from e

    need = _targets(y, target_min)
    if not need:
        return X, y
    minority = np.isin(y, list(need))
    Xm, ym = X[minority], y[minority]
    if len(Xm) > max_rows:
        idx = np.random.default_rng(seed).choice(len(Xm), max_rows, replace=False)
        Xm, ym = Xm[idx], ym[idx]
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(Xm, columns=cols)
    df["label"] = ym
    log.info("training CTGAN on %d minority rows, %d classes, %d epochs (CPU: expect minutes)",
             len(df), len(need), epochs)
    model = CTGAN(epochs=epochs, verbose=True, cuda=False)
    model.fit(df, discrete_columns=["label"])

    new_X, new_y = [X], [y]
    for cls, n_new in need.items():
        real = X[y == cls]
        got, tries = [], 0
        while sum(len(g) for g in got) < n_new and tries < 10:
            tries += 1
            s = model.sample(n_new * 2, condition_column="label", condition_value=cls)
            s = s[s["label"] == cls][cols].to_numpy(dtype=np.float32)
            got.append(s[validate_synthetic(s, real)])
        syn = np.vstack(got)[:n_new] if got else np.empty((0, X.shape[1]), np.float32)
        acc = len(syn) / max(1, n_new)
        log.info("CTGAN %-14s +%5d rows (%.0f%% of requested passed validation)", cls, len(syn), 100 * acc)
        if len(syn):
            new_X.append(syn)
            new_y.append(np.full(len(syn), cls, dtype=y.dtype))
    return np.vstack(new_X).astype(np.float32), np.concatenate(new_y)


def augment(X, y, method: str, cfg: dict, seed: int = 42):
    a = cfg["augment"]
    if method == "none":
        return X, y
    if method == "smote":
        return smote(X, y, a["target_min_per_class"], seed=seed)
    if method == "ctgan":
        return ctgan_augment(X, y, a["target_min_per_class"], a["ctgan_epochs"], a["ctgan_max_rows"], seed)
    raise ValueError(method)
