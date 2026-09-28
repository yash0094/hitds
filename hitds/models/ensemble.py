"""Layer 2: soft-voting ensemble (RF + RBF-SVM + KNN + PyTorch MLP) fused at the
probability level, plus an Isolation Forest trained on benign traffic only for
behaviour with no supervised analogue (zero-days).
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ..common import get_logger
from .mlp import TorchMLP

log = get_logger("hitds.ensemble")


def stratified_cap(y: np.ndarray, max_rows: int, seed: int = 42) -> np.ndarray:
    """Indices of a subsample of <= max_rows that keeps every class (rare classes kept whole)."""
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    budget, keep = max_rows, []
    per = budget // len(classes)
    # rare classes first, whole; leftover budget flows to bigger classes
    for c, n in sorted(zip(classes, counts), key=lambda t: t[1]):
        idx = np.flatnonzero(y == c)
        take = min(n, max(per, 1))
        keep.append(rng.choice(idx, take, replace=False))
        budget -= take
        remaining = len(classes) - len(keep)
        per = budget // remaining if remaining else 0
    return np.concatenate(keep)


class EnsembleDetector:
    MEMBERS = ("rf", "svm", "knn", "mlp")

    def __init__(self, cfg: dict, members: tuple[str, ...] = MEMBERS, seed: int = 42):
        self.cfg = cfg["models"]
        self.members = tuple(members)
        self.seed = seed
        self.models: dict = {}
        self.classes_: np.ndarray | None = None
        self.iforest: IsolationForest | None = None
        self.fit_seconds: dict[str, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.classes_ = np.unique(y)
        c = self.cfg
        for name in self.members:
            t0 = time.time()
            log.info("fitting %s on %d rows", name, len(y))
            if name == "rf":
                m = RandomForestClassifier(**{k: v for k, v in c["rf"].items()}, random_state=self.seed)
                m.fit(X, y, sample_weight=sample_weight)
            elif name == "svm":
                idx = stratified_cap(y, c["svm"]["max_train_rows"], self.seed)
                m = SVC(C=c["svm"]["C"], gamma=c["svm"]["gamma"], kernel="rbf", probability=True,
                        class_weight="balanced", random_state=self.seed)
                m.fit(X[idx], y[idx])
            elif name == "knn":
                idx = stratified_cap(y, c["knn"]["max_train_rows"], self.seed)
                m = KNeighborsClassifier(n_neighbors=c["knn"]["n_neighbors"], weights="distance", n_jobs=-1)
                m.fit(X[idx], y[idx])
            elif name == "mlp":
                mc = c["mlp"]
                m = TorchMLP(mc["hidden"], mc["dropout"], mc["epochs"], mc["batch_size"], mc["lr"], self.seed)
                m.fit(X, y, sample_weight=sample_weight)
            else:
                raise ValueError(name)
            self.models[name] = m
            self.fit_seconds[name] = round(time.time() - t0, 1)
            log.info("  %s done in %.1fs", name, self.fit_seconds[name])

        benign = X[y == "BENIGN"] if np.any(y == "BENIGN") else X
        ic = self.cfg["iforest"]
        idx = np.random.default_rng(self.seed).choice(len(benign), min(len(benign), ic["max_train_rows"]),
                                                      replace=False)
        self.iforest = IsolationForest(n_estimators=ic["n_estimators"], contamination=ic["contamination"],
                                       random_state=self.seed, n_jobs=-1).fit(benign[idx])
        return self

    # ------------------------------------------------------------------ predict
    def _aligned(self, name: str, X: np.ndarray) -> np.ndarray:
        m = self.models[name]
        p = m.predict_proba(X)
        out = np.zeros((len(X), len(self.classes_)), np.float64)
        pos = {c: i for i, c in enumerate(self.classes_)}
        for j, c in enumerate(m.classes_):
            out[:, pos[c]] = p[:, j]
        return out

    def member_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        return {n: self._aligned(n, X) for n in self.members}

    def predict_proba(self, X: np.ndarray, members_out: dict | None = None) -> np.ndarray:
        mp = members_out or self.member_proba(X)
        w = np.array([self.cfg["weights"].get(n, 1.0) for n in self.members], np.float64)
        w = w / w.sum()
        fused = sum(wi * mp[n] for wi, n in zip(w, self.members))
        return fused / fused.sum(1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(1)]

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous relative to benign traffic."""
        return -self.iforest.score_samples(X)
