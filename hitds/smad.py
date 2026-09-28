"""Innate statistical layer, after Dutt, Borah & Maitra, "Immune System Based Intrusion Detection
System (IS-IDS): A Proposed Model", IEEE Access 8 (2020) 34929-34941.

IS-IDS's first layer, SMAD (Statistical Modelling-based Anomaly Detection), plays the innate immune
system: it uses ANOVA to find which traffic characteristics differ between normal and suspicious
behaviour, and a three-sigma rule on normal behaviour to raise a first-hand "suspicious" flag.

Here:
  * ANOVA F-test (sklearn f_classif, benign vs. attack) ranks the features;
  * the top-k features get a benign mean and standard deviation;
  * score(x) = fraction of those features outside mean +/- 3 sigma;
  * the flag threshold is calibrated on validation benign traffic (percentile, default 99.5).
The flag is evidence for the analyst and a zero-day trigger in triage when the ensemble says BENIGN.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import f_classif


class SMAD:
    def __init__(self, top_k: int = 20, sigma: float = 3.0):
        self.top_k, self.sigma = top_k, sigma
        self.idx_ = None
        self.mu_ = self.sd_ = None
        self.threshold_ = 1.0
        self.f_scores_ = None

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "SMAD":
        is_attack = (np.asarray(y) != "BENIGN").astype(int)
        with np.errstate(divide="ignore", invalid="ignore"):
            F, _ = f_classif(Z, is_attack)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0)
        self.f_scores_ = F
        self.idx_ = np.argsort(-F)[: self.top_k]
        benign = Z[is_attack == 0][:, self.idx_]
        self.mu_ = benign.mean(0)
        self.sd_ = np.maximum(benign.std(0), 1e-6)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        dev = np.abs(Z[:, self.idx_] - self.mu_) / self.sd_
        return (dev > self.sigma).mean(1)

    def calibrate(self, Z_benign: np.ndarray, percentile: float = 99.5) -> "SMAD":
        self.threshold_ = float(np.percentile(self.score(Z_benign), percentile)) if len(Z_benign) else 1.0
        return self

    def flag(self, Z: np.ndarray) -> np.ndarray:
        return self.score(Z) > self.threshold_
