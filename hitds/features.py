"""Feature preprocessing: median imputation -> signed log -> Min-Max -> top-k selection.

* Flow statistics are extremely heavy-tailed, so a signed log1p is applied before Min-Max
  (Min-Max alone squashes 99.9% of rows into a tiny range).
* Missing values (only possible in live uploads - training data has NaN rows removed) are filled
  with the training median of that column, never with zero.
* Selection: `rf` (Random Forest importance, project report section 6.1) or `anova` (ANOVA F-test, IS-IDS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    def __init__(self, top_k: int = 0, seed: int = 42, selection: str = "rf"):
        self.top_k, self.seed, self.selection = top_k, seed, selection
        self.columns_: list[str] = []
        self.selected_: list[str] = []
        self.medians_: dict[str, float] = {}
        self.scaler = MinMaxScaler(clip=True)
        self.importances_: dict[str, float] = {}

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.columns_)
        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        X = X.fillna(self.medians_)
        return signed_log1p(X.to_numpy(dtype=np.float64))

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Preprocessor":
        self.columns_ = list(X.columns)
        self.medians_ = X.median(numeric_only=True).to_dict()
        Z = self._matrix(X)
        self.scaler.fit(Z)
        Zs = self.scaler.transform(Z)
        if self.top_k and self.top_k < len(self.columns_):
            if self.selection == "anova":
                with np.errstate(divide="ignore", invalid="ignore"):
                    F, _ = f_classif(Zs, y)
                imp = pd.Series(np.nan_to_num(F), index=self.columns_)
            else:
                rng = np.random.default_rng(self.seed)
                idx = rng.choice(len(X), size=min(len(X), 50000), replace=False)
                rf = RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=self.seed,
                                            class_weight="balanced_subsample")
                rf.fit(Zs[idx], y[idx])
                imp = pd.Series(rf.feature_importances_, index=self.columns_)
            imp = imp.sort_values(ascending=False)
            self.importances_ = imp.to_dict()
            self.selected_ = list(imp.index[: self.top_k])
        else:
            self.selected_ = list(self.columns_)
        self._sel_idx = [self.columns_.index(c) for c in self.selected_]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(self._matrix(X))
        return Z[:, self._sel_idx].astype(np.float32)

    def transform_row(self, raw: dict) -> np.ndarray:
        return self.transform(pd.DataFrame([raw]))[0]

    @property
    def feature_names(self) -> list[str]:
        return list(self.selected_)
