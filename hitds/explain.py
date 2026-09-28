"""Evidence shown to the analyst BEFORE the verdict buttons (anti-automation-bias):
  * feature attribution for the predicted class (SHAP TreeExplainer on the RF member;
    a cheap deviation-x-importance fallback if shap is not installed)
  * the most similar past incidents (training reference set + earlier analyst verdicts)
  * how each ensemble member voted (disagreement is itself evidence)
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    import shap  # noqa: F401

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class Explainer:
    def __init__(self, bundle):
        self.b = bundle
        self._shap = None
        rf = bundle.det.models.get("rf")
        if HAS_SHAP and rf is not None:
            import shap

            self._shap = shap.TreeExplainer(rf)
        self._imp = rf.feature_importances_ if rf is not None else np.ones(len(bundle.pre.feature_names))
        self._nn = NearestNeighbors(n_neighbors=5).fit(bundle.ref_X)

    def attribution(self, z: np.ndarray, cls: str, top: int = 8) -> list[dict]:
        names = self.b.pre.feature_names
        if self._shap is not None:
            sv = self._shap.shap_values(z.reshape(1, -1), check_additivity=False)
            rf_classes = list(self.b.det.models["rf"].classes_)
            k = rf_classes.index(cls) if cls in rf_classes else 0
            vals = sv[k][0] if isinstance(sv, list) else np.asarray(sv)[0, :, k]
            method = "shap"
        else:
            vals = (z - self.b.benign_mean) * self._imp
            method = "deviation_x_importance"
        order = np.argsort(-np.abs(vals))[:top]
        return [{"feature": names[i], "value": round(float(z[i]), 4), "contribution": round(float(vals[i]), 5),
                 "method": method} for i in order]

    def smad_evidence(self, z: np.ndarray) -> dict:
        """IS-IDS style: which ANOVA-selected features sit outside the benign 3-sigma band."""
        sm = getattr(self.b, "smad", None)
        if sm is None:
            return {"flag": False, "score": 0.0, "features": []}
        names = self.b.pre.feature_names
        dev = (z[sm.idx_] - sm.mu_) / sm.sd_
        out = [{"feature": names[i], "sigma": round(float(d), 2)} for i, d in zip(sm.idx_, dev) if abs(d) > sm.sigma]
        score = float(np.mean(np.abs(dev) > sm.sigma))
        return {"flag": score > sm.threshold_, "score": round(score, 3), "threshold": round(sm.threshold_, 3),
                "features": sorted(out, key=lambda r: -abs(r["sigma"]))[:8]}

    def similar(self, z: np.ndarray, feedback: list[dict] | None = None, k: int = 5) -> list[dict]:
        d, idx = self._nn.kneighbors(z.reshape(1, -1), n_neighbors=k)
        out = [{"source": "benchmark", "label": str(self.b.ref_y[i]), "distance": round(float(di), 4)}
               for di, i in zip(d[0], idx[0])]
        if feedback:
            import pandas as pd

            F = self.b.pre.transform(pd.DataFrame([json.loads(f["features_json"]) for f in feedback]))
            dist = np.linalg.norm(F - z, axis=1)
            for j in np.argsort(dist)[:3]:
                out.append({"source": f"analyst verdict on alert #{feedback[j]['alert_id']}",
                            "label": feedback[j]["corrected_label"], "distance": round(float(dist[j]), 4)})
        return sorted(out, key=lambda r: r["distance"])[:k + 2]
