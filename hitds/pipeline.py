"""Train / evaluate / version the full detection bundle.

A *bundle* is everything needed to score one event:
    preprocessor -> SMAD (innate) + ensemble + isolation forest -> triage thresholds -> HMM alert model
plus a small labelled reference set for "similar past incidents".
Bundles are versioned in models/<dataset>/v<N>/ and registry.json points at the live one.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

from .augment import augment
from .common import get_logger
from .data import xy
from .features import Preprocessor
from .hmm import AlertModel
from .models.ensemble import EnsembleDetector, stratified_cap
from .smad import SMAD
from .triage import Triage, normalized_entropy

log = get_logger("hitds.pipeline")


@dataclass
class Bundle:
    pre: Preprocessor
    det: EnsembleDetector
    triage: Triage
    smad: SMAD | None = None
    alert_model: AlertModel | None = None
    version: int = 1
    ref_X: np.ndarray | None = None
    ref_y: np.ndarray | None = None
    benign_mean: np.ndarray | None = None
    metrics: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def classes(self) -> list[str]:
        return list(self.det.classes_)

    @property
    def raw_columns(self) -> list[str]:
        return list(self.pre.columns_)

    def score(self, X_raw: pd.DataFrame) -> dict:
        t0 = time.perf_counter()
        Z = self.pre.transform(X_raw)
        members = self.det.member_proba(Z)
        P = self.det.predict_proba(Z, members)
        an = self.det.anomaly_score(Z)
        smad_score = self.smad.score(Z) if self.smad is not None else np.zeros(len(Z))
        smad_flag = self.smad.flag(Z) if self.smad is not None else np.zeros(len(Z), bool)
        routes = self.triage.route(P, an, smad_flag)
        ms = (time.perf_counter() - t0) * 1000 / max(1, len(Z))
        return {"Z": Z, "P": P, "members": members, "anomaly": an, "smad": smad_score, "smad_flag": smad_flag,
                "routes": routes, "ms_per_event": ms}


# --------------------------------------------------------------------------- metrics
def evaluate_predictions(y_true, P, classes) -> dict:
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    y_pred = classes[P.argmax(1)]
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    benign = y_true == "BENIGN"
    attack = ~benign
    # binary view (attack vs benign): detection rate (DR) and false alarm rate (FAR), as in the NIDS literature
    far = float(np.mean(y_pred[benign] != "BENIGN")) if benign.any() else None
    dr = float(np.mean(y_pred[attack] != "BENIGN")) if attack.any() else None
    try:
        known = np.isin(y_true, classes)
        if len(classes) > 2:
            auc = float(roc_auc_score(y_true[known], P[known], labels=classes, multi_class="ovr", average="macro"))
        else:
            auc = float(roc_auc_score(y_true[known] != "BENIGN", 1 - P[known, list(classes).index("BENIGN")]))
    except Exception:
        auc = None
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(p)), "macro_recall": float(np.mean(r)), "macro_f1": float(np.mean(f)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc_ovr_macro": auc,
        "false_positive_rate": far, "false_alarm_rate": far,
        "attack_detection_rate": dr,
        "per_class": {lab: {"precision": float(pi), "recall": float(ri), "f1": float(fi), "support": int(si)}
                      for lab, pi, ri, fi, si in zip(labels, p, r, f, s)},
        "confusion_labels": labels,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_bundle(bundle: Bundle, df: pd.DataFrame) -> dict:
    X, y = xy(df)
    out = bundle.score(X)
    m = evaluate_predictions(y, out["P"], bundle.classes)
    routes = pd.Series([r["route"] for r in out["routes"]])
    m["route_share"] = routes.value_counts(normalize=True).round(4).to_dict()
    auto = (routes == "AUTO").to_numpy()
    y_pred = np.asarray(bundle.classes)[out["P"].argmax(1)]
    m["human_share"] = float(1 - auto.mean())
    m["saved_rate"] = float(auto.mean())      # SavedRate = 1 - human-annotated / all (HITL review paper)
    m["auto_accuracy"] = float(np.mean(y_pred[auto] == y[auto])) if auto.any() else None
    m["missed_attacks_in_auto"] = int(np.sum(auto & (y != "BENIGN") & (y_pred == "BENIGN")))
    m["ms_per_event"] = round(out["ms_per_event"], 3)
    m["members"] = {n: evaluate_predictions(y, P, bundle.classes)["macro_f1"] for n, P in out["members"].items()}
    return m


# --------------------------------------------------------------------------- training
def train_bundle(cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, method: str | None = None,
                 members: tuple[str, ...] = EnsembleDetector.MEMBERS, feedback: pd.DataFrame | None = None,
                 version: int = 1, pre: Preprocessor | None = None, max_rows: int | None = None) -> Bundle:
    seed = cfg["seed"]
    method = method or cfg["augment"]["method"]
    t0 = time.time()
    X, y = xy(train_df)
    fc = cfg["features"]
    if pre is None:
        pre = Preprocessor(fc["top_k"], seed, fc.get("selection", "rf")).fit(X, y)
    Z = pre.transform(X)
    if max_rows and len(y) > max_rows:
        idx = stratified_cap(y, max_rows, seed)
        Z, y = Z[idx], y[idx]
    smad = SMAD(cfg.get("smad", {}).get("top_k", 20), cfg.get("smad", {}).get("sigma", 3.0)).fit(Z, y)
    Z_real, y_real = Z, y
    Z, y = augment(Z, y, method, cfg, seed)
    w = np.ones(len(y), np.float32)
    n_fb = 0
    if feedback is not None and len(feedback):
        Xf, yf = xy(feedback)
        Zf = pre.transform(Xf)
        wf = feedback.get("_weight", pd.Series(1.0, index=feedback.index)).to_numpy(np.float32)
        Z, y = np.vstack([Z, Zf]), np.concatenate([y, yf])
        w = np.concatenate([w, wf * cfg["feedback"]["feedback_weight"]])
        n_fb = len(yf)
    log.info("training ensemble on %d rows (%d analyst-labelled), %d features, classes=%s",
             len(y), n_fb, Z.shape[1], list(np.unique(y)))
    det = EnsembleDetector(cfg, members, seed).fit(Z, y, sample_weight=w)

    Xv, yv = xy(val_df)
    Zv = pre.transform(Xv)
    Pv = det.predict_proba(Zv)
    smad.calibrate(Zv[yv == "BENIGN"], cfg.get("smad", {}).get("percentile", 99.5))
    tri = Triage.from_config(det.classes_, cfg).calibrate(Pv, det.anomaly_score(Zv), yv)
    pred_v = det.classes_[Pv.argmax(1)]
    am = AlertModel.from_confusion(list(det.classes_), yv, pred_v, tuple(cfg.get("hmm", {}).get("rho", (0, .5, .6, .8))))

    idx = stratified_cap(y_real, 5000, seed)
    b = Bundle(pre=pre, det=det, triage=tri, smad=smad, alert_model=am, version=version,
               ref_X=Z_real[idx], ref_y=y_real[idx],
               benign_mean=Z_real[y_real == "BENIGN"].mean(0) if np.any(y_real == "BENIGN") else Z_real.mean(0))
    b.info = {"augment": method, "members": list(members), "n_train": int(len(y)), "n_feedback": n_fb,
              "train_seconds": round(time.time() - t0, 1), "fit_seconds": det.fit_seconds,
              "triage": tri.stats, "features": pre.feature_names, "raw_columns": pre.columns_,
              "smad_features": [pre.feature_names[i] for i in smad.idx_], "smad_threshold": smad.threshold_,
              "alert_model": {"zeta": dict(zip(am.alert_types, np.round(am.zeta, 5).tolist()))}}
    b.metrics["val"] = evaluate_predictions(yv, Pv, det.classes_)
    log.info("val macro-F1 %.4f  acc %.4f  FAR %s", b.metrics["val"]["macro_f1"], b.metrics["val"]["accuracy"],
             b.metrics["val"]["false_alarm_rate"])
    return b


# --------------------------------------------------------------------------- registry
def save_bundle(bundle: Bundle, models_dir: str | Path) -> Path:
    d = Path(models_dir) / f"v{bundle.version}"
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, d / "bundle.joblib", compress=3)
    (d / "metrics.json").write_text(json.dumps({"info": bundle.info, "metrics": bundle.metrics}, indent=2, default=str))
    return d


def set_live(models_dir: str | Path, version: int, note: str = ""):
    reg_path = Path(models_dir) / "registry.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {"history": []}
    reg["live"] = version
    reg["history"].append({"version": version, "ts": time.time(), "note": note})
    reg_path.write_text(json.dumps(reg, indent=2))


def load_live(models_dir: str | Path) -> Bundle:
    reg = json.loads((Path(models_dir) / "registry.json").read_text())
    return joblib.load(Path(models_dir) / f"v{reg['live']}" / "bundle.joblib")


def next_version(models_dir: str | Path) -> int:
    vs = [int(p.name[1:]) for p in Path(models_dir).glob("v*") if p.name[1:].isdigit()]
    return max(vs, default=0) + 1


__all__ = ["Bundle", "train_bundle", "evaluate_bundle", "evaluate_predictions", "save_bundle", "load_live",
           "set_live", "next_version", "normalized_entropy"]
