"""Cross-dataset generalisation (proposal §27: "training on CIC-IDS2017 with UNSW-NB15 held out ... the
harder and more honest test").

The two datasets use different feature extractors, so both are mapped to a small SHARED flow schema
(binary label: attack vs benign) before training on one and testing on the other, in both directions.
Expect a large drop - that drop is the finding, not a bug.

    python scripts/07_cross_dataset.py            # needs data/raw for both datasets (scripts/01_download.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import load_config, use_dataset  # noqa: E402
from hitds.data import load_split  # noqa: E402
from hitds.features import signed_log1p  # noqa: E402

SHARED = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "fwd_pkt_mean", "bwd_pkt_mean",
          "bytes_per_s", "pkts_per_s"]


def from_cic(df: pd.DataFrame) -> pd.DataFrame:
    dur = df["Flow Duration"].clip(lower=0) / 1e6                      # CIC: microseconds
    out = pd.DataFrame({
        "duration_s": dur, "fwd_pkts": df["Total Fwd Packets"], "bwd_pkts": df["Total Backward Packets"],
        "fwd_bytes": df["Total Length of Fwd Packets"], "bwd_bytes": df["Total Length of Bwd Packets"],
        "fwd_pkt_mean": df["Fwd Packet Length Mean"], "bwd_pkt_mean": df["Bwd Packet Length Mean"]})
    return _rates(out, df["label"])


def from_unsw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "duration_s": df["dur"], "fwd_pkts": df["spkts"], "bwd_pkts": df["dpkts"], "fwd_bytes": df["sbytes"],
        "bwd_bytes": df["dbytes"], "fwd_pkt_mean": df["smean"], "bwd_pkt_mean": df["dmean"]})
    return _rates(out, df["label"])


def _rates(out, label):
    d = out["duration_s"].replace(0, np.nan)
    out["bytes_per_s"] = ((out.fwd_bytes + out.bwd_bytes) / d).fillna(0)
    out["pkts_per_s"] = ((out.fwd_pkts + out.bwd_pkts) / d).fillna(0)
    out["y"] = (label.to_numpy() != "BENIGN").astype(int)
    return out


def run(train, test, name):
    Xtr, Xte = signed_log1p(train[SHARED].to_numpy(float)), signed_log1p(test[SHARED].to_numpy(float))
    rf = RandomForestClassifier(300, n_jobs=-1, class_weight="balanced_subsample", random_state=42).fit(Xtr, train.y)
    p = rf.predict_proba(Xte)[:, 1]
    yhat = (p > 0.5).astype(int)
    return {"setting": name, "f1_attack": float(f1_score(test.y, yhat)), "mcc": float(matthews_corrcoef(test.y, yhat)),
            "roc_auc": float(roc_auc_score(test.y, p)), "DR": float(yhat[test.y == 1].mean()),
            "FAR": float(yhat[test.y == 0].mean()), "n_test": int(len(test))}


if __name__ == "__main__":
    base = load_config()
    c = use_dataset(base, "cicids2017")
    u = use_dataset(base, "unsw_nb15")
    cic_tr, cic_te = from_cic(load_split(c["paths"]["processed"], "train")), from_cic(load_split(c["paths"]["processed"], "test"))
    uns_tr, uns_te = from_unsw(load_split(u["paths"]["processed"], "train")), from_unsw(load_split(u["paths"]["processed"], "test"))
    rows = [run(cic_tr, cic_te, "CIC -> CIC (in-domain, shared features)"),
            run(cic_tr, uns_te, "CIC -> UNSW (cross-dataset)"),
            run(uns_tr, uns_te, "UNSW -> UNSW (in-domain, shared features)"),
            run(uns_tr, cic_te, "UNSW -> CIC (cross-dataset)")]
    out = Path(base["paths"]["reports"]) / "cross_dataset.json"
    out.write_text(json.dumps({"shared_features": SHARED, "rows": rows}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print("wrote", out)
