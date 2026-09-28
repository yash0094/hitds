"""Summarise a live dashboard session - the REAL human-in-the-loop numbers for the report.

    python scripts/05_report_live.py --dataset cicids2017

Reads data/hitds_<dataset>.sqlite and writes reports/<dataset>/live_session.md + .json:
per-model-version workload (escalation volume, SavedRate), verdicts, analyst accuracy against hidden
ground truth, agreement with the model blind vs after reveal (automation-bias monitor), mean/median
time to decision, inter-annotator agreement (Cohen's kappa), retrain history and the audit-chain check.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import load_config, use_dataset  # noqa: E402
from hitds.db import Store  # noqa: E402


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017")
    a = ap.parse_args()
    cfg = use_dataset(load_config(), a.dataset)
    con = sqlite3.connect(cfg["paths"]["db"])
    al = pd.read_sql("SELECT * FROM alerts", con)
    vd = pd.read_sql("SELECT v.*, a.model_version, a.ground_truth, a.pred FROM verdicts v "
                     "JOIN alerts a ON a.id=v.alert_id", con)
    rows = []
    for ver, g in al.groupby("model_version"):
        auto = g[g.route == "AUTO"]
        v = vd[(vd.model_version == ver) & vd.corrected_label.notna()]
        known = v[v.ground_truth.notna()]
        rows.append({
            "model": f"v{ver}", "events": len(g),
            "escalation_volume": int((g.route != "AUTO").sum()),
            "human_share": round((g.route != "AUTO").mean(), 4),
            "saved_rate": round((g.route == "AUTO").mean(), 4),
            "auto_error_rate": round((auto.pred != auto.ground_truth).mean(), 4) if auto.ground_truth.notna().any() else None,
            "missed_attacks_auto": int(((auto.pred == "BENIGN") & (auto.ground_truth.notna()) &
                                        (auto.ground_truth != "BENIGN")).sum()),
            "verdicts": len(v),
            "analyst_accuracy_vs_truth": round((known.corrected_label == known.ground_truth).mean(), 4) if len(known) else None,
            "agree_blind": round(v[v.revealed_model == 0].agreed_with_model.mean(), 4) if (v.revealed_model == 0).any() else None,
            "agree_after_reveal": round(v[v.revealed_model == 1].agreed_with_model.mean(), 4) if (v.revealed_model == 1).any() else None,
            "mean_s_to_decide": round(v.seconds_to_decide.mean(), 1) if len(v) else None,
            "median_s_to_decide": round(v.seconds_to_decide.median(), 1) if len(v) else None,
        })
    store = Store(cfg["paths"]["db"])
    ia = store.inter_annotator()
    hist = pd.read_sql("SELECT version, n_feedback, promoted, note FROM model_versions ORDER BY version", con)
    chain = store.verify_chain()
    df = pd.DataFrame(rows)
    out_dir = Path(cfg["paths"]["reports"])
    (out_dir / "live_session.json").write_text(json.dumps({"per_version": rows, "inter_annotator": ia,
                                                           "retrains": hist.to_dict("records"), "audit": chain},
                                                          indent=2, default=str))
    text = "\n".join([f"# Live session - {a.dataset}", "", "## Per model version", md(df), "",
                      f"## Inter-annotator agreement\n{ia}", "", "## Retrain history", md(hist), "",
                      f"## Audit chain\n{chain}", "",
                      "Ground truth comes from the replayed benchmark labels and is never shown to the analyst."])
    (out_dir / "live_session.md").write_text(text)
    print(text)
    print("wrote", out_dir / "live_session.md")
