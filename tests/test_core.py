"""Unit + integration tests (project report §6.4 test ids noted). Run: pytest -q
Synthetic data only, so the suite runs offline in about a minute."""
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.augment import smote, validate_synthetic  # noqa: E402
from hitds.collect import map_columns, normalise_flows  # noqa: E402
from hitds.common import load_config  # noqa: E402
from hitds.data import CleaningLog, clean_frame, cic_family, make_synthetic, split_and_save, xy  # noqa: E402
from hitds.db import Store, cohen_kappa  # noqa: E402
from hitds.features import Preprocessor  # noqa: E402
from hitds.hmm import AlertModel, HostTracker, gamma_linear  # noqa: E402
from hitds.pipeline import evaluate_bundle, train_bundle  # noqa: E402
from hitds.smad import SMAD  # noqa: E402
from hitds.triage import Triage, normalized_entropy  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    c = load_config()
    c["models"]["mlp"]["epochs"] = 3
    c["models"]["rf"]["n_estimators"] = 40
    return c


@pytest.fixture(scope="module")
def data():
    df = make_synthetic(6000, 1)
    return df.iloc[:4200], df.iloc[4200:5100], df.iloc[5100:]


@pytest.fixture(scope="module")
def bundle(cfg, data):
    return train_bundle(cfg, data[0], data[1], method="smote")


# ------------------------------------------------------------------ data collection
def test_label_mapping():
    assert cic_family("Web Attack \x96 Brute Force") == "WebAttack"
    assert cic_family(" DoS Hulk") == "DoS"


def test_cleaning_removes_dupes_conflicts_nan_negative():
    df = pd.DataFrame({"Flow Duration": [1, 1, 2, 3, -5, np.inf, 7], "Total Fwd Packets": [1, 1, 2, 3, 1, 1, 7],
                       "label": ["BENIGN", "BENIGN", "DoS", "DoS", "DoS", "DoS", "BENIGN"]})
    extra = pd.DataFrame({"Flow Duration": [7], "Total Fwd Packets": [7], "label": ["DoS"]})   # conflicts with last row
    out = clean_frame(pd.concat([df, extra], ignore_index=True), ["Flow Duration", "Total Fwd Packets"], CleaningLog())
    assert sorted(out["Flow Duration"].tolist()) == [1, 2, 3]


def test_split_has_no_leakage(tmp_path, cfg):
    meta = split_and_save(make_synthetic(3000, 3), tmp_path, cfg, 42, dataset="t")
    assert all(v == 0 for v in meta["leakage_overlap"].values())
    assert (tmp_path / "DATA_CARD.md").exists()


def test_cicflowmeter_columns_map_to_cic_names():
    model_cols = ["Total Fwd Packets", "Flow Bytes/s", "Init_Win_bytes_forward", "Destination Port", "Flow Duration"]
    rename, rep = map_columns(["Tot Fwd Pkts", "flow_byts_s", "Init Fwd Win Byts", "Dst Port", "Flow Duration",
                               "Src IP", "Timestamp"], model_cols)
    assert rep["coverage"] == 1.0 and rename["Src IP"] == "src_ip"
    with pytest.raises(ValueError):
        normalise_flows(pd.DataFrame({"a": [1]}), model_cols)


# ------------------------------------------------------------------ preprocessing / augmentation
def test_ut01_minmax_range_and_nan_imputation(data):          # UT-01
    X, y = xy(data[0])
    pre = Preprocessor(10).fit(X, y)
    Z = pre.transform(X)
    assert Z.shape[1] == 10 and Z.min() >= 0 and Z.max() <= 1
    row = X.iloc[0].to_dict()
    row[pre.feature_names[0]] = np.nan
    assert np.isfinite(pre.transform_row(row)).all()


def test_ut02_smote_balances(data):                            # UT-02
    X, y = xy(data[0])
    Z = Preprocessor().fit(X, y).transform(X)
    _, y2 = smote(Z, y, 500)
    assert pd.Series(y2).value_counts().min() >= 500


def test_synthetic_validation_rejects_out_of_range():
    real = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert validate_synthetic(np.array([[0.2, 0.3], [0.9, 0.9]]), real).tolist() == [True, False]


def test_smad_flags_outliers():
    rng = np.random.default_rng(0)
    Z = np.vstack([rng.normal(0.3, 0.02, (500, 6)), rng.normal(0.9, 0.02, (50, 6))])
    y = np.array(["BENIGN"] * 500 + ["DoS"] * 50)
    sm = SMAD(top_k=4).fit(Z, y).calibrate(Z[:500])
    assert sm.flag(Z[500:]).mean() > 0.9 and sm.flag(Z[:500]).mean() < 0.02


# ------------------------------------------------------------------ models + triage
def test_ut04_predict_returns_label_and_confidence(bundle, data):   # UT-04
    out = bundle.score(xy(data[2])[0].iloc[:5])
    assert all(0 <= r["confidence"] <= 1 and r["pred"] in bundle.classes for r in out["routes"])


def test_ut05_ut06_triage_routes():                            # UT-05..08
    t = Triage(["BENIGN", "DoS", "Infiltration"], {"BENIGN": 0, "DoS": 3, "Infiltration": 5}, ["Infiltration"],
               h_threshold=0.3, confidence_floor=0.85)
    P = np.array([[0.98, 0.01, 0.01], [0.5, 0.45, 0.05], [0.01, 0.01, 0.98], [0.02, 0.97, 0.01], [0.97, 0.02, 0.01]])
    routes = [r["route"] for r in t.route(P, smad_flag=np.array([0, 0, 0, 0, 1], bool))]
    assert routes == ["AUTO", "REVIEW", "ESCALATE", "AUTO", "REVIEW"]   # last: benign but SMAD-flagged -> zero-day


def test_entropy_bounds():
    H = normalized_entropy(np.array([[1, 0, 0], [1 / 3, 1 / 3, 1 / 3]], dtype=float))
    assert H[0] < 1e-6 and abs(H[1] - 1) < 1e-6


def test_mlp_numpy_forward_matches_torch():
    torch = pytest.importorskip("torch")
    from hitds.models.mlp import TorchMLP, _Net
    rng = np.random.default_rng(0)
    X = rng.random((300, 8)).astype(np.float32)
    y = np.array(["a", "b", "c"])[rng.integers(0, 3, 300)]
    m = TorchMLP(hidden=(16, 8), epochs=2, verbose=False).fit(X, y)
    net = _Net(8, 3, [16, 8], 0.2)
    net.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in m._state.items()})
    net.eval()
    with torch.no_grad():
        ref = torch.softmax(net(torch.from_numpy(X)), 1).numpy()
    assert np.allclose(m.predict_proba(X), ref, atol=1e-5)


# ------------------------------------------------------------------ Layer 4 (Kim et al.)
def _tracker():
    classes = ["BENIGN", "PortScan", "BruteForce", "DoS"]
    y_true = np.array(["BENIGN"] * 90 + ["PortScan"] * 10 + ["BruteForce"] * 10 + ["DoS"] * 10)
    y_pred = y_true.copy()
    y_pred[:3] = "PortScan"                                     # a few false alerts
    am = AlertModel.from_confusion(classes, y_true, y_pred)
    return HostTracker(classes, am, {"hmm": {"log_threshold": 6.0}})


def test_gamma_confidence_function():
    assert gamma_linear(0.0, 5.0) == 5.0 and gamma_linear(0.5, 5.0) == pytest.approx(1.0)


def test_hmm_detects_progression_and_stays_quiet_on_benign():
    tr = _tracker()
    for i, p in enumerate(["PortScan"] * 4 + ["BENIGN", "BruteForce"] * 3 + ["DoS"] * 4):
        s = tr.update("victim", p, alert_id=i)
    assert s["sprt"] == "compromised" and s["belief"]["safe"] < 0.2
    for i in range(60):
        s2 = tr.update("quiet", "BENIGN", alert_id=1000 + i)
    assert s2["sprt"] == "safe"


def test_investigation_outcome_moves_likelihood_ratio():
    tr = _tracker()
    for i, p in enumerate(["BENIGN"] * 5 + ["PortScan"]):
        tr.update("h", p, alert_id=i)
    base = tr.state("h")["llr"]
    tr_tp = _tracker()
    for i, p in enumerate(["BENIGN"] * 5 + ["PortScan"]):
        tr_tp.update("h", p, alert_id=i)
    tr_tp.investigate(5, 1)
    tr.investigate(5, 0)
    assert tr_tp.state("h")["llr"] > base > tr.state("h")["llr"]
    assert tr_tp.policy_score(5) is None                     # investigated alerts leave the policy ranking


def test_policies_score_open_alerts():
    tr = _tracker()
    for i, p in enumerate(["BENIGN", "PortScan", "BENIGN", "BruteForce"]):
        tr.update("h", p, alert_id=i)
    for pol in ("max_kl", "max_ratio"):
        s = tr.policy_score(3, pol)
        assert s is not None and np.isfinite(s)


# ------------------------------------------------------------------ storage, audit, review
def _row(**kw):
    r = {"host": "h", "src_ip": "1.1.1.1", "dst_ip": "10.0.0.1", "pred": "DoS", "confidence": .6, "entropy": .5,
         "anomaly": 0, "route": "REVIEW", "reason": "x", "priority": 1, "proba": {}, "features": {"a": 1.0}}
    r.update(kw)
    return r


def test_ut09_audit_chain_and_append_only(tmp_path):          # UT-09 + ST-09
    st = Store(tmp_path / "t.sqlite")
    aid = st.add_alerts([_row()], 1)[0]
    st.add_verdict(aid, "me", "TP", "DoS", 1.0, 3.0, "block_source_ip")
    assert st.verify_chain()["ok"] and st.verify_chain()["checked"] == 3
    with pytest.raises(Exception):
        st.conn.execute("DELETE FROM audit_log")
    with pytest.raises(ValueError):
        st.add_verdict(aid, "me", "FP", None, 1.0, 1.0, None)


def test_second_review_and_kappa(tmp_path):
    st = Store(tmp_path / "t.sqlite", second_review_below=0.5)
    a1, a2 = st.add_alerts([_row(), _row()], 1)
    assert st.add_verdict(a1, "alice", "TP", "DoS", 0.4, 2, None)["status"] == "second_review"
    with pytest.raises(ValueError):
        st.add_verdict(a1, "alice", "TP", "DoS", 1.0, 2, None)
    assert st.add_verdict(a1, "bob", "FP", None, 1.0, 2, None)["status"] == "closed"
    assert [f["corrected_label"] for f in st.all_feedback()] == ["BENIGN"]   # second reviewer's label wins
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0


def test_ut10_retrain_with_feedback(cfg, data, bundle):       # UT-10 / IT-07
    fb = data[2].iloc[:50].copy()
    b2 = train_bundle(cfg, data[0], data[1], method="smote", feedback=fb, pre=bundle.pre, version=2)
    assert b2.info["n_feedback"] == 50 and b2.version == 2
    m = evaluate_bundle(b2, data[2])
    assert m["macro_f1"] > 0.5 and 0 <= m["saved_rate"] <= 1


# ------------------------------------------------------------------ web app (IT-06, ST-08)
@pytest.fixture(scope="module")
def client(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("proj")
    import shutil
    from hitds.common import ROOT
    from hitds.pipeline import save_bundle, set_live
    for d in ("hitds", "app"):
        shutil.copytree(ROOT / d, root / d)
    c2 = load_config()
    c2["models"]["mlp"]["epochs"] = 2
    c2["models"]["rf"]["n_estimators"] = 30
    df = make_synthetic(4000, 5)
    (root / "data/processed/synthetic").mkdir(parents=True)
    split_and_save(df, root / "data/processed/synthetic", c2, 42, dataset="synthetic")
    tr = pd.read_pickle(root / "data/processed/synthetic/train.pkl")
    va = pd.read_pickle(root / "data/processed/synthetic/val.pkl")
    b = train_bundle(c2, tr, va, method="smote")
    save_bundle(b, root / "models/synthetic")
    set_live(root / "models/synthetic", 1)
    import yaml
    cfg_path = root / "config.yaml"
    yaml.safe_dump({**c2, "paths": {"raw": str(root / "data/raw"), "processed": str(root / "data/processed"),
                                    "models": str(root / "models"), "reports": str(root / "reports"),
                                    "db": str(root / "data/hitds.sqlite")}}, open(cfg_path, "w"))
    os.environ.update(HITDS_USERS="analyst:pw", HITDS_API_KEY="k", HITDS_SECRET_KEY="s")
    from app.server import create_app
    app = create_app("synthetic", reset=True, config_path=str(cfg_path), testing=True)
    return app.test_client()


def test_app_auth_csrf_and_loop(client):
    assert client.get("/api/queue").status_code == 401
    assert client.post("/login", data={"username": "analyst", "password": "pw"}).status_code == 302
    H = {"X-Requested-With": "hitds"}
    assert client.post("/api/inject", json={"n": 5}).status_code == 400          # CSRF header missing
    assert client.post("/api/inject", json={"n": 200}, headers=H).json["ingested"] == 200
    q = client.get("/api/queue").json["alerts"]
    assert q and "final_priority" in q[0]
    aid = q[0]["id"]
    d = client.get(f"/api/alert/{aid}").json
    assert "ground_truth" not in d["alert"]                                      # analyst never sees the answer
    r = client.post(f"/api/alert/{aid}/verdict", json={"verdict": "FP", "confidence": 1.0}, headers=H).json
    assert r["status"] == "closed"
    assert client.get("/api/audit").json["verify"]["ok"]


def test_app_upload_and_api_ingest(client):
    client.post("/login", data={"username": "analyst", "password": "pw"})
    df = make_synthetic(200, 9).drop(columns=["label", "label_raw"]).rename(columns={"Total Fwd Packets": "Tot Fwd Pkts"})
    df["Src IP"], df["Dst IP"] = "192.168.56.5", "192.168.56.10"
    r = client.post("/api/upload", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), "f.csv")},
                    headers={"X-Requested-With": "hitds"}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.json["coverage"] >= 0.8
    assert client.post("/api/ingest", json={"flows": []}).status_code == 401
    ok = client.post("/api/ingest", json={"flows": df.head(3).to_dict("records")}, headers={"X-API-Key": "k"})
    assert ok.status_code == 200 and ok.json["ingested"] == 3
