"""The running system:
score -> triage -> host HMM (Kim et al.) -> store/audit -> policy-ranked queue -> analyst verdict
      -> HMM investigation update + feedback store -> gated retrain -> promotion.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pandas as pd

from .collect import normalise_flows
from .common import get_logger
from .data import load_split
from .db import Store
from .explain import Explainer
from .hmm import HostTracker
from .pipeline import Bundle, evaluate_bundle, next_version, save_bundle, set_live, train_bundle

log = get_logger("hitds.engine")

AUTO_ACTION = {"PortScan": "rate_limit_source", "Reconnaissance": "rate_limit_source", "Fuzzers": "rate_limit_source",
               "DoS": "rate_limit_source", "BruteForce": "temporary_lockout", "Analysis": "rate_limit_source"}
HUMAN_ACTION = {"DDoS": "upstream_blackhole", "WebAttack": "waf_block_source", "Bot": "isolate_host",
                "Infiltration": "isolate_host", "Heartbleed": "isolate_host", "Backdoor": "isolate_host",
                "Shellcode": "isolate_host", "Worms": "isolate_host", "Exploits": "block_source_ip",
                "Generic": "block_source_ip", "ATTACK": "block_source_ip"}
META_COLS = ("host", "src_ip", "dst_ip", "ground_truth", "label", "label_raw", "signature", "source",
             "src_port", "timestamp", "protocol", "gt_stage", "campaign")


def suggested_action(label: str) -> str | None:
    if label == "BENIGN":
        return None
    return HUMAN_ACTION.get(label) or AUTO_ACTION.get(label) or "block_source_ip"


class Engine:
    def __init__(self, cfg: dict, bundle: Bundle, store: Store):
        self.cfg, self.store = cfg, store
        self._lock = threading.RLock()
        self.retraining = False
        self.last_retrain: dict | None = None
        self.last_latency_ms = None
        self._set_bundle(bundle)

    def _set_bundle(self, bundle: Bundle):
        with self._lock:
            self.bundle = bundle
            self.explainer = Explainer(bundle)
            old = getattr(self, "hmm", None)
            self.hmm = HostTracker(bundle.classes, bundle.alert_model, self.cfg)
            if old is not None and old.classes == self.hmm.classes:
                self.hmm.hosts, self.hmm.index = old.hosts, old.index   # keep host beliefs across model swaps

    # ------------------------------------------------------------------ ingest
    def ingest(self, batch: pd.DataFrame, source: str = "replay") -> list[int]:
        """batch: raw feature columns + host/src_ip/dst_ip (+ optional ground_truth, signature)."""
        batch = batch.reset_index(drop=True)
        X = batch.drop(columns=[c for c in META_COLS if c in batch])
        with self._lock:
            b = self.bundle
            out = b.score(X)
            self.last_latency_ms = round(out["ms_per_event"], 3)
            rows, steps = [], []
            for i, r in enumerate(out["routes"]):
                host = str(batch["host"].iat[i])
                hs = self.hmm.update(host, r["pred"])
                steps.append((host, self.hmm.last_step(host)))
                if r["route"] == "AUTO" and r["pred"] != "BENIGN" and hs["sprt"] == "compromised":
                    r["route"] = "ESCALATE"
                    r["reason"] += f"; host {host} is past the detection threshold (log R={hs['llr']})"
                sig = batch["signature"].iat[i] if "signature" in batch else None
                if sig and isinstance(sig, str) and sig.strip() and r["route"] == "AUTO":
                    r["route"], r["reason"] = "REVIEW", r["reason"] + f"; signature hit: {sig[:80]}"
                r.update(host=host, src_ip=str(batch["src_ip"].iat[i]), dst_ip=str(batch["dst_ip"].iat[i]),
                         source=source, smad=float(out["smad"][i]), signature=sig or None,
                         proba={c: round(float(p), 5) for c, p in zip(b.classes, out["P"][i])},
                         features={k: (None if pd.isna(v) else float(v)) for k, v in X.iloc[i].items()},
                         ground_truth=batch["ground_truth"].iat[i] if "ground_truth" in batch else None,
                         auto_action=AUTO_ACTION.get(r["pred"]) if r["route"] == "AUTO" else None)
                rows.append(r)
            ids = self.store.add_alerts(rows, b.version)
            for (host, st), aid in zip(steps, ids):
                self.hmm.bind(host, st, aid)
        return ids

    def ingest_flows(self, raw: pd.DataFrame, source: str = "upload") -> dict:
        """Real flows (CICFlowMeter CSV / JSON). Host = destination IP (the asset being protected)."""
        flows, report = normalise_flows(raw, self.bundle.raw_columns,
                                        self.cfg.get("collect", {}).get("min_feature_coverage", 0.8))
        flows["src_ip"] = flows.get("src_ip", pd.Series(["unknown"] * len(flows))).fillna("unknown")
        flows["dst_ip"] = flows.get("dst_ip", pd.Series(["unknown"] * len(flows))).fillna("unknown")
        flows["host"] = flows["dst_ip"]
        if "label_raw" in flows:     # a labelled capture (e.g. a CIC CSV) - keep as hidden ground truth
            from .data import cic_family
            flows["ground_truth"] = flows.pop("label_raw").map(cic_family)
        n_max = self.cfg.get("collect", {}).get("max_rows_per_upload", 5000)
        if len(flows) > n_max:
            report["truncated_to"] = n_max
            flows = flows.iloc[:n_max]
        ids = self.ingest(flows, source=source)
        report["ingested"] = len(ids)
        self.store.audit_event("ingest", f"collector:{source}", {k: v for k, v in report.items()
                                                                  if k != "ignored_columns"})
        return report

    # ------------------------------------------------------------------ queue ordering (Layer 4 policy)
    def queue(self, limit: int = 40, analyst: str | None = None) -> list[dict]:
        rows = self.store.open_alerts(300, analyst)
        policy = self.cfg.get("hmm", {}).get("policy", "max_kl")
        w = float(self.cfg.get("hmm", {}).get("policy_weight", 3.0))
        with self._lock:
            scores = [self.hmm.policy_score(r["id"], policy) for r in rows]
        valid = [s for s in scores if s is not None and np.isfinite(s)]
        for r, s in zip(rows, scores):
            # rank-normalise the policy score to [0,1] so it combines with the triage priority scale
            pct = (sum(v <= s for v in valid) / len(valid)) if (s is not None and valid and np.isfinite(s)) else 0.0
            r["policy_score"] = None if s is None else round(float(s), 5)
            r["final_priority"] = round(r["priority"] + w * pct, 3)
        rows.sort(key=lambda r: (-r["final_priority"], r["id"]))
        return rows[:limit]

    # ------------------------------------------------------------------ analyst view
    def alert_detail(self, aid: int) -> dict | None:
        a = self.store.get_alert(aid)
        if not a:
            return None
        feats = {k: v for k, v in json.loads(a["features_json"]).items()}
        with self._lock:
            z = self.bundle.pre.transform_row(feats)
            members = {n: dict(zip(self.bundle.classes, np.round(p[0], 4).tolist()))
                       for n, p in self.bundle.det.member_proba(z.reshape(1, -1)).items()}
            attribution = self.explainer.attribution(z, a["pred"])
            similar = self.explainer.similar(z, self.store.all_feedback()[-500:])
            host = self.hmm.state(a["host"])
            smad = self.explainer.smad_evidence(z)
        a.pop("ground_truth", None)  # the analyst never sees the answer key
        a["proba"] = json.loads(a.pop("proba_json"))
        a.pop("features_json")
        top_raw = sorted(((k, v) for k, v in feats.items() if v is not None), key=lambda kv: -abs(kv[1]))[:12]
        return {"alert": a, "raw_evidence": dict(top_raw), "attribution": attribution, "similar": similar,
                "members": members, "host": host, "smad": smad, "prior_verdicts": self.store.verdicts_for(aid),
                "suggested_action": suggested_action(a["pred"]), "classes": self.bundle.classes}

    def verdict(self, aid, analyst, verdict, corrected_label, analyst_confidence, seconds, action,
                revealed=True) -> dict:
        res = self.store.add_verdict(aid, analyst, verdict, corrected_label, analyst_confidence, seconds, action,
                                     revealed)
        if verdict in ("TP", "FP") and res["status"] == "closed":
            outcome = 0 if res["label"] == "BENIGN" else 1
            with self._lock:
                hs = self.hmm.investigate(aid, outcome)
            if hs:
                res["host_state"] = hs
        pending = len(self.store.unused_feedback())
        res["pending_feedback"] = pending
        if pending >= self.cfg["feedback"]["retrain_every"] and not self.retraining:
            threading.Thread(target=self.retrain, daemon=True).start()
            res["retrain_started"] = True
        return res

    # ------------------------------------------------------------------ retrain + promotion gate
    def feedback_frame(self) -> tuple[pd.DataFrame, list[int]]:
        fb = self.store.all_feedback()
        if not fb:
            return pd.DataFrame(), []
        df = pd.DataFrame([json.loads(f["features_json"]) for f in fb])
        df["label"] = [f["corrected_label"] for f in fb]
        df["label_raw"] = df["label"]
        # reviewer confidence weights the sample (proposal §29: label-noise mitigation)
        df["_weight"] = [float(f.get("analyst_confidence") or 1.0) for f in fb]
        return df, [f["vid"] for f in fb]

    def retrain(self) -> dict:
        self.retraining = True
        t0 = time.time()
        try:
            paths = self.cfg["paths"]
            train = load_split(paths["processed"], "train")
            val = load_split(paths["processed"], "val")
            fb, vids = self.feedback_frame()
            unused = {f["vid"] for f in self.store.unused_feedback()}
            old = self.bundle
            version = next_version(paths["models"])
            log.info("retraining v%d with %d analyst labels", version, len(fb))
            new = train_bundle(self.cfg, train, val, feedback=fb, version=version, pre=old.pre,
                               members=tuple(old.info.get("members", ("rf", "svm", "knn", "mlp"))),
                               method=old.info.get("augment"), max_rows=self.cfg["feedback"].get("retrain_max_rows"))
            m_old, m_new = evaluate_bundle(old, val), evaluate_bundle(new, val)
            tol = self.cfg["feedback"]["promote_if_macro_f1_drop_below"]
            promote = m_new["macro_f1"] >= m_old["macro_f1"] - tol
            summary = {"old_version": old.version, "new_version": version, "n_feedback": len(fb),
                       "val_macro_f1_old": round(m_old["macro_f1"], 4), "val_macro_f1_new": round(m_new["macro_f1"], 4),
                       "val_far_old": m_old["false_alarm_rate"], "val_far_new": m_new["false_alarm_rate"],
                       "human_share_old": round(m_old["human_share"], 4), "human_share_new": round(m_new["human_share"], 4),
                       "seconds": round(time.time() - t0, 1)}
            new.metrics["gate"] = summary
            save_bundle(new, paths["models"])
            self.store.record_model(version, len(fb), summary, promote,
                                    "promoted" if promote else f"rejected: macro-F1 fell more than {tol}")
            self.store.mark_feedback_used([v for v in vids if v in unused], version)
            if promote:
                set_live(paths["models"], version, "feedback retrain")
                self._set_bundle(new)
            summary["promoted"] = promote
            self.last_retrain = summary
            log.info("retrain done: %s", summary)
            return summary
        except Exception as e:
            log.exception("retrain failed")
            self.last_retrain = {"error": str(e)}
            raise
        finally:
            self.retraining = False
