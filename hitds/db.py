"""SQLite storage: alerts, analyst verdicts, actions, model versions and a
hash-chained, append-only audit log.

Compliance design (proposal §27 "system integration issues"):
  * every state change and its audit record are written in ONE transaction,
    so no action can exist without its justification;
  * the audit table rejects UPDATE and DELETE via triggers;
  * each audit row stores sha256(prev_hash + canonical payload), so any edit made
    outside the application breaks the chain and `verify_chain()` reports where.

Label-noise mitigation (proposal §29): a verdict given with low confidence sends the alert to
`second_review`; the final label comes from the second reviewer, and Cohen's kappa between
first and second reviewers is reported as inter-annotator agreement.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, host TEXT, src_ip TEXT, dst_ip TEXT, source TEXT,
    pred TEXT, confidence REAL, entropy REAL, anomaly REAL, smad REAL, signature TEXT,
    route TEXT, reason TEXT, priority REAL, status TEXT,
    proba_json TEXT, features_json TEXT, model_version INTEGER,
    ground_truth TEXT            -- hidden from the analyst; only used to score the loop
);
CREATE INDEX IF NOT EXISTS ix_alerts_status ON alerts(status, priority);
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER, ts REAL, analyst TEXT,
    verdict TEXT CHECK (verdict IN ('TP','FP','UNKNOWN')),
    corrected_label TEXT, analyst_confidence REAL, seconds_to_decide REAL,
    agreed_with_model INTEGER, revealed_model INTEGER, used_in_version INTEGER
);
CREATE INDEX IF NOT EXISTS ix_verdicts_alert ON verdicts(alert_id);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id INTEGER, ts REAL, action TEXT, target TEXT, authorised_by TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    version INTEGER PRIMARY KEY, ts REAL, n_feedback INTEGER, metrics_json TEXT, promoted INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, event TEXT, alert_id INTEGER, actor TEXT, payload TEXT, prev_hash TEXT, hash TEXT
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
"""

GENESIS = "0" * 64
OPEN_STATES = ("open", "second_review")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def cohen_kappa(a: list, b: list) -> float | None:
    if len(a) < 2:
        return None
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


class Store:
    def __init__(self, path: str | Path, second_review_below: float = 0.5, distinct_second_reviewer: bool = True):
        self.path = str(path)
        self.second_review_below = second_review_below
        self.distinct_second_reviewer = distinct_second_reviewer
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ tx + audit
    @contextmanager
    def tx(self):
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def _audit(self, c, event: str, alert_id, actor: str, payload: dict):
        row = c.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else GENESIS
        ts = time.time()
        body = _canon({"ts": ts, "event": event, "alert_id": alert_id, "actor": actor, "payload": payload})
        h = hashlib.sha256((prev + body).encode()).hexdigest()
        c.execute("INSERT INTO audit_log (ts,event,alert_id,actor,payload,prev_hash,hash) VALUES (?,?,?,?,?,?,?)",
                  (ts, event, alert_id, actor, _canon(payload), prev, h))

    def audit_event(self, event: str, actor: str, payload: dict, alert_id=None):
        with self.tx() as c:
            self._audit(c, event, alert_id, actor, payload)

    def verify_chain(self) -> dict:
        prev, n = GENESIS, 0
        for r in self.conn.execute("SELECT * FROM audit_log ORDER BY seq"):
            body = _canon({"ts": r["ts"], "event": r["event"], "alert_id": r["alert_id"], "actor": r["actor"],
                           "payload": json.loads(r["payload"])})
            if r["prev_hash"] != prev or hashlib.sha256((prev + body).encode()).hexdigest() != r["hash"]:
                return {"ok": False, "broken_at_seq": r["seq"], "checked": n}
            prev, n = r["hash"], n + 1
        return {"ok": True, "checked": n, "head": prev}

    # ------------------------------------------------------------------ alerts
    def add_alerts(self, rows: list[dict], model_version: int):
        ids = []
        with self.tx() as c:
            for a in rows:
                status = "open" if a["route"] in ("REVIEW", "ESCALATE") else "auto"
                cur = c.execute(
                    "INSERT INTO alerts (ts,host,src_ip,dst_ip,source,pred,confidence,entropy,anomaly,smad,signature,"
                    "route,reason,priority,status,proba_json,features_json,model_version,ground_truth) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (a.get("ts", time.time()), a["host"], a["src_ip"], a["dst_ip"], a.get("source", "replay"),
                     a["pred"], a["confidence"], a["entropy"], a["anomaly"], a.get("smad", 0.0), a.get("signature"),
                     a["route"], a["reason"], a["priority"], status, _canon(a["proba"]), _canon(a["features"]),
                     model_version, a.get("ground_truth")))
                aid = cur.lastrowid
                ids.append(aid)
                self._audit(c, "scored", aid, f"model:v{model_version}",
                            {"pred": a["pred"], "confidence": round(a["confidence"], 4),
                             "entropy": round(a["entropy"], 4), "route": a["route"], "reason": a["reason"],
                             "source": a.get("source", "replay")})
                if a.get("auto_action"):
                    c.execute("INSERT INTO actions (alert_id,ts,action,target,authorised_by,status) VALUES (?,?,?,?,?,?)",
                              (aid, time.time(), a["auto_action"], a["src_ip"], "policy:auto", "executed(simulated)"))
                    self._audit(c, "action", aid, "policy:auto", {"action": a["auto_action"], "target": a["src_ip"]})
        return ids

    def open_alerts(self, limit: int = 200, analyst: str | None = None):
        rows = [dict(r) for r in self.conn.execute(
            "SELECT id,ts,host,src_ip,dst_ip,source,pred,confidence,entropy,route,reason,priority,status "
            "FROM alerts WHERE status IN ('open','second_review') ORDER BY priority DESC, id ASC LIMIT ?", (limit,))]
        if analyst and self.distinct_second_reviewer:
            first = {r["alert_id"]: r["analyst"] for r in self.conn.execute(
                "SELECT alert_id, analyst FROM verdicts WHERE alert_id IN (SELECT id FROM alerts WHERE status='second_review')")}
            rows = [r for r in rows if not (r["status"] == "second_review" and first.get(r["id"]) == analyst)]
        return rows

    def get_alert(self, aid: int):
        r = self.conn.execute("SELECT * FROM alerts WHERE id=?", (aid,)).fetchone()
        return dict(r) if r else None

    def verdicts_for(self, aid: int):
        return [dict(r) for r in self.conn.execute("SELECT analyst,verdict,corrected_label,analyst_confidence,ts "
                                                   "FROM verdicts WHERE alert_id=? ORDER BY id", (aid,))]

    def recent_alerts(self, limit=30):
        return [dict(r) for r in self.conn.execute("SELECT id,ts,host,src_ip,pred,confidence,route,status,source "
                                                   "FROM alerts ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ verdicts + actions
    def add_verdict(self, aid: int, analyst: str, verdict: str, corrected_label: str | None,
                    analyst_confidence: float, seconds: float, action: str | None, revealed: bool = True) -> dict:
        a = self.get_alert(aid)
        if a is None:
            raise KeyError(aid)
        if a["status"] not in OPEN_STATES:
            raise ValueError(f"alert {aid} is already {a['status']}")
        prior = self.verdicts_for(aid)
        if a["status"] == "second_review" and self.distinct_second_reviewer and prior and prior[-1]["analyst"] == analyst:
            raise ValueError("second review must be done by a different analyst")
        final_label = corrected_label or (a["pred"] if verdict == "TP" else ("BENIGN" if verdict == "FP" else None))
        agreed = int(final_label == a["pred"]) if final_label else None
        if verdict == "UNKNOWN":
            new_status = "needs_info"
        elif a["status"] == "open" and analyst_confidence < self.second_review_below:
            new_status = "second_review"
        else:
            new_status = "closed"
        with self.tx() as c:
            cur = c.execute("INSERT INTO verdicts (alert_id,ts,analyst,verdict,corrected_label,analyst_confidence,"
                            "seconds_to_decide,agreed_with_model,revealed_model) VALUES (?,?,?,?,?,?,?,?,?)",
                            (aid, time.time(), analyst, verdict, final_label, analyst_confidence, seconds, agreed,
                             int(revealed)))
            c.execute("UPDATE alerts SET status=? WHERE id=?", (new_status, aid))
            self._audit(c, "verdict", aid, f"analyst:{analyst}",
                        {"verdict": verdict, "label": final_label, "model_pred": a["pred"], "agreed": agreed,
                         "seconds_to_decide": round(seconds, 1), "analyst_confidence": analyst_confidence,
                         "status": new_status, "saw_model_verdict": bool(revealed)})
            if action and verdict == "TP" and new_status == "closed":
                c.execute("INSERT INTO actions (alert_id,ts,action,target,authorised_by,status) VALUES (?,?,?,?,?,?)",
                          (aid, time.time(), action, a["src_ip"], analyst, "executed(simulated)"))
                self._audit(c, "action", aid, f"analyst:{analyst}", {"action": action, "target": a["src_ip"]})
        return {"verdict_id": cur.lastrowid, "label": final_label, "agreed": agreed, "status": new_status}

    _LATEST = ("SELECT v.id vid, v.alert_id, v.corrected_label, v.analyst_confidence, v.used_in_version, "
               "a.features_json, a.pred FROM verdicts v JOIN alerts a ON a.id=v.alert_id "
               "WHERE v.corrected_label IS NOT NULL AND a.status='closed' "
               "AND v.id = (SELECT MAX(id) FROM verdicts v2 WHERE v2.alert_id = v.alert_id)")

    def all_feedback(self):
        """Final label per closed alert (a second reviewer's label overrides the first)."""
        return [dict(r) for r in self.conn.execute(self._LATEST)]

    def unused_feedback(self):
        return [f for f in self.all_feedback() if f["used_in_version"] is None]

    def mark_feedback_used(self, vids: list[int], version: int):
        with self.tx() as c:
            c.executemany("UPDATE verdicts SET used_in_version=? WHERE id=?", [(version, v) for v in vids])

    def record_model(self, version: int, n_feedback: int, metrics: dict, promoted: bool, note: str):
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?)",
                      (version, time.time(), n_feedback, _canon(metrics), int(promoted), note))
            self._audit(c, "model_promoted" if promoted else "model_rejected", None, "system:retrainer",
                        {"version": version, "n_feedback": n_feedback, "metrics": metrics, "note": note})

    # ------------------------------------------------------------------ metrics
    def inter_annotator(self) -> dict:
        pairs = self.conn.execute(
            "SELECT alert_id, GROUP_CONCAT(corrected_label, '|') labs FROM verdicts WHERE corrected_label IS NOT NULL "
            "GROUP BY alert_id HAVING COUNT(*) >= 2").fetchall()
        a, b = [], []
        for p in pairs:
            labs = p["labs"].split("|")
            a.append(labs[0])
            b.append(labs[1])
        return {"double_reviewed": len(a), "cohen_kappa": cohen_kappa(a, b),
                "raw_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4) if a else None}

    def metrics(self) -> dict:
        q = lambda sql, *a: self.conn.execute(sql, a).fetchone()[0]  # noqa: E731
        total = q("SELECT COUNT(*) FROM alerts")
        by_route = {r["route"]: r["n"] for r in self.conn.execute("SELECT route, COUNT(*) n FROM alerts GROUP BY route")}
        v = self.conn.execute("SELECT COUNT(*) n, AVG(seconds_to_decide) t, AVG(agreed_with_model) ag FROM verdicts").fetchone()
        blind = self.conn.execute("SELECT AVG(agreed_with_model) FROM verdicts WHERE revealed_model=0").fetchone()[0]
        seen = self.conn.execute("SELECT AVG(agreed_with_model) FROM verdicts WHERE revealed_model=1").fetchone()[0]
        auto_wrong = q("SELECT COUNT(*) FROM alerts WHERE route='AUTO' AND ground_truth IS NOT NULL "
                       "AND ground_truth!=pred")
        auto_known = q("SELECT COUNT(*) FROM alerts WHERE route='AUTO' AND ground_truth IS NOT NULL")
        human = by_route.get("REVIEW", 0) + by_route.get("ESCALATE", 0)
        return {
            "total_events": total, "by_route": by_route,
            "human_share": round(human / total, 4) if total else 0,
            "saved_rate": round(1 - human / total, 4) if total else None,
            "open": q("SELECT COUNT(*) FROM alerts WHERE status IN ('open','second_review')"),
            "second_review": q("SELECT COUNT(*) FROM alerts WHERE status='second_review'"),
            "verdicts": v["n"], "mean_seconds_to_decide": round(v["t"] or 0, 1),
            "analyst_model_agreement": round(v["ag"], 3) if v["ag"] is not None else None,
            "agreement_blind": round(blind, 3) if blind is not None else None,
            "agreement_after_reveal": round(seen, 3) if seen is not None else None,
            "auto_error_rate": round(auto_wrong / auto_known, 4) if auto_known else None,
            "missed_attacks_auto_benign": q("SELECT COUNT(*) FROM alerts WHERE route='AUTO' AND pred='BENIGN' "
                                            "AND ground_truth IS NOT NULL AND ground_truth!='BENIGN'"),
            "actions": q("SELECT COUNT(*) FROM actions"),
            "audit_rows": q("SELECT COUNT(*) FROM audit_log"),
            "inter_annotator": self.inter_annotator(),
        }

    def model_history(self):
        return [dict(r) for r in self.conn.execute("SELECT * FROM model_versions ORDER BY version")]

    def audit_tail(self, limit=25):
        return [dict(r) for r in self.conn.execute("SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,))]

    def export_audit(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM audit_log ORDER BY seq")]
