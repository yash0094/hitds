"""HITDS analyst console + REST API (Flask).

Development:   python -m app.server --dataset synthetic --reset
Production:    python serve.py            (waitress WSGI server; settings from environment variables)

Environment variables (all optional in development):
  HITDS_DATASET        dataset/model folder to serve (default cicids2017)
  HITDS_USERS          "alice:password1,bob:password2" - analyst logins (default: analyst / hitds-demo, dev only)
  HITDS_SECRET_KEY     Flask session secret (random per start if unset -> everyone is logged out on restart)
  HITDS_API_KEY        key for machine-to-machine ingest (POST /api/ingest with header X-API-Key)
  HITDS_AUTOSTART      1 = start the traffic replay immediately
  HITDS_RATE           replay events per second
  HITDS_RESET          1 = wipe the demo database on start
"""
from __future__ import annotations

import argparse
import functools
import os
import secrets
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.collect import read_upload  # noqa: E402
from hitds.common import get_logger, load_config, seed_everything, use_dataset  # noqa: E402
from hitds.data import load_split  # noqa: E402
from hitds.db import Store  # noqa: E402
from hitds.engine import Engine  # noqa: E402
from hitds.pipeline import load_live  # noqa: E402
from hitds.stream import TrafficReplay  # noqa: E402

log = get_logger("hitds.app")


class Streamer:
    def __init__(self, engine: Engine, replay: TrafficReplay, rate: float):
        self.engine, self.replay, self.rate = engine, replay, rate
        self.running = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        carry = 0.0
        while True:
            time.sleep(1.0)
            if not self.running:
                continue
            carry += self.rate
            n, carry = int(carry), carry - int(carry)
            if n:
                try:
                    self.engine.ingest(self.replay.next_batch(n))
                except Exception as e:  # keep the demo alive; the error is in the server log
                    log.exception("ingest failed: %s", e)


def _users_from_env() -> dict[str, str]:
    raw = os.environ.get("HITDS_USERS", "").strip()
    if not raw:
        log.warning("HITDS_USERS not set - using development login analyst / hitds-demo. Set it before deploying.")
        raw = "analyst:hitds-demo"
    users = {}
    for pair in raw.split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            users[u.strip()] = generate_password_hash(p.strip())
    return users


def create_app(dataset: str | None = None, reset: bool | None = None, rate: float | None = None,
               autostart: bool | None = None, config_path: str | None = None, testing: bool = False) -> Flask:
    env = os.environ.get
    dataset = dataset or env("HITDS_DATASET", "cicids2017")
    reset = reset if reset is not None else env("HITDS_RESET") == "1"
    autostart = autostart if autostart is not None else env("HITDS_AUTOSTART") == "1"
    base = load_config(config_path or env("HITDS_CONFIG"))
    seed_everything(base["seed"])
    models_root = Path(base["paths"]["models"])
    if not (models_root / dataset / "registry.json").exists():
        available = sorted(p.parent.name for p in models_root.glob("*/registry.json"))
        if not available:
            raise FileNotFoundError(f"no trained model in {models_root} - run scripts/03_train.py "
                                    "(and scripts/06_export_deploy.py for Docker/Render)")
        log.warning("no model for dataset %r; using %r instead (set HITDS_DATASET to silence this)",
                    dataset, available[0])
        dataset = available[0]
    cfg = use_dataset(base, dataset)
    if reset:
        for suffix in ("", "-wal", "-shm"):
            Path(cfg["paths"]["db"] + suffix).unlink(missing_ok=True)
    rv = cfg.get("review", {})
    store = Store(cfg["paths"]["db"], rv.get("second_review_below", 0.5), rv.get("distinct_second_reviewer", True))
    engine = Engine(cfg, load_live(cfg["paths"]["models"]), store)
    replay = TrafficReplay(load_split(cfg["paths"]["processed"], "test"), cfg["stream"]["n_hosts"], base["seed"])
    streamer = Streamer(engine, replay, float(rate or env("HITDS_RATE", cfg["stream"]["rate_per_sec"])))
    streamer.running = autostart
    users = _users_from_env()
    api_key = env("HITDS_API_KEY")

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(SECRET_KEY=env("HITDS_SECRET_KEY") or secrets.token_hex(32), SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Lax", MAX_CONTENT_LENGTH=25 * 1024 * 1024, TESTING=testing,
                      engine=engine, store=store, streamer=streamer, dataset=dataset)
    store.audit_event("server_start", "system", {"dataset": dataset, "model_version": engine.bundle.version})

    # ------------------------------------------------------------------ auth
    def login_required(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            if "user" not in session:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "login required"}), 401
                return redirect(url_for("login", next=request.path))
            # CSRF: state-changing browser calls must carry the custom header (SameSite cookie + header check)
            if request.method == "POST" and request.headers.get("X-Requested-With") != "hitds":
                return jsonify({"error": "missing X-Requested-With header"}), 400
            return fn(*a, **k)
        return wrapper

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            u, p = request.form.get("username", ""), request.form.get("password", "")
            if u in users and check_password_hash(users[u], p):
                session.clear()
                session["user"] = u
                store.audit_event("login", f"analyst:{u}", {"ip": request.remote_addr})
                return redirect(request.args.get("next") or url_for("index"))
            error = "Wrong username or password"
            store.audit_event("login_failed", f"analyst:{u}", {"ip": request.remote_addr})
        return render_template("login.html", error=error, dataset=dataset)

    @app.get("/logout")
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "dataset": dataset, "model_version": engine.bundle.version})

    # ------------------------------------------------------------------ UI
    @app.get("/")
    @login_required
    def index():
        return render_template("index.html", dataset=dataset, user=session["user"])

    # ------------------------------------------------------------------ analyst API
    @app.get("/api/queue")
    @login_required
    def queue():
        return jsonify({"alerts": engine.queue(int(request.args.get("limit", 40)), session["user"]),
                        "hosts": engine.hmm.ranking()[:12], "policy": cfg.get("hmm", {}).get("policy")})

    @app.get("/api/alert/<int:aid>")
    @login_required
    def alert(aid):
        d = engine.alert_detail(aid)
        return (jsonify(d), 200) if d else (jsonify({"error": "not found"}), 404)

    @app.post("/api/alert/<int:aid>/verdict")
    @login_required
    def verdict(aid):
        j = request.get_json(force=True)
        if j.get("verdict") not in ("TP", "FP", "UNKNOWN"):
            return jsonify({"error": "verdict must be TP, FP or UNKNOWN"}), 400
        label = j.get("label") or None
        if label and label not in engine.bundle.classes:
            return jsonify({"error": f"unknown class {label}"}), 400
        try:
            res = engine.verdict(aid, session["user"], j["verdict"], label, float(j.get("confidence", 1.0)),
                                 float(j.get("seconds", 0)), j.get("action") or None, bool(j.get("revealed", True)))
        except (KeyError, ValueError) as e:
            return jsonify({"error": str(e)}), 409
        return jsonify(res)

    @app.get("/api/metrics")
    @login_required
    def metrics():
        m = store.metrics()
        m.update(model_version=engine.bundle.version, classes=engine.bundle.classes,
                 retraining=engine.retraining, last_retrain=engine.last_retrain,
                 pending_feedback=len(store.unused_feedback()), retrain_every=cfg["feedback"]["retrain_every"],
                 streaming=streamer.running, rate=streamer.rate, triage=engine.bundle.triage.stats,
                 latency_ms=engine.last_latency_ms, model_history=store.model_history(), dataset=dataset,
                 user=session["user"], policy=cfg.get("hmm", {}).get("policy"))
        return jsonify(m)

    @app.get("/api/feed")
    @login_required
    def feed():
        return jsonify(store.recent_alerts(int(request.args.get("limit", 25))))

    @app.get("/api/audit")
    @login_required
    def audit():
        return jsonify({"verify": store.verify_chain(), "tail": store.audit_tail(int(request.args.get("limit", 20)))})

    @app.get("/api/audit/export")
    @login_required
    def audit_export():
        import json
        store.audit_event("audit_exported", f"analyst:{session['user']}", {})
        body = json.dumps({"verify": store.verify_chain(), "rows": store.export_audit()}, indent=1, default=str)
        return Response(body, mimetype="application/json",
                        headers={"Content-Disposition": f"attachment; filename=hitds_audit_{dataset}.json"})

    @app.post("/api/stream")
    @login_required
    def stream():
        j = request.get_json(force=True)
        if "running" in j:
            streamer.running = bool(j["running"])
        if "rate" in j:
            streamer.rate = min(100.0, max(0.2, float(j["rate"])))
        return jsonify({"running": streamer.running, "rate": streamer.rate})

    @app.post("/api/inject")
    @login_required
    def inject():
        n = min(500, int(request.get_json(force=True).get("n", 20)))
        return jsonify({"ingested": len(engine.ingest(replay.next_batch(n)))})

    @app.post("/api/upload")
    @login_required
    def upload():
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "no file"}), 400
        try:
            report = engine.ingest_flows(read_upload(f.read(), f.filename), source=f"upload:{session['user']}")
        except ValueError as e:
            return jsonify({"error": str(e)}), 422
        return jsonify(report)

    @app.post("/api/retrain")
    @login_required
    def retrain():
        if engine.retraining:
            return jsonify({"status": "already running"}), 409
        threading.Thread(target=engine.retrain, daemon=True).start()
        return jsonify({"status": "started"})

    # ------------------------------------------------------------------ machine ingest (collectors)
    @app.post("/api/ingest")
    def ingest():
        if not api_key or not secrets.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            return jsonify({"error": "invalid or missing X-API-Key (set HITDS_API_KEY on the server)"}), 401
        import pandas as pd
        j = request.get_json(force=True)
        flows = j.get("flows", j) if isinstance(j, dict) else j
        try:
            report = engine.ingest_flows(pd.DataFrame(flows), source="api")
        except ValueError as e:
            return jsonify({"error": str(e)}), 422
        return jsonify(report)

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--reset", action="store_true", help="wipe the demo database (alerts, verdicts, audit log)")
    ap.add_argument("--autostart", action="store_true", help="start the traffic replay immediately")
    a = ap.parse_args()
    create_app(a.dataset, a.reset or None, a.rate, a.autostart or None).run(host=a.host, port=a.port, threaded=True)
