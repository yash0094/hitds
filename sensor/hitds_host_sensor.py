#!/usr/bin/env python3
"""HITDS host sensor: sends a snapshot of THIS machine (connections, listening ports, process names, CPU,
memory) to the HITDS server every few seconds. The server applies the detection rules, so rules can change
without touching this machine.

    pip install -r sensor/requirements.txt
    python sensor/hitds_host_sensor.py --dry-run                     # show what would be sent, send nothing
    $env:HITDS_API_KEY="<key>"                                         # same value as HITDS_API_KEY on the server
    python sensor/hitds_host_sensor.py --server http://127.0.0.1:5000 --once
    python sensor/hitds_host_sensor.py --server https://hitds.onrender.com --interval 5

Run it from a normal PowerShell window. Some process owners are only visible to an Administrator window;
the sensor skips what it cannot read and says how many it skipped.
It only reads local network and process information. It does not capture packet contents and changes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
import urllib.error
import urllib.request

try:
    import psutil
except ImportError:  # pragma: no cover
    sys.exit("psutil is required:  pip install -r sensor/requirements.txt")

LIMITS = {"established": 300, "syn_sent": 500, "listening": 200, "processes": 800}
_names: dict[int, str] = {}


def pname(pid: int | None) -> str:
    if not pid:
        return ""
    if pid not in _names:
        try:
            _names[pid] = psutil.Process(pid).name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            _names[pid] = ""
    return _names[pid]


def snapshot() -> dict:
    """One reading. Never raises: whatever cannot be read is left out."""
    est, syn, lst, skipped = [], [], [], 0
    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, OSError):
        conns, skipped = [], 1
    for c in conns:
        if c.status == psutil.CONN_ESTABLISHED and c.raddr:
            est.append({"pid": c.pid or 0, "proc": pname(c.pid), "laddr": c.laddr.ip if c.laddr else "",
                        "lport": c.laddr.port if c.laddr else 0, "raddr": c.raddr.ip, "rport": c.raddr.port})
        elif c.status == psutil.CONN_SYN_SENT and c.raddr:
            syn.append({"raddr": c.raddr.ip, "rport": c.raddr.port})
        elif c.status == psutil.CONN_LISTEN and c.laddr:
            lst.append({"pid": c.pid or 0, "proc": pname(c.pid), "port": c.laddr.port})
    procs = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            procs.append({"pid": p.info["pid"], "name": p.info["name"] or ""})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            skipped += 1
    if len(_names) > 5000:
        _names.clear()
    return {"host": socket.gethostname(), "os": platform.system(),
            "cpu_percent": psutil.cpu_percent(interval=0.5), "memory_percent": psutil.virtual_memory().percent,
            "total_connections": len(conns), "established": est[:LIMITS["established"]],
            "syn_sent": syn[:LIMITS["syn_sent"]], "listening": lst[:LIMITS["listening"]],
            "processes": procs[:LIMITS["processes"]], "skipped": skipped}


def post(server: str, key: str, body: dict, timeout: float = 60.0) -> dict:
    """Timeout is long because a free Render instance can take about a minute to wake up."""
    req = urllib.request.Request(server.rstrip("/") + "/api/host/telemetry", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", help="HITDS base URL, e.g. https://hitds.onrender.com")
    ap.add_argument("--key", default=os.environ.get("HITDS_API_KEY", ""), help="API key (or set HITDS_API_KEY)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between snapshots (default 5)")
    ap.add_argument("--once", action="store_true", help="send one snapshot and exit")
    ap.add_argument("--dry-run", action="store_true", help="print a summary of one snapshot; send nothing")
    a = ap.parse_args(argv)
    if not a.dry_run and not (a.server and a.key):
        ap.error("--server and a key (--key or HITDS_API_KEY) are required unless --dry-run")
    while True:
        s = snapshot()
        line = (f"{time.strftime('%H:%M:%S')} sockets={s['total_connections']} established={len(s['established'])} "
                f"listening={len(s['listening'])} processes={len(s['processes'])} cpu={s['cpu_percent']:.0f}% "
                f"mem={s['memory_percent']:.0f}%" + (f" skipped={s['skipped']}" if s["skipped"] else ""))
        if a.dry_run:
            print(line)
            return 0
        try:
            r = post(a.server, a.key, s)
            print(line, "->", f"alerts={r.get('alerts')} new={r.get('new')} repeated={r.get('repeated')}")
        except urllib.error.HTTPError as e:
            print(line, f"-> server said {e.code} {e.reason} (401 = wrong key; set HITDS_API_KEY on the server)")
        except (OSError, urllib.error.URLError) as e:
            print(line, "-> could not reach server:", e)
        if a.once:
            return 0
        time.sleep(max(1.0, a.interval))


if __name__ == "__main__":
    sys.exit(main())
