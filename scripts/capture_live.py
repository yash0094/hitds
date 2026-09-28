"""Collector: turn real traffic into flows and push them to a running HITDS server.

Only capture traffic on networks you own or are authorised to monitor (your isolated
VirtualBox/VMware test-bed from the proposal). Packet capture needs admin rights and, on Windows, Npcap.

Flow extraction uses CICFlowMeter, which produces the same feature family as CIC-IDS2017:
    pip install cicflowmeter            (Python port; CLI: cicflowmeter)

Modes
  1) PCAP file -> flows -> HITDS
       python scripts/capture_live.py pcap --pcap lab_capture.pcap
  2) follow a CSV that CICFlowMeter is writing live, send new rows every few seconds
       cicflowmeter -i "Ethernet" -c live_flows.csv          (separate terminal, as admin)
       python scripts/capture_live.py follow --csv live_flows.csv
  3) send an existing CSV (Java CICFlowMeter output, or a CIC-IDS2017 day file for testing)
       python scripts/capture_live.py send --csv flows.csv
Add --eve /var/log/suricata/eve.json to attach Suricata signature hits as analyst evidence.

Server + key:  --url http://127.0.0.1:5000  --key $HITDS_API_KEY   (defaults read from env HITDS_URL / HITDS_API_KEY)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.collect import attach_signatures, load_suricata_eve  # noqa: E402


def post(df: pd.DataFrame, url: str, key: str, batch: int = 500) -> None:
    for s in range(0, len(df), batch):
        chunk = df.iloc[s: s + batch].replace([float("inf"), float("-inf")], None)
        body = json.loads(chunk.to_json(orient="records"))
        r = requests.post(url.rstrip("/") + "/api/ingest", json={"flows": body}, headers={"X-API-Key": key}, timeout=60)
        if r.status_code != 200:
            print("server rejected batch:", r.status_code, r.text[:300])
            return
        rep = r.json()
        print(f"sent {len(chunk):4d} flows · coverage {rep.get('coverage')} · ingested {rep.get('ingested')}")


def prep(df: pd.DataFrame, eve: str | None) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    if eve:
        idx = load_suricata_eve(eve)
        ren = {c: c for c in df.columns}
        for c in df.columns:
            n = c.lower().replace(" ", "_")
            if n in ("src_ip", "source_ip"):
                ren[c] = "src_ip"
            if n in ("dst_ip", "destination_ip"):
                ren[c] = "dst_ip"
        df = attach_signatures(df.rename(columns=ren), idx)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pcap", "follow", "send"])
    ap.add_argument("--pcap")
    ap.add_argument("--csv")
    ap.add_argument("--eve", default=None)
    ap.add_argument("--url", default=os.environ.get("HITDS_URL", "http://127.0.0.1:5000"))
    ap.add_argument("--key", default=os.environ.get("HITDS_API_KEY", ""))
    ap.add_argument("--every", type=float, default=3.0, help="seconds between polls in follow mode")
    ap.add_argument("--limit", type=int, default=None, help="send at most this many rows (send/pcap modes)")
    a = ap.parse_args()
    if not a.key:
        sys.exit("set HITDS_API_KEY (same value as on the server) or pass --key")

    if a.mode == "pcap":
        out = Path(a.pcap).with_suffix(".flows.csv")
        print("extracting flows with cicflowmeter ...")
        subprocess.run(["cicflowmeter", "-f", a.pcap, "-c", str(out)], check=True)
        post(prep(pd.read_csv(out, low_memory=False, nrows=a.limit), a.eve), a.url, a.key)
    elif a.mode == "send":
        post(prep(pd.read_csv(a.csv, low_memory=False, encoding="latin-1", nrows=a.limit), a.eve), a.url, a.key)
    else:
        seen = 0
        print(f"following {a.csv} (Ctrl+C to stop)")
        while True:
            try:
                df = pd.read_csv(a.csv, low_memory=False)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                time.sleep(a.every)
                continue
            if len(df) > seen:
                post(prep(df.iloc[seen:].copy(), a.eve), a.url, a.key)
                seen = len(df)
            time.sleep(a.every)
