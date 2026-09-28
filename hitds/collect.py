"""Collecting real traffic for scoring: CICFlowMeter CSVs, JSON flows over the API, and
Suricata/Zeek-style signature hits.

CICFlowMeter (Java v4 and the Python `cicflowmeter` package) write *abbreviated* column
names ("Tot Fwd Pkts", "tot_fwd_pkts") while the CIC-IDS2017 CSVs use long names
("Total Fwd Packets"). A model trained on one and fed the other silently receives zeros,
so every incoming column is normalised and mapped here, and the coverage is reported.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import get_logger

log = get_logger("hitds.collect")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# abbreviated CICFlowMeter name (normalised) -> CIC-IDS2017 column name
_ABBREV = {
    "dstport": "Destination Port", "flowduration": "Flow Duration",
    "totfwdpkts": "Total Fwd Packets", "totbwdpkts": "Total Backward Packets",
    "totlenfwdpkts": "Total Length of Fwd Packets", "totlenbwdpkts": "Total Length of Bwd Packets",
    "fwdpktlenmax": "Fwd Packet Length Max", "fwdpktlenmin": "Fwd Packet Length Min",
    "fwdpktlenmean": "Fwd Packet Length Mean", "fwdpktlenstd": "Fwd Packet Length Std",
    "bwdpktlenmax": "Bwd Packet Length Max", "bwdpktlenmin": "Bwd Packet Length Min",
    "bwdpktlenmean": "Bwd Packet Length Mean", "bwdpktlenstd": "Bwd Packet Length Std",
    "flowbytss": "Flow Bytes/s", "flowpktss": "Flow Packets/s",
    "flowiatmean": "Flow IAT Mean", "flowiatstd": "Flow IAT Std", "flowiatmax": "Flow IAT Max",
    "flowiatmin": "Flow IAT Min",
    "fwdiattot": "Fwd IAT Total", "fwdiatmean": "Fwd IAT Mean", "fwdiatstd": "Fwd IAT Std",
    "fwdiatmax": "Fwd IAT Max", "fwdiatmin": "Fwd IAT Min",
    "bwdiattot": "Bwd IAT Total", "bwdiatmean": "Bwd IAT Mean", "bwdiatstd": "Bwd IAT Std",
    "bwdiatmax": "Bwd IAT Max", "bwdiatmin": "Bwd IAT Min",
    "fwdpshflags": "Fwd PSH Flags", "bwdpshflags": "Bwd PSH Flags",
    "fwdurgflags": "Fwd URG Flags", "bwdurgflags": "Bwd URG Flags",
    "fwdheaderlen": "Fwd Header Length", "bwdheaderlen": "Bwd Header Length",
    "fwdpktss": "Fwd Packets/s", "bwdpktss": "Bwd Packets/s",
    "pktlenmin": "Min Packet Length", "pktlenmax": "Max Packet Length", "pktlenmean": "Packet Length Mean",
    "pktlenstd": "Packet Length Std", "pktlenvar": "Packet Length Variance",
    "finflagcnt": "FIN Flag Count", "synflagcnt": "SYN Flag Count", "rstflagcnt": "RST Flag Count",
    "pshflagcnt": "PSH Flag Count", "ackflagcnt": "ACK Flag Count", "urgflagcnt": "URG Flag Count",
    "cweflagcount": "CWE Flag Count", "cweflagcnt": "CWE Flag Count", "eceflagcnt": "ECE Flag Count",
    "downupratio": "Down/Up Ratio", "pktsizeavg": "Average Packet Size",
    "fwdsegsizeavg": "Avg Fwd Segment Size", "bwdsegsizeavg": "Avg Bwd Segment Size",
    "fwdbytsbavg": "Fwd Avg Bytes/Bulk", "fwdpktsbavg": "Fwd Avg Packets/Bulk", "fwdblkrateavg": "Fwd Avg Bulk Rate",
    "bwdbytsbavg": "Bwd Avg Bytes/Bulk", "bwdpktsbavg": "Bwd Avg Packets/Bulk", "bwdblkrateavg": "Bwd Avg Bulk Rate",
    "subflowfwdpkts": "Subflow Fwd Packets", "subflowfwdbyts": "Subflow Fwd Bytes",
    "subflowbwdpkts": "Subflow Bwd Packets", "subflowbwdbyts": "Subflow Bwd Bytes",
    "initfwdwinbyts": "Init_Win_bytes_forward", "initbwdwinbyts": "Init_Win_bytes_backward",
    "fwdactdatapkts": "act_data_pkt_fwd", "fwdsegsizemin": "min_seg_size_forward",
    "activemean": "Active Mean", "activestd": "Active Std", "activemax": "Active Max", "activemin": "Active Min",
    "idlemean": "Idle Mean", "idlestd": "Idle Std", "idlemax": "Idle Max", "idlemin": "Idle Min",
}
_META = {"srcip": "src_ip", "sourceip": "src_ip", "dstip": "dst_ip", "destinationip": "dst_ip",
         "srcport": "src_port", "sourceport": "src_port", "timestamp": "timestamp", "protocol": "protocol",
         "flowid": "flow_id", "label": "label_raw"}


def map_columns(columns: list[str], model_columns: list[str]) -> tuple[dict, dict]:
    """Return (rename map, report). Accepts CIC-IDS2017 long names, CICFlowMeter v4 and python-cicflowmeter."""
    exact = {_norm(c): c for c in model_columns}
    rename, unknown = {}, []
    for c in columns:
        n = _norm(c)
        if n in exact:
            rename[c] = exact[n]
        elif n in _ABBREV and _ABBREV[n] in model_columns:
            rename[c] = _ABBREV[n]
        elif n in _META:
            rename[c] = _META[n]
        else:
            unknown.append(c)
    mapped = set(v for v in rename.values() if v in model_columns)
    report = {"model_features": len(model_columns), "mapped": len(mapped),
              "coverage": round(len(mapped) / max(1, len(model_columns)), 4),
              "missing_model_features": sorted(set(model_columns) - mapped), "ignored_columns": unknown}
    return rename, report


def normalise_flows(df: pd.DataFrame, model_columns: list[str], min_coverage: float = 0.8) -> tuple[pd.DataFrame, dict]:
    """Map/clean an incoming flow table to the model's raw feature columns (+ meta columns)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename, report = map_columns(list(df.columns), model_columns)
    df = df.rename(columns=rename)
    df = df.loc[:, ~df.columns.duplicated()]
    if report["coverage"] < min_coverage:
        raise ValueError(f"only {report['coverage']:.0%} of the model's features found in the upload; "
                         f"missing e.g. {report['missing_model_features'][:8]}")
    feats = pd.DataFrame(index=df.index)
    for c in model_columns:
        feats[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    bad = feats.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    report["rows_in"] = int(len(df))
    report["rows_with_missing_values_filled"] = int(bad.sum())
    # missing features for a row are filled with the training median (set by the caller) - never silently zero
    meta = pd.DataFrame(index=df.index)
    for m in ("src_ip", "dst_ip", "src_port", "timestamp", "protocol", "label_raw", "signature"):
        if m in df:
            meta[m] = df[m].astype(str)
    return pd.concat([feats, meta], axis=1), report


def read_upload(content: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".json"):
        data = json.loads(content.decode("utf-8"))
        return pd.DataFrame(data if isinstance(data, list) else data.get("flows", []))
    return pd.read_csv(io.BytesIO(content), encoding="latin-1", low_memory=False)


# --------------------------------------------------------------------------- signature evidence (Suricata EVE)
def load_suricata_eve(path: str | Path) -> dict[tuple, list[str]]:
    """Index Suricata eve.json alerts by (src_ip, dst_ip, dst_port) -> signature names.
    Signature hits are shown to the analyst as evidence next to the ML score (proposal: signature matching)."""
    idx: dict[tuple, list[str]] = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") != "alert":
                continue
            key = (ev.get("src_ip"), ev.get("dest_ip"), str(ev.get("dest_port")))
            idx.setdefault(key, []).append(ev["alert"].get("signature", "?"))
    return idx


def attach_signatures(flows: pd.DataFrame, eve_index: dict) -> pd.DataFrame:
    if not eve_index or "src_ip" not in flows:
        return flows
    port = flows.get("Destination Port", pd.Series([""] * len(flows))).astype("Int64").astype(str)
    flows["signature"] = ["; ".join(eve_index.get((s, d, p), [])) for s, d, p in
                          zip(flows["src_ip"], flows.get("dst_ip", ""), port)]
    return flows
