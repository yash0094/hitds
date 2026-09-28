"""Dataset collection, cleaning, label harmonisation and leakage-safe splitting.

Every dataset ends up in the same shape:
    numeric feature columns (float32)  +  `label` (attack family, str)  +  `label_raw` (original label)

Precision rules applied here (each step is counted and written to a data card):
  1. downloads are pinned by SHA-256 in data/raw/<ds>/MANIFEST.json and re-verified on every run
  2. +/-inf -> NaN; rows with NaN in any feature are dropped (not zero-filled - a missing
     Flow Bytes/s is not the same as zero bytes)
  3. physically impossible rows are dropped (negative durations / counts / lengths - a known
     CICFlowMeter artefact in CIC-IDS2017)
  4. exact duplicate rows are removed BEFORE splitting. CIC-IDS2017 contains hundreds of thousands
     of duplicate flows; if duplicates straddle train/test the scores are inflated (leakage)
  5. feature vectors that appear with more than one label are removed (label noise)
  6. constant columns and pure identifiers are dropped
  7. large classes are capped AFTER cleaning; rare classes are kept whole
  8. the split is stratified 70/15/15 and a hash check proves no feature row is in two splits
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import get_logger

log = get_logger("hitds.data")

# --------------------------------------------------------------------------- labels
CIC_FAMILY = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce", "SSH-Patator": "BruteForce",
    "Bot": "Bot",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}


def cic_family(raw: str) -> str:
    raw = str(raw).strip()
    if raw in CIC_FAMILY:
        return CIC_FAMILY[raw]
    # "Web Attack � Brute Force", "Web Attack – XSS", "Web Attack - Sql Injection" (encoding varies by mirror)
    if raw.lower().startswith("web attack"):
        return "WebAttack"
    return raw


def unsw_family(raw: str) -> str:
    raw = str(raw).strip()
    if raw == "" or raw.lower() in {"normal", "nan"}:
        return "BENIGN"
    if raw.lower().startswith("backdoor"):   # "Backdoor" and "Backdoors" both appear
        return "Backdoor"
    return raw


# Columns that must never be non-negative-violating (CIC names, after strip)
NON_NEGATIVE_HINTS = re.compile(r"(Duration|Packets|Length|Bytes$|Count|Total|Size|IAT (Max|Min|Mean|Total))",
                                re.IGNORECASE)
IDENTIFIER_COLS = {"Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP", "Source Port", "Src Port",
                   "Timestamp", "id", "srcip", "dstip", "sport", "Stime", "Ltime"}


# --------------------------------------------------------------------------- download + integrity
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(repo: str, files: list[str], out_dir: str | Path, revision: str | None = None) -> dict:
    """Download files from a Hugging Face dataset repo, then pin/verify them in MANIFEST.json."""
    from huggingface_hub import hf_hub_download

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"repo": repo, "files": {}}
    for fname in files:
        log.info("downloading %s/%s", repo, fname)
        p = Path(hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=str(out_dir),
                                 revision=revision))
        digest, size = sha256_file(p), p.stat().st_size
        old = manifest["files"].get(fname)
        if old and old["sha256"] != digest:
            log.warning("%s changed upstream since it was pinned (sha256 %s -> %s). Results may not be comparable "
                        "with earlier runs.", fname, old["sha256"][:12], digest[:12])
        manifest["files"][fname] = {"sha256": digest, "bytes": size}
        log.info("  ok %.1f MB sha256=%s", size / 1e6, digest[:16])
    manifest["repo"], manifest["revision"] = repo, revision or "main"
    man_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_manifest(raw_dir: str | Path) -> dict:
    raw_dir = Path(raw_dir)
    man_path = raw_dir / "MANIFEST.json"
    if not man_path.exists():
        return {"ok": None, "reason": "no MANIFEST.json (files not downloaded by scripts/01_download.py)"}
    man = json.loads(man_path.read_text())
    bad = [f for f, m in man["files"].items()
           if not (raw_dir / f).exists() or sha256_file(raw_dir / f) != m["sha256"]]
    return {"ok": not bad, "mismatched": bad, "repo": man.get("repo")}


# --------------------------------------------------------------------------- cleaning
class CleaningLog(dict):
    def step(self, name: str, before: int, after: int):
        self[name] = {"before": int(before), "after": int(after), "removed": int(before - after)}
        if before != after:
            log.info("  %-38s %9d -> %9d  (-%d)", name, before, after, before - after)


def _to_float(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for c in feature_cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].astype(np.float64).replace([np.inf, -np.inf], np.nan)
    return df


def clean_frame(df: pd.DataFrame, feature_cols: list[str], clog: CleaningLog, prefix: str = "") -> pd.DataFrame:
    """Steps 2-5 on one frame. `label` must already exist."""
    n = len(df)
    df = _to_float(df, feature_cols)
    df = df.dropna(subset=feature_cols)
    clog.step(prefix + "drop NaN/inf rows", n, len(df))
    n = len(df)
    nonneg = [c for c in feature_cols if NON_NEGATIVE_HINTS.search(c) and not c.lower().startswith("init")]
    if nonneg:
        df = df[(df[nonneg] >= 0).all(axis=1)]
    clog.step(prefix + "drop impossible negative values", n, len(df))
    n = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["label"])
    clog.step(prefix + "drop exact duplicate rows", n, len(df))
    n = len(df)
    conflict = df.duplicated(subset=feature_cols, keep=False)
    df = df[~conflict]
    clog.step(prefix + "drop rows with conflicting labels", n, len(df))
    return df


def finalize(df: pd.DataFrame, max_per_class: int, max_benign: int, seed: int, clog: CleaningLog) -> pd.DataFrame:
    feats = [c for c in df.columns if c not in ("label", "label_raw")]
    const = [c for c in feats if df[c].nunique(dropna=False) <= 1]
    df = df.drop(columns=const)
    clog["dropped_constant_columns"] = const
    parts = []
    for lab, g in df.groupby("label"):
        cap = max_benign if lab == "BENIGN" else max_per_class
        parts.append(g.sample(n=min(len(g), cap), random_state=seed))
    n = len(df)
    df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    clog.step("cap large classes (after cleaning)", n, len(df))
    feats = [c for c in df.columns if c not in ("label", "label_raw")]
    df[feats] = df[feats].astype(np.float32)
    return df


# --------------------------------------------------------------------------- loaders
def load_cicids2017(raw_dir: str | Path, cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, CleaningLog]:
    dcfg = cfg["datasets"]["cicids2017"]
    clog = CleaningLog()
    frames, raw_counts = [], {}
    for fname in dcfg["files"]:
        f = Path(raw_dir) / fname
        if not f.exists():
            log.warning("missing %s (run scripts/01_download.py)", f.name)
            continue
        log.info("reading %s", f.name)
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]                     # 'Fwd Header Length' appears twice
        df = df.drop(columns=[c for c in df.columns if c in IDENTIFIER_COLS or c.endswith(".1")])
        df = df.drop(columns=[c for c in dcfg.get("drop_features", []) if c in df.columns])
        df["label_raw"] = df["Label"].astype(str).str.strip()
        df["label"] = df["label_raw"].map(cic_family)
        df = df.drop(columns=["Label"])
        for k, v in df["label_raw"].value_counts().items():
            raw_counts[k] = raw_counts.get(k, 0) + int(v)
        feats = [c for c in df.columns if c not in ("label", "label_raw")]
        df = clean_frame(df, feats, clog, prefix=f"[{fname[:22]}] ")
        # float32 early keeps the 2.8M-row dataset inside 16 GB
        df[feats] = df[feats].astype(np.float32)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No CIC-IDS2017 files found in " + str(raw_dir))
    df = pd.concat(frames, ignore_index=True)
    feats = [c for c in df.columns if c not in ("label", "label_raw")]
    n = len(df)
    df = df.drop_duplicates(subset=feats + ["label"])
    clog.step("drop duplicates across days", n, len(df))
    n = len(df)
    df = df[~df.duplicated(subset=feats, keep=False)]
    clog.step("drop label conflicts across days", n, len(df))
    clog["raw_label_counts"] = raw_counts
    clog["clean_label_counts"] = df["label"].value_counts().to_dict()
    return finalize(df, dcfg["max_per_class"], dcfg["max_benign"], seed, clog), clog


def load_unsw_nb15(raw_dir: str | Path, cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, CleaningLog]:
    dcfg = cfg["datasets"]["unsw_nb15"]
    clog = CleaningLog()
    frames = [pd.read_csv(Path(raw_dir) / f, low_memory=False) for f in dcfg["files"] if (Path(raw_dir) / f).exists()]
    if not frames:
        raise FileNotFoundError("No UNSW-NB15 files found in " + str(raw_dir))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["label_raw"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["label"] = df["label_raw"].map(unsw_family)
    clog["raw_label_counts"] = df["label_raw"].value_counts().to_dict()
    # one-hot the small categorical columns (top 15 values each, rest -> 'other')
    for col in ["proto", "service", "state"]:
        if col in df.columns:
            top = df[col].value_counts().index[:15]
            df[col] = df[col].where(df[col].isin(top), "other")
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=np.float32)], axis=1)
    drop = {"attack_cat", "proto", "service", "state"} | IDENTIFIER_COLS | {"dsport"}
    df = df.drop(columns=[c for c in df.columns if c in drop])
    df = df.drop(columns=[c for c in df.columns if c not in ("label", "label_raw") and df[c].dtype == object])
    # the binary 'label' column of UNSW was overwritten by our family label above - that is intended
    feats = [c for c in df.columns if c not in ("label", "label_raw")]
    df = clean_frame(df, feats, clog)
    clog["clean_label_counts"] = df["label"].value_counts().to_dict()
    return finalize(df, dcfg["max_per_class"], dcfg["max_benign"], seed, clog), clog


def load_nslkdd(raw_dir: str | Path, cfg: dict, seed: int = 42) -> tuple[pd.DataFrame, CleaningLog]:
    """Legacy baseline. The HF mirror stores `class` numerically; verify the mapping on first run."""
    clog = CleaningLog()
    frames = [pd.read_csv(Path(raw_dir) / f) for f in cfg["datasets"]["nslkdd"]["files"]
              if (Path(raw_dir) / f).exists()]
    if not frames:
        raise FileNotFoundError("No NSL-KDD files found in " + str(raw_dir))
    df = pd.concat(frames, ignore_index=True)
    cls = df["class"]
    if cls.dtype == object:
        df["label"] = np.where(cls.str.lower().str.strip() == "normal", "BENIGN", "ATTACK")
    else:
        counts = cls.value_counts().to_dict()
        log.info("NSL-KDD class value counts: %s", counts)
        normal_value = cfg["datasets"]["nslkdd"].get("normal_value", 0)
        df["label"] = np.where(cls == normal_value, "BENIGN", "ATTACK")
    df["label_raw"] = cls.astype(str)
    for col in ["protocol_type", "service", "flag"]:
        if col in df.columns and df[col].dtype == object:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=np.float32)], axis=1).drop(columns=[col])
    df = df.drop(columns=["class"])
    feats = [c for c in df.columns if c not in ("label", "label_raw")]
    df = clean_frame(df, feats, clog)
    clog["clean_label_counts"] = df["label"].value_counts().to_dict()
    return finalize(df, 10**9, 10**9, seed, clog), clog


LOADERS = {"cicids2017": load_cicids2017, "unsw_nb15": load_unsw_nb15, "nslkdd": load_nslkdd}


# --------------------------------------------------------------------------- synthetic (offline / tests)
SYN_CLASSES = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Bot", "Infiltration", "Heartbleed"]
SYN_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Mean", "Bwd Packet Length Max",
    "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd PSH Flags", "SYN Flag Count", "RST Flag Count", "ACK Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Idle Mean", "Destination Port", "URG Flag Count",
]


def make_synthetic(n: int = 20000, seed: int = 42, n_features: int = 30) -> pd.DataFrame:
    """CIC-IDS2017-shaped synthetic flows (non-negative, heavy-tailed). ONLY for tests and offline smoke runs.
    Classes overlap on purpose so the classifier is sometimes unsure. Never report numbers from this data."""
    rng = np.random.default_rng(seed)
    props = np.array([0.62, 0.12, 0.08, 0.08, 0.04, 0.025, 0.02, 0.01, 0.005])
    counts = np.maximum((props / props.sum() * n).astype(int), 30)
    names = SYN_FEATURES[:n_features]
    centers = rng.normal(0, 1.0, size=(len(SYN_CLASSES), n_features))
    centers[0] = 0.0
    centers[2] = centers[1] + rng.normal(0, 0.35, n_features)       # DDoS ~ DoS
    centers[7] = centers[0] + rng.normal(0, 0.45, n_features)       # Infiltration ~ BENIGN
    centers[6] = centers[5] + rng.normal(0, 0.5, n_features)        # Bot ~ WebAttack
    rows, labels = [], []
    for k, (cls, c) in enumerate(zip(SYN_CLASSES, counts)):
        x = centers[k] + rng.normal(0, 0.55, size=(c, n_features))
        rows.append(np.expm1(np.abs(x) * 2.0))
        labels += [cls] * c
    df = pd.DataFrame(np.vstack(rows).astype(np.float32), columns=names)
    df["label"] = labels
    df["label_raw"] = labels
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- split + save
def row_hashes(df: pd.DataFrame) -> pd.Series:
    feats = df.drop(columns=[c for c in ("label", "label_raw") if c in df.columns])
    return pd.util.hash_pandas_object(feats, index=False)


def split_and_save(df: pd.DataFrame, out_dir: str | Path, cfg: dict, seed: int = 42,
                   holdout_class: str | None = None, clog: CleaningLog | None = None,
                   dataset: str = "") -> dict:
    """70/15/15 stratified split + leakage check. `holdout_class` is removed from train/val (simulated zero-day)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vc = df["label"].value_counts()
    rare = vc[vc < 10].index.tolist()
    if rare:
        log.warning("dropping classes with < 10 rows after cleaning (cannot be split 70/15/15): %s", rare)
        df = df[~df.label.isin(rare)]
    s = cfg["split"]
    train, rest = train_test_split(df, test_size=1 - s["train"], stratify=df["label"], random_state=seed)
    val, test = train_test_split(rest, test_size=s["test"] / (s["val"] + s["test"]), stratify=rest["label"],
                                 random_state=seed)
    if holdout_class:
        if holdout_class not in set(df.label):
            raise ValueError(f"holdout class {holdout_class!r} not in {sorted(set(df.label))}")
        moved = pd.concat([train[train.label == holdout_class], val[val.label == holdout_class]])
        train, val = train[train.label != holdout_class], val[val.label != holdout_class]
        test = pd.concat([test, moved])
        log.info("zero-day holdout: %s removed from train/val (%d rows moved to test)", holdout_class, len(moved))
    # leakage check: no identical feature row may appear in two splits
    h = {k: set(row_hashes(v)) for k, v in (("train", train), ("val", val), ("test", test))}
    overlap = {"train&val": len(h["train"] & h["val"]), "train&test": len(h["train"] & h["test"]),
               "val&test": len(h["val"] & h["test"])}
    if any(overlap.values()):
        raise RuntimeError(f"leakage: identical feature rows in more than one split {overlap}")
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.reset_index(drop=True).to_pickle(out_dir / f"{name}.pkl")
    meta = {
        "dataset": dataset,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts": {k: v["label"].value_counts().to_dict() for k, v in
                         (("train", train), ("val", val), ("test", test))},
        "n_features": int(df.shape[1] - 2),
        "features": [c for c in df.columns if c not in ("label", "label_raw")],
        "holdout_class": holdout_class,
        "leakage_overlap": overlap,
        "cleaning": dict(clog or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    write_data_card(meta, out_dir / "DATA_CARD.md")
    return meta


def write_data_card(meta: dict, path: Path):
    c = meta.get("cleaning", {})
    lines = [f"# Data card: {meta.get('dataset')}", "",
             f"Features: {meta['n_features']} · zero-day holdout: {meta.get('holdout_class')}", "",
             "## Cleaning steps", "| step | before | after | removed |", "|---|---|---|---|"]
    for k, v in c.items():
        if isinstance(v, dict) and "before" in v:
            lines.append(f"| {k} | {v['before']:,} | {v['after']:,} | {v['removed']:,} |")
    if c.get("dropped_constant_columns"):
        lines += ["", "Dropped constant columns: " + ", ".join(c["dropped_constant_columns"])]
    lines += ["", "## Class counts per split", "| class | train | val | test |", "|---|---|---|---|"]
    classes = sorted(set().union(*[set(v) for v in meta["class_counts"].values()]))
    for cl in classes:
        lines.append(f"| {cl} | " + " | ".join(str(meta["class_counts"][s].get(cl, 0)) for s in ("train", "val", "test"))
                     + " |")
    if c.get("raw_label_counts"):
        lines += ["", "## Raw labels before cleaning", "| raw label | rows |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in sorted(c["raw_label_counts"].items(), key=lambda t: -t[1])]
    lines += ["", f"Leakage check (identical feature rows shared between splits): {meta['leakage_overlap']}"]
    path.write_text("\n".join(lines))


def load_split(processed_dir: str | Path, name: str) -> pd.DataFrame:
    return pd.read_pickle(Path(processed_dir) / f"{name}.pkl")


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return df.drop(columns=[c for c in ("label", "label_raw") if c in df.columns]), df["label"].to_numpy()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)
