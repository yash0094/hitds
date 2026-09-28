"""Export a self-contained deployment bundle into deploy/ (used by the Dockerfile and Hugging Face Space).

Copies the LIVE model version and capped data splits (enough to replay traffic and to retrain on analyst
feedback), so the deployed app needs no raw datasets and no internet.

    python scripts/06_export_deploy.py --dataset cicids2017
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import ROOT, load_config, use_dataset  # noqa: E402
from hitds.data import load_split  # noqa: E402
from hitds.models.ensemble import stratified_cap  # noqa: E402

CAPS = {"train": 60000, "val": 30000, "test": 30000}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--out", default=str(ROOT / "deploy"))
    args = ap.parse_args()
    base = load_config()
    cfg = use_dataset(base, args.dataset)
    out = Path(args.out)
    reg = json.loads((Path(cfg["paths"]["models"]) / "registry.json").read_text())
    live = reg["live"]
    m_out = out / "models" / args.dataset
    d_out = out / "data" / "processed" / args.dataset
    shutil.rmtree(m_out, ignore_errors=True)
    shutil.rmtree(d_out, ignore_errors=True)
    m_out.mkdir(parents=True)
    d_out.mkdir(parents=True)
    shutil.copytree(Path(cfg["paths"]["models"]) / f"v{live}", m_out / f"v{live}")
    (m_out / "registry.json").write_text(json.dumps({"live": live, "history": [{"version": live, "note": "exported"}]},
                                                    indent=2))
    sizes = {}
    for name, cap in CAPS.items():
        df = load_split(cfg["paths"]["processed"], name)
        df = df.iloc[stratified_cap(df["label"].to_numpy(), cap, base["seed"])].reset_index(drop=True)
        df.to_pickle(d_out / f"{name}.pkl")
        sizes[name] = len(df)
    for f in ("meta.json", "DATA_CARD.md"):
        if (Path(cfg["paths"]["processed"]) / f).exists():
            shutil.copy(Path(cfg["paths"]["processed"]) / f, d_out / f)
    shutil.copy(ROOT / "config.yaml", out / "config.yaml")
    mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    bundle_mb = (m_out / f"v{live}" / "bundle.joblib").stat().st_size / 1e6
    print(json.dumps({"dataset": args.dataset, "live_version": live, "rows": sizes, "bundle_MB": round(bundle_mb, 1),
                      "deploy_folder_MB": round(mb, 1), "path": str(out)}, indent=2))
    if bundle_mb > 400:
        print("WARNING: the model file is large. For hosting, set models.rf.max_depth: 30 and n_estimators: 120 in "
              "config.yaml, retrain (scripts/03_train.py --fresh) and export again.")
