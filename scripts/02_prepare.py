"""Clean, de-duplicate, harmonise labels, cap and split 70/15/15 with a leakage check.
Writes data/processed/<dataset>/{train,val,test}.pkl, meta.json and DATA_CARD.md.

    python scripts/02_prepare.py --dataset cicids2017
    python scripts/02_prepare.py --dataset cicids2017 --holdout-class Infiltration --tag zeroday
    python scripts/02_prepare.py --dataset synthetic                     # offline smoke test only
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import load_config, seed_everything, use_dataset  # noqa: E402
from hitds.data import LOADERS, CleaningLog, make_synthetic, split_and_save, verify_manifest  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017", choices=list(LOADERS) + ["synthetic"])
    ap.add_argument("--holdout-class", default=None, help="remove this class from train/val (zero-day test)")
    ap.add_argument("--tag", default=None, help="save under <dataset>_<tag> (e.g. zeroday) instead of <dataset>")
    ap.add_argument("--n", type=int, default=20000, help="rows for --dataset synthetic")
    args = ap.parse_args()
    base = load_config()
    seed_everything(base["seed"])
    name = f"{args.dataset}_{args.tag}" if args.tag else args.dataset
    cfg = use_dataset(base, name)
    if args.dataset == "synthetic":
        df, clog = make_synthetic(args.n, base["seed"]), CleaningLog()
    else:
        raw = Path(base["paths"]["raw"]) / args.dataset
        v = verify_manifest(raw)
        print("integrity:", v)
        if v["ok"] is False:
            sys.exit("raw files do not match MANIFEST.json - re-run scripts/01_download.py")
        df, clog = LOADERS[args.dataset](raw, base, base["seed"])
    meta = split_and_save(df, cfg["paths"]["processed"], base, base["seed"], args.holdout_class, clog, name)
    shutil.copy(Path(cfg["paths"]["processed"]) / "DATA_CARD.md", Path(cfg["paths"]["reports"]) / "DATA_CARD.md")
    print(json.dumps({k: meta[k] for k in ("rows", "class_counts", "n_features", "holdout_class", "leakage_overlap")},
                     indent=2))
    print("data card:", Path(cfg["paths"]["reports"]) / "DATA_CARD.md")
