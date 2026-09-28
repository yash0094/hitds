"""Download the benchmark datasets from Hugging Face into data/raw/<dataset>/.

    python scripts/01_download.py                   # all three (~1 GB)
    python scripts/01_download.py --dataset cicids2017
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import load_config  # noqa: E402
from hitds.data import download  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "cicids2017", "unsw_nb15", "nslkdd"])
    args = ap.parse_args()
    cfg = load_config()
    names = ["cicids2017", "unsw_nb15", "nslkdd"] if args.dataset == "all" else [args.dataset]
    for name in names:
        d = cfg["datasets"][name]
        man = download(d["hf_repo"], d["files"], Path(cfg["paths"]["raw"]) / name, d.get("revision"))
        print(f"{name}: {len(man['files'])} files pinned in data/raw/{name}/MANIFEST.json")
