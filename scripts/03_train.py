"""Train the detection bundle (preprocessor + ensemble + isolation forest + calibrated triage),
evaluate it on the test split, save it as models/<dataset>/v1 and make it live.

    python scripts/03_train.py --dataset cicids2017
    python scripts/03_train.py --dataset cicids2017 --augment ctgan
    python scripts/03_train.py --dataset cicids2017 --members rf,mlp      # faster
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import load_config, seed_everything, use_dataset  # noqa: E402
from hitds.data import load_split  # noqa: E402
from hitds.pipeline import evaluate_bundle, save_bundle, set_live, train_bundle  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--augment", default=None, choices=[None, "none", "smote", "ctgan"])
    ap.add_argument("--members", default="rf,svm,knn,mlp")
    ap.add_argument("--fresh", action="store_true", help="delete existing model versions and the demo database")
    args = ap.parse_args()
    base = load_config()
    seed_everything(base["seed"])
    cfg = use_dataset(base, args.dataset)
    if args.fresh:
        shutil.rmtree(cfg["paths"]["models"], ignore_errors=True)
        Path(cfg["paths"]["models"]).mkdir(parents=True)
        Path(cfg["paths"]["db"]).unlink(missing_ok=True)
    train = load_split(cfg["paths"]["processed"], "train")
    val = load_split(cfg["paths"]["processed"], "val")
    test = load_split(cfg["paths"]["processed"], "test")
    b = train_bundle(cfg, train, val, method=args.augment, members=tuple(args.members.split(",")), version=1)
    b.metrics["test"] = evaluate_bundle(b, test)
    d = save_bundle(b, cfg["paths"]["models"])
    set_live(cfg["paths"]["models"], 1, "initial training")
    t = b.metrics["test"]
    print(json.dumps({k: t[k] for k in ("accuracy", "macro_f1", "weighted_f1", "mcc", "roc_auc_ovr_macro",
                                        "false_positive_rate", "attack_detection_rate", "route_share",
                                        "auto_accuracy", "missed_attacks_in_auto", "members")}, indent=2))
    print("per-class recall:", {k: round(v["recall"], 3) for k, v in t["per_class"].items()})
    print("saved to", d)
