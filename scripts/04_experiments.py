"""Quantitative evaluation - the proposal's incremental experiment sequence (§27), with the protocols
of the papers it builds on:

  E1  augmentation:   none vs SMOTE vs CTGAN -> per-class recall (rare classes)          [Zeng & Nait-Abdesselam]
  E2  ensemble:       each member alone vs soft-voting fusion (+ latency)                [ensemble literature]
  E3  autonomy curve: sweep the entropy percentile -> analyst workload vs detection quality
  E4  feedback loop:  simulated analyst, 50 labels / iteration, entropy vs hybrid vs random sampling,
                      warm start (full benchmark) or cold start (500 labelled rows, HITL review paper),
                      repeated seeds + paired t-test, SavedRate                          [Settles; HITL review]
  E5  Layer 4:        no investigation vs static priority vs Max Ratio vs Max KL, budget B per slot,
                      analyst error omega -> mean time to detection (MTD), mean time between false
                      detections (MTBFD), detection rate, belief MSE                     [Kim, Dán & Zhu 2024]

E4/E5 use an ORACLE analyst (dataset labels, optionally with error omega). That measures what
human feedback can buy under stated assumptions; the real human numbers come from the dashboard
database (scripts/05_report_live.py). Say so in the report.

    python scripts/04_experiments.py --dataset cicids2017
    python scripts/04_experiments.py --dataset cicids2017 --only E4 --e4-mode cold --repeats 5
    python scripts/04_experiments.py --dataset cicids2017_zeroday --only E4      # after --tag zeroday prepare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hitds.common import get_logger, load_config, seed_everything, use_dataset  # noqa: E402
from hitds.data import load_split, xy  # noqa: E402
from hitds.hmm import STATES, HostTracker  # noqa: E402
from hitds.models.ensemble import stratified_cap  # noqa: E402
from hitds.pipeline import evaluate_bundle, evaluate_predictions, train_bundle  # noqa: E402
from hitds.stream import TrafficReplay  # noqa: E402
from hitds.triage import normalized_entropy  # noqa: E402

log = get_logger("hitds.exp")


def plot(fig_path, fn):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 4))
        fn(ax)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting must never kill an experiment run
        log.warning("plot failed: %s", e)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)  # noqa: E731
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows])


# --------------------------------------------------------------------------- E1
def e1_augmentation(cfg, train, val, ev, members, out):
    rows, methods = [], ["none", "smote"]
    try:
        import ctgan  # noqa: F401

        methods.append("ctgan")
    except ImportError:
        log.warning("ctgan not installed - E1 runs without it")
    for m in methods:
        t0 = time.time()
        b = train_bundle(cfg, train, val, method=m, members=members)
        r = evaluate_bundle(b, ev)
        row = {"augment": m, "macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "DR": r["attack_detection_rate"],
               "FAR": r["false_alarm_rate"], "mcc": r["mcc"], "train_s": round(time.time() - t0, 1)}
        for c, pc in r["per_class"].items():
            if c != "BENIGN":
                row[f"recall_{c}"] = round(pc["recall"], 4)
        rows.append(row)
        log.info("E1 %s -> macro-F1 %.4f", m, r["macro_f1"])
    out["E1_augmentation"] = rows


# --------------------------------------------------------------------------- E2
def e2_ensemble(cfg, train, val, ev, out):
    b = train_bundle(cfg, train, val, members=("rf", "svm", "knn", "mlp"))
    X, y = xy(ev)
    t0 = time.perf_counter()
    Z = b.pre.transform(X)
    mp = b.det.member_proba(Z)
    P = b.det.predict_proba(Z, mp)
    ms = (time.perf_counter() - t0) * 1000 / len(Z)
    rows = []
    for name, Pm in list(mp.items()) + [("soft-voting ensemble", P)]:
        r = evaluate_predictions(y, Pm, b.classes)
        rows.append({"model": name, "macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "DR": r["attack_detection_rate"],
                     "FAR": r["false_alarm_rate"], "mcc": r["mcc"], "roc_auc": r["roc_auc_ovr_macro"] or float("nan"),
                     "fit_s": b.det.fit_seconds.get(name, round(sum(b.det.fit_seconds.values()), 1))})
    rows[-1]["ms_per_event_batch"] = round(ms, 3)
    out["E2_ensemble"] = rows
    return b


# --------------------------------------------------------------------------- E3
def e3_autonomy_curve(b, val, ev, out, fig_dir):
    Xv, _ = xy(val)
    Pv = b.det.predict_proba(b.pre.transform(Xv))
    X, y = xy(ev)
    s = b.score(X)
    P, pred = s["P"], np.asarray(b.classes)[s["P"].argmax(1)]
    rows = []
    saved = (b.triage.entropy_percentile, b.triage.h_threshold, b.triage.confidence_floor)
    b.triage.confidence_floor = 0.0          # isolate the effect of the entropy threshold
    for pct in [0, 25, 50, 70, 80, 90, 95, 97.5, 99, 100]:
        b.triage.set_percentile(pct, Pv)
        routes = np.array([r["route"] for r in b.triage.route(P, s["anomaly"], s["smad_flag"])])
        auto = routes == "AUTO"
        human = ~auto
        sys_pred = np.where(human, y, pred)   # the (oracle) analyst corrects what it sees
        rows.append({"entropy_percentile": pct, "h_threshold": round(b.triage.h_threshold, 4),
                     "human_share": float(human.mean()), "saved_rate": float(auto.mean()),
                     "auto_error_rate": float(np.mean(pred[auto] != y[auto])) if auto.any() else 0.0,
                     "missed_attacks_auto": int(np.sum(auto & (y != "BENIGN") & (pred == "BENIGN"))),
                     "system_macro_f1": float(f1_score(y, sys_pred, average="macro", zero_division=0)),
                     "model_only_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))})
    b.triage.entropy_percentile, b.triage.h_threshold, b.triage.confidence_floor = saved
    out["E3_autonomy_curve"] = rows
    df = pd.DataFrame(rows)
    plot(fig_dir / "E3_autonomy_curve.png", lambda ax: (
        ax.plot(df.human_share * 100, df.system_macro_f1, "o-", label="system macro-F1 (model + analyst)"),
        ax.axhline(df.model_only_macro_f1.iloc[0], ls="--", c="gray", label="model alone"),
        ax.set_xlabel("% of events sent to a human"), ax.set_ylabel("macro-F1"),
        ax.set_title("Autonomy curve: analyst workload vs detection quality"), ax.legend()))


# --------------------------------------------------------------------------- E4
def _select(strategy, b, pool_X, avail, per_iter, rng):
    if strategy == "random":
        return rng.choice(avail, min(per_iter, len(avail)), replace=False)
    Za = b.pre.transform(pool_X.iloc[avail])
    P = b.det.predict_proba(Za)
    H = normalized_entropy(P)
    if strategy == "entropy":
        return avail[np.argsort(-H)[:per_iter]]
    # hybrid: half the budget on the most uncertain flows, half on flows the ensemble calls BENIGN but the
    # isolation forest / SMAD find least benign-like - the confident mistakes entropy never surfaces
    pred_benign = np.asarray(b.classes)[P.argmax(1)] == "BENIGN"
    odd = b.det.anomaly_score(Za) + (b.smad.score(Za) if b.smad is not None else 0)
    odd = np.where(pred_benign, odd, -np.inf)
    first = list(avail[np.argsort(-H)[: per_iter // 2]])
    taken = set(first)
    second = [i for i in avail[np.argsort(-odd)] if i not in taken][: per_iter - len(first)]
    return np.array(first + second)


def e4_feedback_loop(cfg, train, val, ev, members, iters, per_iter, mode, repeats, out, fig_dir, zero_day=None):
    """mode=warm: model starts from the full (capped) benchmark, labels come from a held-out pool.
    mode=cold: model starts from `initial_pool` labelled rows; the rest of train is the unlabelled pool."""
    from scipy.stats import ttest_rel

    results = {s: [] for s in ("entropy", "hybrid", "random")}
    for rep in range(repeats):
        seed = cfg["seed"] + rep
        rng0 = np.random.default_rng(seed)
        if mode == "cold":
            init_idx = stratified_cap(train["label"].to_numpy(), cfg.get("e4", {}).get("initial_pool", 500), seed)
            base = train.iloc[init_idx]
            pool = train.drop(train.index[init_idx])
            pool = pool.iloc[stratified_cap(pool["label"].to_numpy(), 30000, seed)]
        else:
            base, pool = train, ev.sample(frac=0.5, random_state=seed)
        ev_rep = ev.drop(pool.index, errors="ignore")
        for strategy in results:
            rng = np.random.default_rng(seed)
            labelled: list[int] = []
            rows = []
            b = train_bundle(cfg, base, val, members=members, max_rows=cfg["feedback"]["retrain_max_rows"])
            pre = b.pre
            Xp, _ = xy(pool)
            for it in range(iters + 1):
                r = evaluate_bundle(b, ev_rep)
                row = {"strategy": strategy, "repeat": rep, "iteration": it, "labels": len(labelled),
                       "macro_f1": r["macro_f1"], "accuracy": r["accuracy"], "FAR": r["false_alarm_rate"],
                       "DR": r["attack_detection_rate"], "human_share": r["human_share"],
                       "annotation_saved_rate": 1 - (len(labelled) + (len(base) if mode == "cold" else 0))
                       / (len(pool) + (len(base) if mode == "cold" else 0)),
                       "missed_attacks_auto": r["missed_attacks_in_auto"]}
                if zero_day:
                    row[f"recall_{zero_day}"] = r["per_class"].get(zero_day, {}).get("recall", 0.0)
                rows.append(row)
                log.info("E4 %s rep %d %-7s it %d labels %4d  macro-F1 %.4f  human %.3f", mode, rep, strategy, it,
                         len(labelled), r["macro_f1"], r["human_share"])
                if it == iters:
                    break
                avail = np.setdiff1d(np.arange(len(pool)), labelled)
                labelled += list(_select(strategy, b, Xp, avail, per_iter, rng))
                fb = pool.iloc[labelled].copy()
                b = train_bundle(cfg, base, val, members=members, feedback=fb, pre=pre, version=it + 2,
                                 max_rows=cfg["feedback"]["retrain_max_rows"])
            results[strategy] += rows
        del rng0
    out["E4_feedback_loop"] = {"mode": mode, "repeats": repeats, "rows": results}
    # paired t-test on the final iteration across repeats
    final = {s: [r["macro_f1"] for r in rows if r["iteration"] == iters] for s, rows in results.items()}
    tests = []
    for a in ("entropy", "hybrid"):
        if repeats >= 2:
            t, p = ttest_rel(final[a], final["random"])
            tests.append({"comparison": f"{a} vs random", "mean_diff": float(np.mean(final[a]) - np.mean(final["random"])),
                          "t": float(t), "p_value": float(p), "repeats": repeats})
    out["E4_ttests"] = tests
    for key, lab in [("macro_f1", "macro-F1"), ("human_share", "share of events sent to a human")] + \
                    ([(f"recall_{zero_day}", f"recall on unseen class '{zero_day}'")] if zero_day else []):
        def _p(ax, key=key, lab=lab):
            for s, rows in results.items():
                df = pd.DataFrame(rows).groupby("labels")[key].agg(["mean", "std"]).reset_index()
                ax.errorbar(df["labels"], df["mean"], yerr=df["std"].fillna(0), marker="o", capsize=3,
                            label=f"{s} sampling")
            ax.set_xlabel("analyst labels added"), ax.set_ylabel(lab), ax.set_title(f"Feedback loop ({mode} start): {lab}")
            ax.legend()
        plot(fig_dir / f"E4_{mode}_{key}.png", _p)


# --------------------------------------------------------------------------- E5
def e5_layer4(cfg, b, ev, n_events, budget, slot, omega, thetas, out, fig_dir):
    """Kim et al. evaluation on classifier-generated alerts. Oracle analyst answers correctly with prob 1-omega."""
    replay = TrafficReplay(ev, cfg["stream"]["n_hosts"], cfg["seed"], max_campaigns=cfg["stream"]["n_hosts"] - 2)
    batch = replay.next_batch(n_events)
    meta_cols = ["ground_truth", "host", "src_ip", "dst_ip", "gt_stage", "campaign"]
    s = b.score(batch.drop(columns=meta_cols))
    preds = np.asarray(b.classes)[s["P"].argmax(1)]
    prio = np.array([r["priority"] for r in s["routes"]])
    hosts, gt_stage, gt = batch["host"].to_numpy(), batch["gt_stage"].to_numpy(), batch["ground_truth"].to_numpy()
    onset = {}
    for t, (h, st) in enumerate(zip(hosts, gt_stage)):
        if st > 0 and h not in onset:
            onset[h] = t
    rows, curves = [], []
    for log_theta in thetas:
        for policy in ("none", "static", "max_ratio", "max_kl"):
            c = {**cfg, "hmm": {**cfg.get("hmm", {}), "log_threshold": log_theta, "analyst_error": omega}}
            tr = HostTracker(b.classes, b.alert_model, c)
            rng = np.random.default_rng(cfg["seed"])
            detected, false_det, mse, investigated = {}, 0, [], 0
            open_alerts: dict[int, int] = {}          # alert id (= event index) -> event index
            for t in range(n_events):
                h = hosts[t]
                tr.update(h, preds[t], alert_id=t)
                if preds[t] != "BENIGN":
                    open_alerts[t] = t
                b_h = tr.hosts[h].belief()
                mse.append(float(np.sum((b_h - np.eye(len(STATES))[gt_stage[t]]) ** 2)))
                if policy != "none" and budget and t % slot == 0 and open_alerts:
                    cand = list(open_alerts)
                    if policy == "static":
                        order = sorted(cand, key=lambda a: -prio[a])
                    else:
                        sc = {a: tr.policy_score(a, policy) for a in cand}
                        order = sorted((a for a in cand if sc[a] is not None and np.isfinite(sc[a])), key=lambda a: -sc[a])
                    for a in order[:budget]:
                        truth = 1 if gt[a] != "BENIGN" else 0
                        outcome = truth if rng.random() > omega else 1 - truth
                        tr.investigate(a, outcome)
                        open_alerts.pop(a, None)
                        investigated += 1
                if len(open_alerts) > 500:            # alerts too old to matter
                    for a in sorted(open_alerts)[:100]:
                        open_alerts.pop(a)
                stt = tr.state(h)
                if stt["sprt"] == "compromised" and h not in detected:
                    if gt_stage[t] == 0:
                        false_det += 1
                        tr.reset_host(h)          # false detection: analyst clears the host, tracking restarts
                    else:
                        detected[h] = t
            delays = [detected[h] - onset[h] for h in detected if h in onset]
            benign_steps = int(np.sum(gt_stage == 0))
            rows.append({"log_theta": log_theta, "policy": policy, "campaigns": len(onset),
                         "detected": len(delays), "detection_rate": len(delays) / max(1, len(onset)),
                         "MTD_events": float(np.mean(delays)) if delays else float("nan"),
                         "false_detections": false_det,
                         "MTBFD_events": benign_steps / false_det if false_det else float("inf"),
                         "belief_MSE": float(np.mean(mse)), "investigations": investigated})
            log.info("E5 logθ=%s %-9s MTD %s  det %d/%d  false %d", log_theta, policy, rows[-1]["MTD_events"],
                     len(delays), len(onset), false_det)
    out["E5_layer4"] = {"omega": omega, "budget_per_slot": budget, "slot_events": slot, "rows": rows}
    df = pd.DataFrame(rows)

    def _p(ax):
        for pol, g in df.groupby("policy"):
            ax.plot(g["log_theta"], g["MTD_events"], "o-", label=pol)
        ax.set_xlabel("log detection threshold θ"), ax.set_ylabel("mean time to detection (events)")
        ax.set_title("Layer 4: time to detection by alert-prioritisation policy"), ax.legend()
    plot(fig_dir / "E5_mtd.png", _p)


# --------------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cicids2017")
    ap.add_argument("--only", default="E1,E2,E3,E4,E5")
    ap.add_argument("--members", default="rf,mlp", help="members for E1/E3/E4/E5 (full ensemble is slow to retrain)")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--per-iter", type=int, default=50)
    ap.add_argument("--e4-mode", default="warm", choices=["warm", "cold"])
    ap.add_argument("--repeats", type=int, default=1, help="E4 repeats with different seeds (>=2 enables t-tests)")
    ap.add_argument("--e5-events", type=int, default=4000)
    ap.add_argument("--e5-budget", type=int, default=1, help="investigations per slot (paper: B)")
    ap.add_argument("--e5-slot", type=int, default=5, help="events per time slot")
    ap.add_argument("--e5-omega", type=float, default=0.1, help="analyst error probability (paper: omega)")
    ap.add_argument("--e5-thetas", default="4,8,12,16")
    args = ap.parse_args()
    base = load_config()
    seed_everything(base["seed"])
    cfg = use_dataset(base, args.dataset)
    fig_dir = Path(cfg["paths"]["reports"])
    meta = json.loads((Path(cfg["paths"]["processed"]) / "meta.json").read_text())
    train = load_split(cfg["paths"]["processed"], "train")
    val = load_split(cfg["paths"]["processed"], "val")
    test = load_split(cfg["paths"]["processed"], "test")
    members = tuple(args.members.split(","))
    only = set(args.only.split(","))
    out = {"dataset": args.dataset, "meta": {k: meta[k] for k in ("rows", "holdout_class", "leakage_overlap")},
           "members_for_loops": members, "timestamp": time.ctime(), "args": vars(args)}
    t0 = time.time()
    if "E1" in only:
        e1_augmentation(cfg, train, val, test, members, out)
    b = e2_ensemble(cfg, train, val, test, out) if "E2" in only else None
    if b is None and ({"E3", "E5"} & only):
        b = train_bundle(cfg, train, val, members=members)
    if "E3" in only:
        e3_autonomy_curve(b, val, test, out, fig_dir)
    if "E4" in only:
        e4_feedback_loop(cfg, train, val, test, members, args.iters, args.per_iter, args.e4_mode, args.repeats,
                         out, fig_dir, meta.get("holdout_class"))
    if "E5" in only:
        e5_layer4(cfg, b, test, args.e5_events, args.e5_budget, args.e5_slot, args.e5_omega,
                  [float(x) for x in args.e5_thetas.split(",")], out, fig_dir)
    out["total_seconds"] = round(time.time() - t0, 1)

    suffix = "" if only == {"E1", "E2", "E3", "E4", "E5"} else "_" + "_".join(sorted(only))
    (fig_dir / f"results{suffix}.json").write_text(json.dumps(out, indent=2, default=str))
    md = [f"# HITDS results - {args.dataset}", f"_generated {out['timestamp']}; test rows {len(test)}, "
          f"zero-day holdout: {meta.get('holdout_class')}; oracle analyst in E3-E5_", ""]
    for key, title in [("E1_augmentation", "E1 Augmentation"), ("E2_ensemble", "E2 Members vs ensemble"),
                       ("E3_autonomy_curve", "E3 Autonomy curve")]:
        if key in out:
            md += [f"## {title}", md_table(out[key]), ""]
    if "E4_feedback_loop" in out:
        e4 = out["E4_feedback_loop"]
        md += [f"## E4 Feedback loop ({e4['mode']} start, {e4['repeats']} repeat(s))"]
        for s, rows in e4["rows"].items():
            df = pd.DataFrame(rows).drop(columns=["repeat"]).groupby(["strategy", "iteration", "labels"]).mean().reset_index()
            md += [f"### {s} sampling (mean over repeats)", md_table(df.to_dict("records")), ""]
        if out.get("E4_ttests"):
            md += ["### Paired t-tests on final macro-F1", md_table(out["E4_ttests"]), ""]
    if "E5_layer4" in out:
        e5 = out["E5_layer4"]
        md += [f"## E5 Layer 4 (omega={e5['omega']}, budget {e5['budget_per_slot']} per {e5['slot_events']} events)",
               md_table(e5["rows"]), ""]
    (fig_dir / f"results{suffix}.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {fig_dir}/results{suffix}.md and figures ({out['total_seconds']}s)")
