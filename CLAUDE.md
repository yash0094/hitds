# CLAUDE.md: HITDS (Human-in-the-Loop Intrusion Detection System)

You are working in the HITDS repository with the student who owns it. The student will paste the steps
from `CLAUDE_CODE_STEPS.md` one at a time. For each step: do it, run its **check**, fix failures at the cause
(never loosen a check or fake a number), then tick the phase in `BUILD_PLAN.md` and add a line to
`docs/CHANGELOG.md`. Keep going inside a step without asking unless it says **ASK**.

## What this project is
FAER Scholar Award 2025-26 entry (T. John Institute of Technology, Bangalore). Judges want a **live demo**.
Machine learning scores network flows, and a human analyst (the student) reviews only the uncertain or
high-severity alerts on a web console. Verdicts retrain the model and update per-host attack beliefs, and every
step goes into a hash-chained audit log. The target is a **deployable** app (see `docs/DEPLOY.md`).

Layers:
1. Imbalance: SMOTE baseline, CTGAN (`hitds/augment.py`)
2. Detection: soft-voting RF + RBF-SVM + KNN + PyTorch MLP, Isolation Forest, and the IS-IDS SMAD statistical
   layer (`hitds/models/`, `hitds/smad.py`)
3. Entropy triage: AUTO / REVIEW / ESCALATE (`hitds/triage.py`)
4. Per-host HMM after Kim, Dán & Zhu (IEEE TIFS 2024): alert model, investigation updates, sequential
   test, Max-KL / Max-Ratio queue ordering (`hitds/hmm.py`)

Around them: `hitds/data.py` (Hugging Face download, SHA-256 manifest, cleaning, leakage-safe split, data
card), `hitds/collect.py` (CICFlowMeter column mapping, Suricata evidence), `hitds/db.py` (SQLite, audit
chain, second review, kappa), `hitds/engine.py` (live loop, retrain + promotion gate), `hitds/stream.py`
(replay with kill-chain campaigns and ground-truth stages), `app/` (Flask console, login, upload, API),
`serve.py` (waitress), `scripts/01-07`, `tests/`. Paper mapping: `docs/PAPERS.md`. Every requirement and
its status: `docs/REQUIREMENTS_TRACEABILITY.md`.

## Machine constraints
- Windows 10/11, **Intel i7-7600U (2 cores / 4 threads), 16 GB RAM, no CUDA GPU** (Intel HD 620).
- CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Never enable CUDA.
- Keep the class caps in `config.yaml`. CIC-IDS2017 is about 2.8M rows raw.
- If a step would run longer than ~20 minutes, shrink it (caps, `--members rf,mlp`, fewer epochs), say what
  you shrank, and record it in the CHANGELOG.
- Commands are written for PowerShell. Use `python`, not `python3`.

## Datasets (already chosen, Hugging Face)
| key | repo | files | role |
|---|---|---|---|
| cicids2017 | `c01dsnap/CIC-IDS2017` | 8 daily `*.pcap_ISCX.csv` (~885 MB) | main training + live demo |
| unsw_nb15 | `Mouwiya/UNSW-NB15` | `UNSW_NB15_training-set.csv` | second dataset + cross-dataset |
| nslkdd | `Mireu-Lab/NSL-KDD` | `train.csv`, `test.csv` | legacy baseline (binary) |
First-run checks: (1) NSL-KDD `class` is numeric, so confirm which value means normal (~53%) and set
`datasets.nslkdd.normal_value`; (2) after the first successful download, pin `revision:` in `config.yaml` to the
commit hash printed by the Hugging Face cache, or recorded in MANIFEST.json, for reproducibility.

## Rules
1. **Never invent, round up or copy results.** Every number in a report, README or slide must come from
   `reports/<dataset>/results*.json`, `models/<dataset>/v*/metrics.json`, `DATA_CARD.md` or `live_session.json`.
   The student's existing project report and draft paper contain results (97.2% accuracy, 95.1→97.8% F1, 42 ms,
   "28 tests passed", "accuracy up to 99%, annotation reduction up to 98%") that this code did not produce.
   Phase 9 replaces them with measured numbers.
2. E3/E4/E5 use a simulated **oracle** analyst. Always label it so. Real human numbers come only from
   `scripts/05_report_live.py`.
3. `make_synthetic()` data is for tests and smoke runs only. Never report its numbers.
4. Do not weaken the audit log (triggers, hash chain, single-transaction writes) or the login/CSRF/API-key checks.
5. The analyst must never see `ground_truth` (UI or API).
6. The dashboard stays dependency-free (no CDN) so the demo works offline.
7. Run `pytest -q` after any change under `hitds/` or `app/`. All tests must pass. The torch-parity test must run
   (not skip) once torch is installed.
8. Actions are simulated. Never add code that changes real firewall rules or network settings.
9. Packet capture only on the student's own isolated lab network or VMs.

## Commands
```
python scripts/01_download.py [--dataset cicids2017]
python scripts/02_prepare.py --dataset cicids2017 [--holdout-class Infiltration --tag zeroday]
python scripts/03_train.py  --dataset cicids2017 --fresh [--augment ctgan] [--members rf,svm,knn,mlp]
python scripts/04_experiments.py --dataset cicids2017 [--only E3,E5] [--e4-mode cold --repeats 5]
python scripts/05_report_live.py --dataset cicids2017
python scripts/06_export_deploy.py --dataset cicids2017
python scripts/07_cross_dataset.py
python -m app.server --dataset synthetic --reset       # dev server (login analyst / hitds-demo)
.\run_demo.ps1 [-Public]                               # production server (+ public tunnel)
pytest -q
```

## Already verified (cloud sandbox, synthetic data, no torch/shap/ctgan/waitress, no Docker daemon)
20 of 21 tests pass (the torch-parity test skipped). Verified: the web app through the test client (login, CSRF,
upload with CICFlowMeter-style column names, API-key ingest, audit export); a production-server run from a copy
of the Docker file layout; the engine with second review and HMM updates; experiments E3/E4 (warm and cold, with
t-tests) and E5; the deploy export.
**Not yet verified:** real Hugging Face downloads and cleaning on real CSVs, anything needing torch, shap, ctgan
or waitress, the Docker build, cicflowmeter capture, and every timing on the laptop.
