# BUILD_PLAN: phases for Claude Code (tick each when its check passes)

Each phase lists **do** and **check**. If a check fails, fix the cause and record it in `docs/CHANGELOG.md`.
Times are rough estimates for the i7-7600U laptop.

## Phase 0: Environment and tests (≈15 min) ☐
**Do**: create `.venv` (Python 3.11 or 3.12), install CPU torch, `pip install -r requirements.txt`, `pip install waitress`.
**Check**: `python -c "import torch,sklearn,flask,shap,ctgan,waitress;print(torch.__version__, torch.cuda.is_available())"`
prints a version and `False`, and `pytest -q` passes **with 0 skipped** (the torch-parity test must run).

## Phase 1: Offline smoke demo (≈5 min) ☐
**Do**: `02_prepare.py --dataset synthetic`, `03_train.py --dataset synthetic --fresh`, `.\run_demo.ps1 -Dataset synthetic`.
**Check**: log in (analyst / hitds-demo) at http://127.0.0.1:5000, start traffic, review 3 alerts with J/T/F/Enter.
Also mark one alert "unsure" and finish it as `reviewer2` (hitds-review) in a private window. Confirm "What drove
the score" says `shap`; if the SHAP output shape errors, fix `Explainer.attribution`. Confirm the audit pill is green.

## Phase 2: Download (≈10-30 min, network) ☐
**Do**: `python scripts/01_download.py`.
**Check**: `data/raw/<ds>/MANIFEST.json` exists for all three datasets. Record the dataset commit hashes (from the
HF cache or `huggingface_hub.HfApi().dataset_info(repo).sha`) into `revision:` in `config.yaml`.

## Phase 3: Prepare (≈10 min) ☐
**Do**: `02_prepare.py` for `cicids2017`, `unsw_nb15`, `nslkdd`; then `--dataset cicids2017 --holdout-class Infiltration --tag zeroday`.
**Check**: each `reports/<ds>/DATA_CARD.md` shows the cleaning steps with non-zero duplicate removal for CIC, and the
leakage overlap is all zeros. CIC classes are BENIGN, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration,
Heartbleed (Heartbleed may drop out if fewer than 10 rows survive cleaning; report that). Peak RAM stays under ~11 GB.
NSL-KDD: set `normal_value` correctly.

## Phase 4: Train (≈10-25 min per dataset) ☐
**Do**: `03_train.py --dataset cicids2017 --fresh`; the same for `unsw_nb15` and `nslkdd` (`--members rf,mlp` for NSL).
**Check**: test macro-F1, DR, FAR, per-class recall and `ms_per_event` printed; `models/<ds>/v1/metrics.json` exists.
If CIC takes more than 25 min, lower `svm.max_train_rows` / `mlp.epochs` and record it.

## Phase 5: CTGAN (≈20-60 min) ☐
**Do**: time `03_train.py --dataset cicids2017 --augment ctgan` with `ctgan_epochs: 50` first, then raise if affordable.
**Check**: log shows the share of synthetic rows per class that passed validation. Keep whichever of SMOTE/CTGAN has the
better validation macro-F1 as `augment.method`, then retrain v1 with `--fresh`.

## Phase 6: Experiments (≈1-3 h, can run unattended) ☐
**Do**:
```
python scripts/04_experiments.py --dataset cicids2017
python scripts/04_experiments.py --dataset cicids2017 --only E4 --e4-mode cold --repeats 5
python scripts/04_experiments.py --dataset cicids2017_zeroday --only E4 --repeats 3
python scripts/04_experiments.py --dataset unsw_nb15 --only E1,E2,E3
```
**Check**: `results*.md` and PNGs exist. Write `reports/SUMMARY.md`: one paragraph and the key measured numbers per
experiment, with caveats (oracle analyst, replay hosts are simulated, class caps, CTGAN epochs scaled down, known CIC-IDS2017
labelling issues). Say plainly where the loop did **not** help (e.g., if entropy ≈ random).

## Phase 7: Cross-dataset (≈10 min) ☐
**Do**: `python scripts/07_cross_dataset.py`. **Check**: `reports/cross_dataset.json` exists. Add the table to SUMMARY.md.

## Phase 8: Real-traffic path (≈20 min) ☐
**Do**: with the server running and `HITDS_API_KEY` set, `python scripts/capture_live.py send --csv
data/raw/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv --limit 2000` and upload a small CSV through the UI. **Check**: coverage ≥ 0.95, and alerts show `source: api` /
`upload:…`. Optional, only in the student's own VM lab: install cicflowmeter + Npcap and run `capture_live.py follow`.

## Phase 9: Report alignment ☐
**Do**: write `reports/REPORT_UPDATES.md`, listing every number or claim in the project report, the FAER proposal and the draft
paper that must change, with replacement text and tables generated from the measured files.
**ASK** before editing any Word/PDF.

## Phase 10: Deploy ☐
**Do**: `06_export_deploy.py --dataset cicids2017`; `.\run_demo.ps1 -Public` and open the tunnel URL from a phone.
If Docker Desktop is installed: `docker build -t hitds .` and `docker run -p 7860:7860 -e HITDS_USERS=... hitds`.
Optional hosting per `docs/DEPLOY.md` (**ASK** before anything that costs money).
**Check**: `/healthz` OK through the public URL, login works, and the queue fills.

## Phase 11: Demo rehearsal ☐
**Do**: fresh `.\run_demo.ps1`, about 60 real verdicts (mix blind and revealed, 3 "unsure" → second review), one retrain,
then `05_report_live.py`. **Check**: the `docs/DEMO_SCRIPT.md` run-through takes under 8 minutes with no errors in the server log
or browser console; `live_session.md` numbers go into SUMMARY.md as "live study (n = 1 analyst)".
