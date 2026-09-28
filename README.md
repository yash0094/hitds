# HITDS: Human-in-the-Loop Intrusion Detection System

FAER Scholar Award 2025-26 · T. John Institute of Technology, Bangalore

Machine learning scores every network flow. A human analyst reviews only the uncertain or high-severity alerts.
Each verdict retrains the model and updates a per-host attack-progression model, and every step lands in a
tamper-evident audit trail.

```
 Hugging Face datasets ─┐                 ┌─ AUTO (logged / reversible action)
 CICFlowMeter CSV/API ──┼► clean ► SMAD (ANOVA 3σ) + ensemble (RF·SVM·KNN·PyTorch MLP) + IsolationForest ► entropy triage ─┼─ REVIEW ─┐
 Suricata signatures ───┘                                                                     ├─ ESCALATE┤
                         per-host HMM (Kim et al.): belief · sequential test · Max-KL queue order ────────┘         ▼
                                                                                               analyst console (login)
       gated retrain every 50 labels ◄── final labels (second review if unsure) ◄── verdict ◄───┘
       all of it ─► hash-chained, append-only audit log (exportable)
```

**Start here:** [`CLAUDE_CODE_STEPS.md`](CLAUDE_CODE_STEPS.md) walks you from this zip to a deployed app with Claude
Code in VS Code, one prompt per step.

## Quick start (Windows, CPU)
```
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
python scripts/02_prepare.py --dataset synthetic ; python scripts/03_train.py --dataset synthetic --fresh
.\run_demo.ps1 -Dataset synthetic          # http://127.0.0.1:5000  (analyst / hitds-demo)
```

## Layout
| path | what |
|---|---|
| `config.yaml` | every setting: datasets + revisions, caps, models, triage, SMAD, HMM policy, review, collection |
| `hitds/data.py` | HF download + SHA-256 manifest, cleaning (dedup, conflicts, NaN, impossible values), leakage-safe split, data card |
| `hitds/collect.py` | CICFlowMeter → CIC-IDS2017 column mapping, upload parsing, Suricata EVE evidence |
| `hitds/features.py` | median impute, signed-log + Min-Max, RF or ANOVA feature selection |
| `hitds/augment.py` | SMOTE, CTGAN + synthetic-row validation |
| `hitds/smad.py` | IS-IDS innate statistical layer (ANOVA + three-sigma) |
| `hitds/models/` | PyTorch MLP (NumPy inference), soft-voting ensemble, Isolation Forest |
| `hitds/triage.py` | normalised entropy, AUTO / REVIEW / ESCALATE, severity priority |
| `hitds/hmm.py` | Kim, Dán & Zhu alert HMM: ζ/δ from data, γ(ω) updates, MSGPRT, Max-KL / Max-Ratio |
| `hitds/explain.py` | SHAP attribution, SMAD evidence, similar incidents |
| `hitds/db.py` | SQLite store, audit hash chain, second review, Cohen's kappa |
| `hitds/engine.py` | live loop, upload/API ingest, policy-ranked queue, retrain + promotion gate |
| `app/`, `serve.py` | Flask console (login, CSRF, API key), waitress production server |
| `scripts/` | 01 download · 02 prepare · 03 train · 04 experiments · 05 live report · 06 deploy export · 07 cross-dataset · capture_live |
| `Dockerfile`, `docker-compose.yml`, `render.yaml`, `deploy/hf_space/` | deployment |
| `docs/` | requirements traceability, paper mapping, deployment guide, demo script, changelog |

## Datasets
[CIC-IDS2017](https://huggingface.co/datasets/c01dsnap/CIC-IDS2017) ·
[UNSW-NB15](https://huggingface.co/datasets/Mouwiya/UNSW-NB15) ·
[NSL-KDD](https://huggingface.co/datasets/Mireu-Lab/NSL-KDD)

## Honest-reporting notes
- Experiments E3-E5 simulate the analyst with dataset labels (an *oracle*, optionally with error ω). The real human
  numbers come from `scripts/05_report_live.py` after a console session.
- The replay invents hosts and IPs (the public CSVs have none usable), so Layer 4 is shown on simulated campaigns
  built from real flows. Uploaded or collected flows use their real IPs.
- Large classes are capped to fit a 16 GB laptop; CTGAN epochs are scaled down from the literature. Both are in the data card and config.
- Containment actions are simulated and logged; nothing touches a real firewall.
