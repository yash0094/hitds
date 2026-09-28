# Requirements traceability

Every requirement from the FAER proposal, the project report, our chats and the five reference
papers, with where it is implemented. Status: **done** = implemented and tested on synthetic data ·
**verify** = implemented but must be run on the laptop / real data · **deferred** = stated honestly as
out of scope.

## From our chats
| Requirement | Where | Status |
|---|---|---|
| Datasets collected from Hugging Face | `hitds/data.py::download`, `scripts/01_download.py`, `config.yaml` (repos + revisions) | verify |
| Precise data collection | SHA-256 manifest, dedup, label conflicts, NaN/negative removal, leakage check, data card (`hitds/data.py`) | done |
| Build automated with Claude Code in VS Code | `CLAUDE.md`, `BUILD_PLAN.md`, `CLAUDE_CODE_STEPS.md` | done |
| Python + PyTorch | `hitds/models/mlp.py` (PyTorch MLP, NumPy inference) | verify (needs torch) |
| Laptop: i7-7600U, 16 GB, no GPU | CPU-only torch, class caps, SVM/KNN subsampling, `retrain_max_rows` | verify timing |
| Judges want a live demo | Flask console, traffic replay, `run_demo.ps1`, `docs/DEMO_SCRIPT.md` | done |
| The student is the human in the loop | login, verdicts, blind review, second review | done |
| Deployable app | `serve.py` (waitress), auth, API key, `/healthz`, `Dockerfile`, `docker-compose.yml`, `render.yaml`, HF Space template, `docs/DEPLOY.md` | verify (Docker build on laptop) |
| Polished zip + step-by-step Claude Code instructions | this package, `CLAUDE_CODE_STEPS.md` | done |

## From the FAER proposal
| Requirement (proposal section) | Where | Status |
|---|---|---|
| Layer 1: CTGAN augmentation, SMOTE baseline (§21, §27) | `hitds/augment.py`, E1 | CTGAN: verify |
| Validate synthetic rows against protocol constraints (§29) | `augment.validate_synthetic` | done |
| Layer 2: soft-voting RF + RBF-SVM + KNN + NN at probability level (§21) | `hitds/models/ensemble.py`, E2 | done |
| Isolation Forest for behaviour with no supervised analogue (§27) | `EnsembleDetector.iforest`, triage zero-day route | done |
| Layer 3: Shannon-entropy uncertainty, threshold at a percentile (§20, §24) | `hitds/triage.py`, E3 | done |
| Three routes: autonomous / human review / escalation (§20) | `Triage.route` (AUTO / REVIEW / ESCALATE) | done |
| Severity weighting (§24) | `config.yaml triage.severity`, priority formula | done |
| Layer 4: HMM safe→intrusion→exploit→impact, Bayesian filtering, Max-KL (§21, §27) | `hitds/hmm.py` (Kim et al. model), E5 | done |
| Dashboard with evidence, feature attribution, comparable incidents (§20) | `app/`, `hitds/explain.py` (SHAP + nearest incidents) | SHAP: verify |
| Evidence before verdict; automation-bias monitoring (§29) | blind review, agreement blind vs revealed | done |
| Verdicts TP / FP / unknown become training samples (§20) | `db.add_verdict`, `engine.feedback_frame` | done |
| Incremental retraining without full rebuild; promote only after offline evaluation (§27) | `engine.retrain` (capped rows + promotion gate on validation) | done |
| Immutable audit trail: evidence → score → decision → action (§20, §24) | `hitds/db.py` hash chain, triggers, single-transaction writes, export | done |
| No irreversible action taken silently (§21) | high-severity classes ESCALATE; actions only after TP verdict; simulated executor | done |
| Metrics: accuracy, precision, recall, F1, FPR, MCC, ROC-AUC, per class, confusion matrices (§27) | `pipeline.evaluate_predictions` | done |
| Human-loop metrics: escalation volume, labels to reach F1, mean time to decision, inter-annotator agreement (§27) | E3/E4, `05_report_live.py`, Cohen's kappa | done |
| Second review when reviewers disagree; weight by reviewer confidence (§29) | `review.second_review_below`, `_weight` in feedback | done |
| 70/15/15 split, stratified (§27) | `data.split_and_save` | done |
| Stratified 5-fold CV (§27) | not in pipeline; repeated seeds + paired t-tests in E4 instead | deferred (can add) |
| Train CIC-IDS2017, test UNSW-NB15 cross-dataset (§27) | `scripts/07_cross_dataset.py` (shared 9-feature schema) | verify |
| NSL-KDD legacy baseline (§21) | `load_nslkdd` | verify mapping |
| Ablation, one layer at a time (§27) | E1 (augmentation), E2 (ensemble), E3 (AL threshold), E5 (Layer 4 on/off) | done |
| Report where the human loop does NOT help (§27) | E4 includes random baseline + t-tests; E5 includes "none" policy | done |
| Signature matching (Suricata/Zeek) (§18, §27) | `collect.load_suricata_eve` → evidence + forces review | verify with Suricata |
| CICFlowMeter PCAP → flows (§27) | `scripts/capture_live.py`, column mapping in `collect.py` | verify |
| Isolated VM test-bed with generated attacks (§27) | capture script supports it; building the VMs is manual | deferred (manual) |
| Flask or FastAPI; REST + WebSocket (§27) | Flask REST; polling instead of WebSocket | done (polling) |
| SQLite dev, PostgreSQL evaluation (§27) | SQLite only | deferred |
| MLflow tracking (§27) | versioned `models/<ds>/vN/metrics.json` + registry | deferred (optional) |
| Concept-drift monitoring (§29) | per-version metrics in live session report | partial |

## From the project report
| Requirement | Where | Status |
|---|---|---|
| Confidence threshold 0.85 (§4.2, §6.1) | `triage.confidence_floor: 0.85` | done |
| Retrain after every 50 labels (§4.2) | `feedback.retrain_every: 50` | done |
| Min-Max normalisation, label encoding, feature selection by RF importance (§6.1) | `hitds/features.py` (+ signed log, ANOVA option) | done |
| Unit/integration/system tests UT/IT/ST (§6.4) | `tests/test_core.py` (ids in comments) | done |
| Performance: <100 ms/prediction, >50 alerts/s (§6.4) | `ms_per_event` in metrics and dashboard | verify on laptop |
| **Results tables in the report (97.2%, 95.1→97.8% F1, 42 ms…)** | were not produced by code; replace from `results*.json` | **must replace** |

## From the reference papers
| Paper | Used for | Where |
|---|---|---|
| Kim, Dán & Zhu, IEEE TIFS 2024 | alert model (ζ, δ), γ(ω) updates, pruned-HMM MSGPRT, Max-Ratio / Max-KL, MTD/MTBFD evaluation | `hitds/hmm.py`, E5 |
| Dutt, Borah & Maitra, IEEE Access 2020 (IS-IDS) | SMAD innate layer: ANOVA feature ranking + three-sigma rule | `hitds/smad.py`, `features.selection: anova` |
| HITL-IDS review draft (your paper) | 500-sample cold start, 50/iteration, SavedRate, belief MSE, paired t-tests, MTD/MTBFD | E4 `--e4-mode cold --repeats`, E5 |
| Gala et al., AI-based NIDS review | detection rate / false alarm rate reporting, dataset caveats | `pipeline.evaluate_predictions` (DR, FAR) |
| Ahmad et al., JCBI 2024 (DB IDS, NSL-KDD, KNN/SVM/DT/CNN) | NSL-KDD baseline comparison point | `load_nslkdd` |
