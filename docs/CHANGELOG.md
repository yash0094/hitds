# Changelog

## v0.2: precise data collection, paper-faithful Layer 4, deployable app
- **Data**: SHA-256 manifest + revision pinning; NaN/inf rows dropped (not zero-filled); impossible negative values
  dropped; exact duplicates and label-conflicting rows removed before splitting; post-split leakage check;
  `DATA_CARD.md` per dataset; `--tag` for separate split variants (e.g. zero-day).
- **Collection**: CICFlowMeter (Java and Python) column mapping with coverage report; CSV/JSON upload in the console;
  API-key ingest endpoint; `scripts/capture_live.py` (PCAP, follow a live CSV, send a file, Suricata EVE merge).
- **Layer 4** rewritten after Kim, Dán & Zhu (IEEE TIFS 2024): ζ/δ alert model estimated from the validation
  confusion matrix, γ(ω) investigation updates, pruned-hypothesis sequential test, Max-KL / Max-Ratio queue ordering.
- **IS-IDS SMAD** innate layer (ANOVA + three-sigma), shown as evidence and used as a zero-day trigger; ANOVA feature
  selection option.
- **Review**: second review for unsure verdicts (a different login), final label from the second reviewer, Cohen's kappa,
  blind vs revealed agreement (automation-bias monitor), audit export.
- **Experiments**: E4 cold start (500 labelled, HITL review protocol), repeats + paired t-tests, SavedRate; E5 as a Kim et
  al. evaluation (no investigation / static / Max-Ratio / Max-KL → MTD, MTBFD, detection rate, belief MSE); DR/FAR
  everywhere; cross-dataset script.
- **Deployment**: login + CSRF + API key, waitress `serve.py`, `/healthz`, Dockerfile (optional torch; MLP scores in
  NumPy), docker-compose, Render blueprint, Hugging Face Space template, `run_demo.ps1 -Public` (Cloudflare quick tunnel),
  deploy export script, GitHub Actions CI.
- Verified in a sandbox on synthetic data: 20/21 tests pass (the torch-parity test is skipped without torch).

## v0.1: initial scaffold
- Pipeline, dashboard, experiments, tests. Finding from the synthetic dry run: pure entropy sampling missed a
  held-out class that the model was confidently wrong about; hybrid (entropy + anomaly) sampling found it.
