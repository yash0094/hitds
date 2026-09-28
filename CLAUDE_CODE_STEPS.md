# Step-by-step: from this zip to a deployable HITDS with Claude Code in VS Code

Paste each **Prompt** into the Claude Code panel in VS Code, one step at a time. Wait for it to finish and
report before moving on. If a step fails, paste the error back and say "fix the cause, then re-run the check".

---

## Before you start (once, about 20 min)
1. Install **Python 3.11** from python.org. On the first screen, tick "Add python.exe to PATH".
2. Install **Git** (git-scm.com) and **VS Code** with the **Claude Code** extension, and sign in.
3. Optional, for step 10: **Docker Desktop**, and `winget install --id Cloudflare.cloudflared` for a public demo link.
4. Unzip `HITDS_project.zip` into e.g. `C:\Projects\hitds` (a path without spaces is easiest).
5. VS Code → *File → Open Folder* → select the `hitds` folder → trust the workspace → open the Claude Code panel.
6. Keep the laptop plugged in. Training and experiments use all 4 CPU threads for long stretches.

---

## Step 0: Orientation and environment
**Prompt**
> Read CLAUDE.md, BUILD_PLAN.md, docs/REQUIREMENTS_TRACEABILITY.md and docs/PAPERS.md. Then do Phase 0: create a
> .venv with Python 3.11, install CPU-only PyTorch, requirements.txt and waitress, and run pytest. All tests must pass
> with none skipped. Initialise a git repo and make a first commit. Tell me what you installed and the test result.

**You should see**: all tests passed, 0 skipped, and `torch.cuda.is_available()` → False.

## Step 1: Offline smoke demo
**Prompt**
> Do Phase 1 of BUILD_PLAN.md. Start the synthetic demo with run_demo.ps1 and tell me when to open the browser.

**You do**: open http://127.0.0.1:5000, log in as `analyst` / `hitds-demo`, click *Start traffic*, and review a few
alerts (J → look at the evidence → T or F → Enter). Pick "unsure" on one, then open a private window, log in as
`reviewer2` / `hitds-review` and finish it. Tell Claude Code "looks good" or what looked wrong.

## Step 2: Download the datasets from Hugging Face
**Prompt**
> Do Phase 2: download all three datasets, show me the MANIFEST.json summaries, and pin the dataset revisions in config.yaml.

## Step 3: Clean and split
**Prompt**
> Do Phase 3. Show me each DATA_CARD.md summary: rows removed at every cleaning step, final class counts, and the leakage
> check. Verify the NSL-KDD normal_value and fix config.yaml if needed. Also create the zero-day split (holdout Infiltration, tag zeroday).

**Read the data cards yourself.** They show how many duplicate flows CIC-IDS2017 contains, which is a strong point for the judges.

## Step 4: Train the models
**Prompt**
> Do Phase 4 for cicids2017, then unsw_nb15, then nslkdd. For each, report macro-F1, detection rate, false alarm rate,
> per-class recall, ms per event and training time. Keep training under 25 minutes per dataset by adjusting config caps if needed.

## Step 5: CTGAN vs SMOTE
**Prompt**
> Do Phase 5. Time a 50-epoch CTGAN run first, tell me the estimate for more epochs, and ask me before running anything
> longer than 45 minutes. Keep whichever augmentation validates better as the default and retrain v1.

## Step 6: Run the experiments (leave the laptop running)
**Prompt**
> Do Phase 6. Run the experiment commands one after another, then write reports/SUMMARY.md exactly as described in the
> plan: measured numbers only, oracle-analyst caveat, and where the human loop did not help. Show me SUMMARY.md at the end.

## Step 7: Cross-dataset test
**Prompt**
> Do Phase 7 and add the cross-dataset table to SUMMARY.md with a two-sentence interpretation.

## Step 8: Real traffic path
**Prompt**
> Do Phase 8. Set HITDS_API_KEY, start the server, send 2,000 rows of the Friday PortScan CSV through
> scripts/capture_live.py, and upload a 300-row CSV through the UI. Report the feature coverage and what showed up in the queue.

## Step 9: Fix the report numbers
**Prompt**
> Do Phase 9. Compare the results in my project report, FAER proposal and draft paper with the measured results and
> write reports/REPORT_UPDATES.md with replacement text and tables. Do not edit my Word or PDF files yet; ask me first.

(Put the three documents into the project's `docs/` folder first so Claude Code can read them.)

## Step 10: Deploy
**Prompt**
> Do Phase 10. Export the deploy bundle for cicids2017, start run_demo.ps1 -Public, and give me the public URL to test
> from my phone. If Docker Desktop is running, also build and run the Docker image and confirm /healthz. Show me
> docs/DEPLOY.md's options with costs and ask before setting up anything paid.

## Step 11: Rehearse the live demo
**Prompt**
> Do Phase 11 with me. Start a fresh session. I will review about 60 alerts (some blind, some unsure). Then trigger a retrain,
> run 05_report_live.py, and time a full run-through of docs/DEMO_SCRIPT.md. List anything that felt slow or broke.

---

### If something goes wrong
- *"pip can't find torch"* → use Python 3.11 or 3.12. The torch CPU wheel does not support every new Python release.
- *MemoryError during prepare/train* → ask Claude Code to lower `max_benign` / `max_per_class` in `config.yaml`.
- *Training too slow* → "use --members rf,mlp for experiments and lower svm.max_train_rows".
- *Browser shows a login loop* → set `HITDS_SECRET_KEY` (run_demo.ps1 does it for you).
- *Queue empty* → click *Start traffic* or *+30*. Only uncertain or high-severity events reach the queue, by design.
