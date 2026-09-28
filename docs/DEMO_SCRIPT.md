# Live demo script (about 7-8 minutes)

**Before judges arrive**: `.\run_demo.ps1` (add `-Public` if they want the link). Log in, go full screen, and click **+30** twice
so the queue isn't empty. Have a second browser profile logged in as `reviewer2`.

1. **The problem (30 s).** "SOCs drown in alerts. Most are false, and analysts stop trusting the tools. We don't automate the
   analyst away. We treat analyst attention as the scarce resource and measure it."
2. **Traffic (30 s).** *Start traffic* at 5/s. KPIs: share handled autonomously vs sent to a human, ms per event.
3. **Open the top alert, J (90 s).** Top to bottom: *why you're seeing this* (entropy / confidence / zero-day reason) →
   raw flow evidence → SHAP attribution → **SMAD** 3-sigma features (IS-IDS innate layer) → similar incidents → **host HMM**
   belief and log-likelihood ratio (Kim et al.'s sequential test). "The evidence comes before the model's opinion."
4. **Blind review (30 s).** Tick *Blind review*: the model's verdict stays hidden until R. The header shows agreement blind vs
   after reveal. That comparison is our automation-bias measurement.
5. **Decide (60 s).** T/F/U, confidence, authorise the suggested action, Enter. Choose "unsure" once: the alert goes to
   **second review**, and you finish it as reviewer2. Mention Cohen's kappa in the header.
6. **Real traffic (45 s).** *Upload flows* with a small CICFlowMeter CSV. Point out the feature-coverage toast and the
   `upload:` source tag.
7. **Audit (45 s).** Audit panel: hash chain verified; *Export audit log*. "Every score, verdict and action is chained, and
   editing a row breaks the chain. That is the evidence ISO 27001 / SOC 2 / CERT-In expect."
8. **The loop (60 s).** Feedback bar → *Retrain now*. The history shows validation F1 old→new, the human-share change,
   and the promotion gate's decision.
9. **Results (60 s).** `reports/.../E3_autonomy_curve.png`, the E4 figures (entropy vs hybrid vs random, with t-tests) and
   E5 (MTD by policy). Measured numbers only; call the analyst in E3-E5 an oracle.

Keys: **J** next · **T** malicious · **F** benign · **U** need info · **R** reveal · **Enter** submit.
If asked "is it blocking real traffic?": no. Actions are simulated and logged; enforcement is future work.
