# How the reference papers map to the code

## Kim, Dán & Zhu, "Human-in-the-Loop Cyber Intrusion Detection Using Active Learning", IEEE TIFS 19 (2024)
Implemented in `hitds/hmm.py`, evaluated in E5.

| Paper element | Implementation |
|---|---|
| Attack graph as HMM states | `STATES = safe, intrusion, exploit, impact`; transition matrix `hmm.A` in config |
| Alert j fires in state i with prob δ_j,i; false alerts ζ_j | estimated from the validation confusion matrix (`AlertModel.from_confusion`) |
| P(Y_j=1 \| S=i) = 1-(1-ζ)(1-δ) (eq. 4) | `HostHMM._obs_lik` |
| Investigation outcome updates via γ(ω) (eqs. 1-3), linear γ (eq. 2) | `HostHMM.investigate`, `gamma_linear` |
| Pruned hypotheses + MSGPRT, R = p_ĥ / p_1 vs θ (eq. 5) | `HostHMM.log_ratio`, `hmm.log_threshold` |
| Belief (eq. 8) | `HostHMM.belief` |
| Max Ratio and Max KL policies, approximation (18) | `score_max_ratio`, `score_max_kl`, `hmm.policy` |
| Baselines: no investigation, static priority | E5 `none`, `static` |
| Metrics: MTD, MTBFD | E5 `MTD_events`, `MTBFD_events` |

Deviations (state these in the report): the 4-stage chain replaces the paper's 9-alert DoS graph; one
alert per event (the predicted class) instead of many simultaneous binary alerts; the MKL score uses the
investigated alert's Bernoulli KL rather than the full sum in eq. 16; per-host history is windowed; the
simulated analyst errs with probability ω on every alert.

## Dutt, Borah & Maitra, "Immune System Based IDS (IS-IDS)", IEEE Access 8 (2020)
SMAD (the innate layer) uses ANOVA to find the characteristics that separate normal from suspicious
behaviour, and a three-sigma test to raise first-hand suspicion. In `hitds/smad.py`, ANOVA F ranks the
features, benign mean ± 3σ bands are fitted, and the score is the share of features outside the band.
It is shown to the analyst and triggers review when the ensemble says BENIGN. The adaptive layer (AIAD:
T-cell/B-cell file and user checks) is host-based and not applicable to flow data; not implemented.

## HITL-IDS review draft (your paper)
Protocol taken from its methodology section: an initial labelled pool of 500, 50 high-entropy samples per
iteration, SavedRate = 1 - annotated/all, paired t-tests across random splits, MTD, MTBFD, belief MSE
(`scripts/04_experiments.py` E4 `--e4-mode cold --repeats 10`, E5). Its CTGAN settings (3,000 epochs,
batch 1,500 on a TPU) are far beyond a 2-core laptop; `augment.ctgan_epochs` is scaled down, and that is
a stated limitation. The draft's abstract claims results ("accuracy up to 99%, annotation cost reduction of
up to 98%"). Rewrite them as targets, or replace them with the measured E4 numbers.

## Gala et al., "AI-Based Techniques for Network-Based IDS: A Review"
Detection rate and false alarm rate are reported next to macro-F1 (`DR`, `FAR` in every table). The
review's dataset caveats (age of KDD-family data) support treating NSL-KDD only as a legacy baseline.

## Ahmad et al., "Enhancing Database Security through AI-Based IDS", JCBI 7(2) (2024)
KNN/SVM/DT/CNN on NSL-KDD (reported 98.4% for CNN). It is useful only as a published comparison point
for the NSL-KDD baseline. Compare on the same split definition, or say that the splits differ.
