"""Layer 4: situational awareness per host, after Kim, Dán & Zhu,
"Human-in-the-Loop Cyber Intrusion Detection Using Active Learning", IEEE TIFS 19 (2024) 8658-8672.

Model (paper §III), with our classifier's output as the alert source:
  * attack states  S = {safe, intrusion, exploit, impact}   (left-to-right attack graph)
  * alert types    J = the non-benign classes of the ensemble; each scored flow on a host is one
                   time step whose alert vector Y_t is the one-hot predicted class (all zeros if BENIGN)
  * false-alert probability  zeta_j   = P(pred = j | benign traffic)          <- validation confusion matrix
  * true-alert probability   delta_j,i = rho_i * P(pred = j | attack traffic of stage i)   (delta_j,safe = 0)
  * P(Y_j = 1 | S = s_i) = 1 - (1 - zeta_j)(1 - delta_j,i)                      (paper eq. 4)
  * analyst investigation of alert (t1, j1) with error probability omega updates that alert's
    zeta/delta through the confidence function gamma(omega)                     (paper eqs. 1-3)
  * hypotheses h = 1..4 are pruned HMMs (h keeps the first h states). The defender runs a
    multi-hypothesis sequential probability ratio test: R = p_hhat / p_1 vs threshold theta (eq. 5)
  * alert prioritisation policies (paper §IV): Max Ratio (MR) - largest expected change of the
    likelihood ratio - and Max KL (MKL) - largest expected change of the KL divergence between the
    alert distributions under hhat and h1. Both use the paper's approximation (18): the effect of
    investigating an alert observed at t1 is evaluated with the forward variables at t1.

Deviations from the paper, stated so they can be reported honestly:
  * the attack graph is the 4-stage chain above, not the paper's 9-alert DoS graph;
  * one alert (the predicted class) per time step instead of many simultaneous binary alerts;
  * the MKL score uses the KL divergence of the investigated alert's Bernoulli distribution
    (not the full sum over t' in eq. 16) - cheaper, same ranking intent;
  * history per host is windowed (config `hmm.window`); older steps are folded into the prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STATES = ["safe", "intrusion", "exploit", "impact"]
STAGE_OF_CLASS = {
    "BENIGN": 0,
    "PortScan": 1, "Reconnaissance": 1, "Fuzzers": 1, "Analysis": 1,
    "BruteForce": 2, "WebAttack": 2, "Infiltration": 2, "Exploits": 2, "Backdoor": 2, "Shellcode": 2, "ATTACK": 2,
    "DoS": 3, "DDoS": 3, "Bot": 3, "Heartbleed": 3, "Worms": 3, "Generic": 3,
}
DEFAULT_A = np.array([
    [0.995, 0.004, 0.0008, 0.0002],
    [0.0, 0.93, 0.06, 0.01],
    [0.0, 0.0, 0.92, 0.08],
    [0.0, 0.0, 0.0, 1.0],
])


def gamma_linear(omega: float, gamma0: float) -> float:
    """Paper eq. (2), linear confidence function, omega in [0, 0.5], gamma0 > 1."""
    return 2.0 * (1.0 - gamma0) * omega + gamma0


def bern_kl(p: float, q: float) -> float:
    p, q = min(max(p, 1e-9), 1 - 1e-9), min(max(q, 1e-9), 1 - 1e-9)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


# --------------------------------------------------------------------------- alert model from data
@dataclass
class AlertModel:
    alert_types: list[str]          # J
    zeta: np.ndarray                # (J,)
    delta: np.ndarray               # (J, I)

    @classmethod
    def from_confusion(cls, classes: list[str], y_true: np.ndarray, y_pred: np.ndarray, rho=(0.0, 0.5, 0.6, 0.8)):
        """Estimate zeta and delta from a labelled validation set."""
        alert_types = [c for c in classes if c != "BENIGN"]
        J, I = len(alert_types), len(STATES)
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        benign = y_true == "BENIGN"
        zeta = np.array([np.mean(y_pred[benign] == j) if benign.any() else 0.01 for j in alert_types])
        delta = np.zeros((J, I))
        for i in range(1, I):
            cls_i = [c for c in set(y_true) if STAGE_OF_CLASS.get(c, 2) == i and c != "BENIGN"]
            mask = np.isin(y_true, cls_i) if cls_i else (~benign)
            if not mask.any():
                continue
            for k, j in enumerate(alert_types):
                delta[k, i] = rho[i] * np.mean(y_pred[mask] == j)
        return cls(alert_types, np.clip(zeta, 1e-4, 0.5), np.clip(delta, 0.0, 0.999))


# --------------------------------------------------------------------------- per-host HMM
@dataclass(eq=False)          # identity comparison: steps hold numpy arrays
class Step:
    j: int                         # alert type index, -1 = no alert (benign)
    zeta: np.ndarray               # (J,) possibly investigation-adjusted
    delta: np.ndarray              # (J, I)
    alert_id: int | None = None
    investigated: bool = False
    alpha: list = field(default_factory=list)   # per hypothesis: normalised forward vector at this step
    logp: np.ndarray | None = None              # per hypothesis: log P(Y_1:t)


class HostHMM:
    def __init__(self, model: AlertModel, A: np.ndarray, window: int = 400):
        self.m, self.window = model, window
        self.I = len(STATES)
        self.H = self.I
        self.A_h = []
        for h in range(1, self.H + 1):
            Ah = A[:h, :h].copy()
            Ah = Ah / Ah.sum(1, keepdims=True)
            self.A_h.append(Ah)
        self.init = [np.eye(h)[0] for h in range(1, self.H + 1)]   # start in 'safe'
        self.steps: list[Step] = []
        self.logp = np.zeros(self.H)
        self.alpha = [v.copy() for v in self.init]

    def _obs_lik(self, st: Step) -> np.ndarray:
        """P(Y_t | S = i) for all states, product over alert types (paper eq. 4)."""
        fire = 1.0 - (1.0 - st.zeta[:, None]) * (1.0 - st.delta)        # (J, I)
        lik = np.prod(1.0 - fire, axis=0)                                # all silent
        if st.j >= 0:
            lik = lik / np.maximum(1.0 - fire[st.j], 1e-12) * fire[st.j]
        return np.maximum(lik, 1e-300)

    def _forward_step(self, st: Step, alpha_prev, logp_prev):
        lik = self._obs_lik(st)
        alphas, logps = [], np.empty(self.H)
        for h in range(self.H):
            a = (alpha_prev[h] @ self.A_h[h]) * lik[: h + 1]
            s = a.sum()
            alphas.append(a / s)
            logps[h] = logp_prev[h] + np.log(s)
        st.alpha, st.logp = alphas, logps
        return alphas, logps

    def observe(self, j: int, alert_id: int | None) -> Step:
        st = Step(j=j, zeta=self.m.zeta.copy(), delta=self.m.delta.copy(), alert_id=alert_id)
        self.alpha, self.logp = self._forward_step(st, self.alpha, self.logp)
        self.steps.append(st)
        if len(self.steps) > self.window:
            self._fold(len(self.steps) // 2)
        return st

    def _fold(self, k: int):
        """Drop the oldest k steps; their evidence is kept as the new initial distribution + offset."""
        cut = self.steps[k - 1]
        self.init = [a.copy() for a in cut.alpha]
        self._base_logp = cut.logp.copy()
        self.steps = self.steps[k:]

    def recompute(self, from_idx: int = 0):
        alpha = [a.copy() for a in (self.steps[from_idx - 1].alpha if from_idx > 0 else self.init)]
        logp = (self.steps[from_idx - 1].logp.copy() if from_idx > 0 else getattr(self, "_base_logp", np.zeros(self.H)).copy())
        for st in self.steps[from_idx:]:
            alpha, logp = self._forward_step(st, alpha, logp)
        self.alpha, self.logp = alpha, logp

    # ------------------------------------------------------------------ inference
    def belief(self, alpha=None, logp=None) -> np.ndarray:
        """Paper eq. (8): belief over states, marginalised over hypotheses (uniform prior)."""
        alpha = self.alpha if alpha is None else alpha
        logp = self.logp if logp is None else logp
        w = np.exp(logp - logp.max())
        b = np.zeros(self.I)
        for h in range(self.H):
            b[: h + 1] += w[h] * alpha[h]
        return b / b.sum()

    def log_ratio(self, logp=None) -> tuple[float, int]:
        """log R_t = log p_hhat - log p_1, hhat = argmax_{h>1} p_h (paper eq. 5)."""
        logp = self.logp if logp is None else logp
        hhat = int(np.argmax(logp[1:])) + 1
        return float(logp[hhat] - logp[0]), hhat

    # ------------------------------------------------------------------ investigation (paper eqs. 1-3)
    def investigate(self, idx: int, outcome: int, omega: float, gamma0: float):
        st = self.steps[idx]
        if st.j < 0:
            return
        g = gamma_linear(omega, gamma0)
        j = st.j
        if outcome == 1:     # analyst says true positive
            st.zeta[j] = st.zeta[j] / g
            st.delta[j] = np.minimum(g * st.delta[j], 0.999)
        else:                # analyst says false positive
            st.zeta[j] = min(g * st.zeta[j], 1.0 - 1e-6)
            st.delta[j] = st.delta[j] / g
        st.investigated = True
        self.recompute(idx)

    # ------------------------------------------------------------------ policies (paper §IV, approx. 18)
    def _outcome_terms(self, idx: int, omega: float, gamma0: float):
        st = self.steps[idx]
        j = st.j
        g = gamma_linear(omega, gamma0)
        z, d = st.zeta[j], st.delta[j]                            # scalar, (I,)
        f = 1.0 - (1.0 - z) * (1.0 - d)                            # P(alert fires | i)
        p_true = 1.0 - (z * (1.0 - d)) / np.maximum(f, 1e-12)     # P(alert is a true positive | fired, i)
        pi = self.belief(st.alpha, st.logp)
        p_o1 = float(pi @ (p_true * (1 - omega) + (1 - p_true) * omega))
        after = {}
        for o, (z2, d2) in {1: (z / g, np.minimum(g * d, 0.999)), 0: (min(g * z, 1 - 1e-6), d / g)}.items():
            f2 = 1.0 - (1.0 - z2) * (1.0 - d2)
            after[o] = f2 / np.maximum(f, 1e-12)                     # c_i(o): likelihood change factor
        return st, f, {1: p_o1, 0: 1 - p_o1}, after

    def score_max_ratio(self, idx: int, omega: float, gamma0: float) -> float:
        """log |E[R_t1 after investigation] - R_t1| (ranked descending)."""
        st, f, po, c = self._outcome_terms(idx, omega, gamma0)
        logR, hhat = self.log_ratio(st.logp)
        a = st.alpha[hhat]
        ratio = sum(po[o] * float(a @ c[o][: hhat + 1]) / c[o][0] for o in (0, 1))
        return float(logR + np.log(abs(ratio - 1.0) + 1e-12))

    def score_max_kl(self, idx: int, omega: float, gamma0: float) -> float:
        """|E[D(q_hhat || q_1) after] - D(q_hhat || q_1)| for the investigated alert's Bernoulli."""
        st, f, po, c = self._outcome_terms(idx, omega, gamma0)
        _, hhat = self.log_ratio(st.logp)
        a_h, a_1 = st.alpha[hhat], st.alpha[0]
        before = bern_kl(float(a_h @ f[: hhat + 1]), float(a_1 @ f[:1]))
        exp_after = 0.0
        for o in (0, 1):
            f2 = f * c[o]
            exp_after += po[o] * bern_kl(float(a_h @ f2[: hhat + 1]), float(a_1 @ f2[:1]))
        return abs(exp_after - before)


# --------------------------------------------------------------------------- manager used by the engine
class HostTracker:
    """All hosts + alert-id index. Keeps the dict shape the dashboard expects."""

    def __init__(self, classes: list[str], model: AlertModel | None = None, cfg: dict | None = None):
        c = (cfg or {}).get("hmm", {})
        self.classes = list(classes)
        self.model = model or AlertModel([k for k in classes if k != "BENIGN"],
                                         np.full(len(classes) - 1, 0.01),
                                         np.tile([0.0, 0.3, 0.4, 0.5], (len(classes) - 1, 1)))
        self.A = np.array(c.get("A", DEFAULT_A.tolist()), float)
        self.log_theta = float(c.get("log_threshold", 8.0))
        self.omega = float(c.get("analyst_error", 0.1))
        self.gamma0 = float(c.get("gamma0", 5.0))
        self.window = int(c.get("window", 400))
        self.policy = c.get("policy", "max_kl")
        self.hosts: dict[str, HostHMM] = {}
        self.index: dict[int, tuple[str, Step]] = {}
        self.jidx = {k: i for i, k in enumerate(self.model.alert_types)}

    def _host(self, host: str) -> HostHMM:
        if host not in self.hosts:
            self.hosts[host] = HostHMM(self.model, self.A, self.window)
        return self.hosts[host]

    def update(self, host: str, pred: str, alert_id: int | None = None) -> dict:
        hm = self._host(host)
        st = hm.observe(self.jidx.get(pred, -1), alert_id)
        if alert_id is not None and st.j >= 0:
            self.index[alert_id] = (host, st)
        return self.state(host)

    def bind(self, host: str, step: Step, alert_id: int):
        step.alert_id = alert_id
        if step.j >= 0:
            self.index[alert_id] = (host, step)

    def last_step(self, host: str) -> Step:
        return self.hosts[host].steps[-1]

    def investigate(self, alert_id: int, outcome: int) -> dict | None:
        if alert_id not in self.index:
            return None
        host, st = self.index[alert_id]
        hm = self.hosts[host]
        try:
            idx = hm.steps.index(st)
        except ValueError:          # folded out of the window
            return None
        hm.investigate(idx, outcome, self.omega, self.gamma0)
        return self.state(host)

    def policy_score(self, alert_id: int, policy: str | None = None) -> float | None:
        policy = policy or self.policy
        if alert_id not in self.index or policy == "static":
            return None
        host, st = self.index[alert_id]
        hm = self.hosts[host]
        if st.investigated or st not in hm.steps:
            return None
        idx = hm.steps.index(st)
        if policy == "max_ratio":
            return hm.score_max_ratio(idx, self.omega, self.gamma0)
        return hm.score_max_kl(idx, self.omega, self.gamma0)

    def state(self, host: str) -> dict:
        hm = self.hosts.get(host)
        if hm is None:
            b, logR, hhat = np.eye(4)[0], 0.0, 1
        else:
            b = hm.belief()
            logR, hhat = hm.log_ratio()
        detected = logR > self.log_theta
        prior = np.array([0.97, 0.01, 0.01, 0.01])
        return {"host": host, "belief": {s: round(float(v), 4) for s, v in zip(STATES, b)},
                "stage": STATES[int(np.argmax(b))],
                "max_kl": round(float(np.sum(b * np.log(np.maximum(b, 1e-12) / prior))), 4),
                "llr": round(logR, 3), "hypothesis": STATES[hhat] if detected else None,
                # the MSGPRT only confirms attack hypotheses against the null; "safe" is a belief-based display label
                "sprt": "compromised" if detected else ("safe" if b[0] >= 0.95 else "undecided")}

    def ranking(self) -> list[dict]:
        return sorted((self.state(h) for h in self.hosts), key=lambda s: -s["llr"])

    def reset_host(self, host: str):
        self.hosts.pop(host, None)
        self.index = {k: v for k, v in self.index.items() if v[0] != host}
