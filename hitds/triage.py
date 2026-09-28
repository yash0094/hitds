"""Layer 3: entropy-based alert triage (the autonomy boundary).

Each scored event goes to exactly one route:
    AUTO      - confident and low-stakes: logged (benign) or reversible auto-action (e.g. rate-limit)
    REVIEW    - model is unsure (high entropy / low confidence) or the event is anomalous
    ESCALATE  - confident but high-severity: a human must authorise any consequential action

Thresholds are not fixed constants: the entropy cut-off is a percentile of the
validation-set entropy distribution, so the automation/oversight balance can be
tuned to analyst capacity (proposal §24).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def normalized_entropy(P: np.ndarray) -> np.ndarray:
    """H(p) / log(K) in [0, 1]. 0 = certain, 1 = uniform."""
    P = np.clip(P, 1e-12, 1.0)
    H = -(P * np.log(P)).sum(1)
    return H / np.log(P.shape[1]) if P.shape[1] > 1 else np.zeros(len(P))


@dataclass
class Triage:
    classes: list[str]
    severity: dict[str, int]
    always_escalate: list[str]
    entropy_percentile: float = 90
    confidence_floor: float = 0.85
    anomaly_percentile: float = 99.5
    auto_action_max_severity: int = 3
    h_threshold: float = 0.5
    a_threshold: float = np.inf
    stats: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, classes, cfg):
        t = cfg["triage"]
        return cls(classes=list(classes), severity=t["severity"], always_escalate=t["always_escalate"],
                   entropy_percentile=t["entropy_percentile"], confidence_floor=t["confidence_floor"],
                   anomaly_percentile=t["anomaly_percentile"])

    def calibrate(self, P_val: np.ndarray, anomaly_val: np.ndarray | None = None, y_val=None):
        H = normalized_entropy(P_val)
        self.h_threshold = float(np.percentile(H, self.entropy_percentile))
        if anomaly_val is not None:
            ref = anomaly_val if y_val is None else anomaly_val[np.asarray(y_val) == "BENIGN"]
            self.a_threshold = float(np.percentile(ref, self.anomaly_percentile))
        self.stats = {"h_threshold": self.h_threshold, "a_threshold": self.a_threshold,
                      "entropy_percentile": self.entropy_percentile}
        return self

    def set_percentile(self, pct: float, P_val: np.ndarray):
        self.entropy_percentile = pct
        self.h_threshold = float(np.percentile(normalized_entropy(P_val), pct))

    def route(self, P: np.ndarray, anomaly: np.ndarray | None = None, smad_flag: np.ndarray | None = None) -> list[dict]:
        classes = np.array(self.classes)
        H = normalized_entropy(P)
        conf = P.max(1)
        pred = classes[P.argmax(1)]
        sev = np.array([self.severity.get(c, 3) for c in classes], np.float64)
        exp_sev = P @ sev
        anomaly = np.zeros(len(P)) if anomaly is None else anomaly
        smad_flag = np.zeros(len(P), bool) if smad_flag is None else smad_flag
        out = []
        for i in range(len(P)):
            uncertain = H[i] > self.h_threshold or conf[i] < self.confidence_floor
            # the anomaly detector only overrules the classifier when the classifier says "benign":
            # a flow the classifier calls normal but that looks unlike any benign traffic is a zero-day candidate.
            iforest_hit = anomaly[i] > self.a_threshold
            anomalous = (iforest_hit or bool(smad_flag[i])) and pred[i] == "BENIGN"
            reasons = []
            if uncertain:
                reasons.append(f"uncertain (H={H[i]:.2f} > {self.h_threshold:.2f} or conf={conf[i]:.2f} < "
                               f"{self.confidence_floor})")
            if anomalous:
                who = " + ".join(n for n, hit in (("isolation forest", iforest_hit), ("SMAD 3-sigma", smad_flag[i])) if hit)
                reasons.append(f"classified benign but unlike benign baseline ({who}) - possible zero-day")
            if uncertain or anomalous:
                route = "REVIEW"
            elif pred[i] in self.always_escalate or (pred[i] != "BENIGN" and
                                                     self.severity.get(pred[i], 3) > self.auto_action_max_severity):
                route = "ESCALATE"
                reasons.append(f"confident {pred[i]} - severity {self.severity.get(pred[i], 3)} needs human sign-off")
            else:
                route = "AUTO"
                reasons.append("confident benign" if pred[i] == "BENIGN" else
                               f"confident low-severity {pred[i]} - reversible auto-action")
            # queue priority: expected severity, boosted by uncertainty and anomaly
            priority = float(exp_sev[i] * (1.0 + H[i]) + (2.0 if anomalous else 0.0))
            out.append({"pred": str(pred[i]), "confidence": float(conf[i]), "entropy": float(H[i]),
                        "anomaly": float(anomaly[i]), "smad_flag": bool(smad_flag[i]), "route": route, "reason": "; ".join(reasons),
                        "priority": round(priority, 3), "expected_severity": float(exp_sev[i])})
        return out
