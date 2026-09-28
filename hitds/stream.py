"""Simulated live traffic for the demo and for the Layer-4 evaluation: replays held-out test flows
as if they were arriving on a small network.

The public CSVs have no usable host identity, so hosts/IPs are assigned here. Attack rows are
grouped into *campaigns* against one victim host that follow the kill-chain order
(intrusion -> exploit -> impact), interleaved with that host's normal traffic. Each row carries
the victim's true attack stage (`gt_stage`), which is what time-to-detection and belief-MSE are
measured against. Benign rows are spread over all hosts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hmm import STAGE_OF_CLASS


class TrafficReplay:
    def __init__(self, test_df: pd.DataFrame, n_hosts: int = 12, seed: int = 42, attack_share: float = 0.25,
                 max_campaigns: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.df = test_df.reset_index(drop=True)
        self.hosts = [f"srv-{i:02d}" for i in range(n_hosts)]
        self.host_ip = {h: f"10.0.1.{10 + i}" for i, h in enumerate(self.hosts)}
        lab = self.df["label"].to_numpy()
        self.benign_idx = np.flatnonzero(lab == "BENIGN")
        self.attack_idx = {c: np.flatnonzero(lab == c) for c in np.unique(lab) if c != "BENIGN"}
        self.attack_share = attack_share
        self.max_campaigns = max_campaigns
        self.n_campaigns = 0
        self.stage = {h: 0 for h in self.hosts}       # true attack stage per host
        self.campaign_of = {h: None for h in self.hosts}
        self._queue: list[tuple[int, str, str, str, int]] = []

    def _external_ip(self):
        r = self.rng.integers
        return f"{r(11, 223)}.{r(0, 255)}.{r(0, 255)}.{r(1, 254)}"

    def _new_campaign(self) -> bool:
        if self.max_campaigns is not None and self.n_campaigns >= self.max_campaigns:
            return False
        clean = [h for h in self.hosts if self.stage[h] == 0]
        if not clean or not self.attack_idx:
            return False
        victim = clean[self.rng.integers(len(clean))]
        attacker = self._external_ip()
        classes = list(self.attack_idx)
        k = int(self.rng.integers(1, min(3, len(classes)) + 1))
        chosen = sorted(self.rng.choice(classes, size=k, replace=False), key=lambda c: STAGE_OF_CLASS.get(c, 2))
        self.n_campaigns += 1
        cid = self.n_campaigns
        for c in chosen:
            n = int(self.rng.integers(4, 14))
            for i in self.rng.choice(self.attack_idx[c], size=n, replace=True):
                self._queue.append((int(i), victim, attacker, self.host_ip[victim], cid))
        return True

    def next_batch(self, n: int) -> pd.DataFrame:
        rows = []
        for _ in range(n):
            if self.rng.random() < self.attack_share and (self._queue or self._new_campaign()):
                i, host, src, dst, cid = self._queue.pop(0)
                self.stage[host] = max(self.stage[host], STAGE_OF_CLASS.get(self.df["label"].iat[i], 2))
                self.campaign_of[host] = cid
                rows.append((i, host, src, dst, self.stage[host], cid))
            else:
                h = self.hosts[self.rng.integers(len(self.hosts))]
                src = self.host_ip[self.hosts[self.rng.integers(len(self.hosts))]] if self.rng.random() < .6 \
                    else self._external_ip()
                rows.append((int(self.rng.choice(self.benign_idx)), h, src, self.host_ip[h], self.stage[h],
                             self.campaign_of[h]))
        out = self.df.iloc[[r[0] for r in rows]].copy().reset_index(drop=True)
        out["ground_truth"] = out["label"]
        out = out.drop(columns=["label", "label_raw"])
        out["host"] = [r[1] for r in rows]
        out["src_ip"] = [r[2] for r in rows]
        out["dst_ip"] = [r[3] for r in rows]
        out["gt_stage"] = [r[4] for r in rows]
        out["campaign"] = [r[5] for r in rows]
        return out
