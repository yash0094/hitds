"""Feed-forward neural network member of the ensemble (PyTorch, CPU-friendly).

Exposes a scikit-learn style API (fit / predict_proba / classes_) so it can sit
inside the soft-voting ensemble next to RF, SVM and KNN.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # the test container has no torch; the laptop will
    HAS_TORCH = False


if HAS_TORCH:

    class _Net(nn.Module):
        def __init__(self, n_in: int, n_out: int, hidden: list[int], dropout: float):
            super().__init__()
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, n_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


class TorchMLP:
    def __init__(self, hidden=(256, 128, 64), dropout=0.2, epochs=25, batch_size=512, lr=1e-3, seed=42,
                 verbose=True):
        self.hidden, self.dropout, self.epochs = list(hidden), dropout, epochs
        self.batch_size, self.lr, self.seed, self.verbose = batch_size, lr, seed, verbose
        self.classes_ = None
        self._state = None
        self._n_in = None
        self._fallback = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        self.classes_ = np.unique(y)
        if not HAS_TORCH:
            from sklearn.neural_network import MLPClassifier

            self._fallback = MLPClassifier(hidden_layer_sizes=tuple(self.hidden), max_iter=self.epochs * 4,
                                           early_stopping=True, random_state=self.seed)
            self._fallback.fit(X, np.searchsorted(self.classes_, y))  # int labels (sklearn string-label bug)
            return self

        torch.manual_seed(self.seed)
        idx = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([idx[c] for c in y], dtype=np.int64)
        w = np.ones(len(y), np.float32) if sample_weight is None else sample_weight.astype(np.float32)
        # class-balanced loss: rare classes are worth more per row
        counts = np.bincount(yi, minlength=len(self.classes_)).astype(np.float32)
        cw = torch.tensor((counts.sum() / (len(counts) * np.maximum(counts, 1))) ** 0.5)

        self._n_in = X.shape[1]
        net = _Net(self._n_in, len(self.classes_), self.hidden, self.dropout)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss(weight=cw, reduction="none")
        Xt, yt, wt = torch.from_numpy(X.astype(np.float32)), torch.from_numpy(yi), torch.from_numpy(w)
        n = len(Xt)
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(n)
            total = 0.0
            for s in range(0, n, self.batch_size):
                b = perm[s: s + self.batch_size]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                loss = (loss_fn(net(Xt[b]), yt[b]) * wt[b]).mean()
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            sched.step()
            if self.verbose and (ep % 5 == 0 or ep == self.epochs - 1):
                print(f"    mlp epoch {ep + 1:3d}/{self.epochs}  loss {total / n:.4f}")
        # weights stored as numpy so a deployed app can score without PyTorch installed
        self._state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        return self

    # ------------------------------------------------------------------ predict
    def _numpy_forward(self, X: np.ndarray) -> np.ndarray:
        """Eval-mode forward pass in NumPy (Linear -> BatchNorm -> ReLU per hidden layer, Dropout is a no-op)."""
        st = {k: np.asarray(v) for k, v in self._state.items()}
        h = X.astype(np.float32)
        idx = 0
        for _ in self.hidden:
            h = h @ st[f"net.{idx}.weight"].T + st[f"net.{idx}.bias"]
            bn = idx + 1
            h = (h - st[f"net.{bn}.running_mean"]) / np.sqrt(st[f"net.{bn}.running_var"] + 1e-5)
            h = h * st[f"net.{bn}.weight"] + st[f"net.{bn}.bias"]
            h = np.maximum(h, 0.0)
            idx += 4
        logits = h @ st[f"net.{idx}.weight"].T + st[f"net.{idx}.bias"]
        logits = logits - logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict_proba(X)
        return np.vstack([self._numpy_forward(X[s: s + 8192]) for s in range(0, len(X), 8192)])

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
