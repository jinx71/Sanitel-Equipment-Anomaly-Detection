"""A small autoencoder implemented from scratch in NumPy.

Written deliberately without a deep-learning framework so the reconstruction-
error mechanism (forward pass, MSE loss, manual backpropagation, Adam updates)
is fully explicit and the project stays dependency-light enough to deploy on a
free tier. The network compresses standardised sensor vectors through a narrow
bottleneck and is trained to reconstruct them; the per-sample reconstruction
error is the anomaly score.

Architecture:  d -> 8 (tanh) -> bottleneck (tanh) -> 8 (tanh) -> d (linear)
"""

from __future__ import annotations

import numpy as np

from .base import BaseDetector


def _tanh(x):
    return np.tanh(x)


def _dtanh(a):
    # derivative of tanh expressed via its output a = tanh(x)
    return 1.0 - a * a


class _Adam:
    """Minimal Adam optimiser over a dict of named parameters."""

    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            m_hat = self.m[k] / (1 - self.b1 ** self.t)
            v_hat = self.v[k] / (1 - self.b2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AutoencoderDetector(BaseDetector):
    name = "autoencoder"
    label = "Autoencoder (NumPy)"
    description = (
        "Neural reconstruction model, hand-implemented in NumPy. Compresses each "
        "reading through a narrow bottleneck and reconstructs it; anomalies that "
        "do not fit the learned manifold reconstruct poorly and score higher. "
        "Captures non-linear sensor relationships the linear PCA model cannot."
    )

    def __init__(
        self,
        bottleneck: int = 2,
        hidden: int = 8,
        epochs: int = 400,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    def _init_params(self, d: int) -> dict:
        rng = np.random.default_rng(self.seed)
        h, b = self.hidden, max(1, min(self.bottleneck, d - 1))

        def w(shape):  # Xavier/Glorot-style init
            fan_in, fan_out = shape
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=shape)

        return {
            "W1": w((d, h)), "b1": np.zeros(h),
            "W2": w((h, b)), "b2": np.zeros(b),
            "W3": w((b, h)), "b3": np.zeros(h),
            "W4": w((h, d)), "b4": np.zeros(d),
        }

    @staticmethod
    def _forward(p, X):
        a1 = _tanh(X @ p["W1"] + p["b1"])
        a2 = _tanh(a1 @ p["W2"] + p["b2"])   # bottleneck
        a3 = _tanh(a2 @ p["W3"] + p["b3"])
        out = a3 @ p["W4"] + p["b4"]         # linear output
        return a1, a2, a3, out

    def _backward(self, p, X, cache):
        a1, a2, a3, out = cache
        n = X.shape[0]
        dout = (2.0 / n) * (out - X)              # dL/d(out), L = mean SE
        grads = {}
        grads["W4"] = a3.T @ dout
        grads["b4"] = dout.sum(axis=0)
        da3 = dout @ p["W4"].T
        dz3 = da3 * _dtanh(a3)
        grads["W3"] = a2.T @ dz3
        grads["b3"] = dz3.sum(axis=0)
        da2 = dz3 @ p["W3"].T
        dz2 = da2 * _dtanh(a2)
        grads["W2"] = a1.T @ dz2
        grads["b2"] = dz2.sum(axis=0)
        da1 = dz2 @ p["W2"].T
        dz1 = da1 * _dtanh(a1)
        grads["W1"] = X.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return grads

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        params = self._init_params(X.shape[1])
        opt = _Adam(params, lr=self.lr)
        for _ in range(self.epochs):
            cache = self._forward(params, X)
            grads = self._backward(params, X, cache)
            opt.step(grads)
        *_, out = self._forward(params, X)
        # per-sample squared reconstruction error
        return np.sum((X - out) ** 2, axis=1)
