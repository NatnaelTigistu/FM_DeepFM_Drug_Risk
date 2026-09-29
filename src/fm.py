"""
fm.py  —  Factorization Machine from scratch (NumPy only)
==========================================================

Model
-----
For a row with Drug 1 = i and Drug 2 = j:

    z = b + w1[i] + w2[j] + dot(v1[i], v2[j])
    p = sigmoid(z)

Parameters
----------
  b        scalar bias
  w1       (n_drugs,)     first-order weight,  Drug-1 side
  w2       (n_drugs,)     first-order weight,  Drug-2 side
  v1       (n_drugs, k)   embedding matrix,    Drug-1 side
  v2       (n_drugs, k)   embedding matrix,    Drug-2 side

Gradients (derived analytically)
---------------------------------
For a batch of B samples, the mean BCE loss is:
    L = -(1/B) Σ [ y·log(p) + (1-y)·log(1-p) ]

Because d(sigmoid) and d(BCE) cancel cleanly:
    dL/dz = (p - y) / B                   ← per-sample residual, normalised

Then by the chain rule:
    dL/db     = Σ dL/dz
    dL/dw1[i] = Σ dL/dz   (sum over samples where Drug-1 = i)
    dL/dw2[j] = Σ dL/dz   (sum over samples where Drug-2 = j)
    dL/dv1[i] = Σ dL/dz · v2[j]
    dL/dv2[j] = Σ dL/dz · v1[i]
"""

import numpy as np


SEED = 42


# ── FM class ──────────────────────────────────────────────────────────────────

class FM:
    """Factorization Machine with separate Drug-1 / Drug-2 parameters."""

    def __init__(self, n_drugs: int, k: int, seed: int = SEED):
        """
        n_drugs : number of unique drugs (same vocabulary for both sides)
        k       : embedding / latent-factor dimension
        """
        rng = np.random.default_rng(seed)

        self.n_drugs = n_drugs
        self.k = k

        # Glorot-style scale so dot products start near zero but are not all zero
        scale = 1.0 / np.sqrt(k)

        self.b  = 0.0                                           # scalar
        self.w1 = np.zeros(n_drugs, dtype=np.float64)          # (n_drugs,)
        self.w2 = np.zeros(n_drugs, dtype=np.float64)          # (n_drugs,)
        self.v1 = rng.uniform(-scale, scale, (n_drugs, k))     # (n_drugs, k)
        self.v2 = rng.uniform(-scale, scale, (n_drugs, k))     # (n_drugs, k)

    # ── forward pass ──────────────────────────────────────────────────────────

    def forward(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        """
        idx1, idx2 : integer arrays of shape (B,)
                     Drug-1 and Drug-2 indices for each sample in the batch.
        Returns    : logits z, shape (B,)
        """
        linear   = self.b + self.w1[idx1] + self.w2[idx2]          # (B,)
        dot_prod = np.sum(self.v1[idx1] * self.v2[idx2], axis=1)   # (B,)
        return linear + dot_prod

    def predict_proba(self, idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
        return sigmoid(self.forward(idx1, idx2))

    # ── loss ──────────────────────────────────────────────────────────────────

    def loss(self, idx1: np.ndarray, idx2: np.ndarray,
             y: np.ndarray) -> float:
        p = self.predict_proba(idx1, idx2)
        return bce_loss(p, y)

    # ── analytical gradients ──────────────────────────────────────────────────

    def gradients(
        self,
        idx1: np.ndarray,
        idx2: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """
        Compute analytical gradients of mean BCE loss w.r.t. all parameters.

        Returns a dict with keys: 'b', 'w1', 'w2', 'v1', 'v2'
        Each value has the same shape as the corresponding parameter.
        """
        B  = len(y)
        p  = self.predict_proba(idx1, idx2)    # (B,)
        r  = (p - y) / B                       # per-sample normalised residual (B,)

        # gradient of scalar bias
        grad_b = float(r.sum())

        # first-order weights  —  sparse accumulation
        grad_w1 = np.zeros_like(self.w1)
        grad_w2 = np.zeros_like(self.w2)
        np.add.at(grad_w1, idx1, r)
        np.add.at(grad_w2, idx2, r)

        # embedding matrices  —  sparse accumulation
        grad_v1 = np.zeros_like(self.v1)
        grad_v2 = np.zeros_like(self.v2)
        # r[:, None] broadcasts (B,) → (B, k)
        np.add.at(grad_v1, idx1, r[:, None] * self.v2[idx2])
        np.add.at(grad_v2, idx2, r[:, None] * self.v1[idx1])

        return {"b": grad_b, "w1": grad_w1, "w2": grad_w2,
                "v1": grad_v1, "v2": grad_v2}

    # ── parameter update (SGD step) ───────────────────────────────────────────

    def step(self, grads: dict, lr: float) -> None:
        """Apply one gradient-descent update."""
        self.b  -= lr * grads["b"]
        self.w1 -= lr * grads["w1"]
        self.w2 -= lr * grads["w2"]
        self.v1 -= lr * grads["v1"]
        self.v2 -= lr * grads["v2"]

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        np.savez(path, b=self.b, w1=self.w1, w2=self.w2,
                 v1=self.v1, v2=self.v2, n_drugs=self.n_drugs, k=self.k)

    @classmethod
    def load(cls, path: str) -> "FM":
        d = np.load(path)
        model = cls(int(d["n_drugs"]), int(d["k"]))
        model.b  = float(d["b"])
        model.w1 = d["w1"]
        model.w2 = d["w2"]
        model.v1 = d["v1"]
        model.v2 = d["v2"]
        return model


# ── standalone functions ──────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable: use different branches for positive / negative z
    out = np.where(z >= 0,
                   1.0 / (1.0 + np.exp(-z)),
                   np.exp(z) / (1.0 + np.exp(z)))
    return out


def bce_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy.  eps clips log(0)."""
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ── numerical gradient check ──────────────────────────────────────────────────

def numerical_gradient_check(
    model: FM,
    idx1: np.ndarray,
    idx2: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5,
    n_samples: int = 8,
    seed: int = SEED,
) -> None:
    """
    Compare analytical gradients against finite-difference estimates for a
    random subset of parameters.  Passes if max relative error < 1e-4.
    """
    rng = np.random.default_rng(seed)
    analytic = model.gradients(idx1, idx2, y)

    def loss_fn():
        return model.loss(idx1, idx2, y)

    checks = []

    # ── helper: check one in-place ndarray parameter ──────────────────────────
    def check_param(name: str, param: np.ndarray, grad: np.ndarray):
        flat_idx = rng.choice(param.size, size=min(n_samples, param.size),
                               replace=False)
        for fi in flat_idx:
            orig = param.flat[fi]
            param.flat[fi] = orig + eps
            lp = loss_fn()
            param.flat[fi] = orig - eps
            lm = loss_fn()
            param.flat[fi] = orig           # restore
            num = (lp - lm) / (2 * eps)
            ana = grad.flat[fi]
            denom = max(abs(num), abs(ana), 1e-8)
            rel_err = abs(num - ana) / denom
            checks.append((name, fi, ana, num, rel_err))

    # Bias is a Python float — mutate model.b directly
    orig_b = model.b
    model.b = orig_b + eps;  lp = loss_fn()
    model.b = orig_b - eps;  lm = loss_fn()
    model.b = orig_b
    num_b   = (lp - lm) / (2 * eps)
    ana_b   = analytic["b"]
    denom_b = max(abs(num_b), abs(ana_b), 1e-8)
    checks.append(("b", 0, ana_b, num_b, abs(num_b - ana_b) / denom_b))

    check_param("w1", model.w1, analytic["w1"])
    check_param("w2", model.w2, analytic["w2"])
    check_param("v1", model.v1, analytic["v1"])
    check_param("v2", model.v2, analytic["v2"])

    max_err = max(c[4] for c in checks)
    print(f"\nNumerical gradient check  ({len(checks)} entries sampled)")
    print(f"  Max relative error : {max_err:.2e}")
    for name, fi, ana, num, err in checks:
        flag = "  <--  FAIL" if err > 1e-4 else ""
        print(f"  {name}[{fi}]  analytic={ana:+.6f}  numerical={num:+.6f}  "
              f"rel_err={err:.2e}{flag}")

    if max_err < 1e-4:
        print("\n[PASS] All gradients match numerically.")
    else:
        print("\n[FAIL] Gradient mismatch detected.")


# ── toy example ───────────────────────────────────────────────────────────────

def run_toy_example() -> None:
    """
    Walk through one forward pass with hand-checkable numbers.

    Setup: 3 drugs, k=2.  Manually set every parameter so the arithmetic
    is transparent.
    """
    print("=" * 60)
    print("TOY EXAMPLE  —  one forward pass, step by step")
    print("=" * 60)

    n_drugs, k = 3, 2
    model = FM(n_drugs, k)

    # Set parameters to simple numbers
    model.b     = 0.1
    model.w1[:] = [0.2, -0.1,  0.3]    # weights for Drug-1 positions 0,1,2
    model.w2[:] = [-0.1, 0.4, 0.05]   # weights for Drug-2 positions 0,1,2
    model.v1[:] = [[0.5, -0.3],        # embedding of drug-0 in Drug-1 role
                   [0.2,  0.6],
                   [-0.4, 0.1]]
    model.v2[:] = [[0.3,  0.7],        # embedding of drug-0 in Drug-2 role
                   [-0.2, 0.5],
                   [0.1, -0.6]]

    # Sample: Drug-1 = drug 0, Drug-2 = drug 1, true label = 1 (Severe)
    i, j, y_true = 0, 1, 1.0

    idx1 = np.array([i])
    idx2 = np.array([j])
    y    = np.array([y_true])

    # Forward, broken into parts
    bias   = model.b
    lin1   = model.w1[i]
    lin2   = model.w2[j]
    dot    = float(np.dot(model.v1[i], model.v2[j]))
    z      = bias + lin1 + lin2 + dot
    p      = float(sigmoid(np.array([z]))[0])
    loss   = bce_loss(np.array([p]), y)

    print(f"\n  Drug-1 index  i = {i}    Drug-2 index  j = {j}    True label y = {y_true}")
    print(f"\n  bias          b        = {bias:+.4f}")
    print(f"  linear term   w1[{i}]   = {lin1:+.4f}")
    print(f"  linear term   w2[{j}]   = {lin2:+.4f}")
    print(f"  v1[{i}]                 = {model.v1[i]}")
    print(f"  v2[{j}]                 = {model.v2[j]}")
    print(f"  dot(v1[{i}], v2[{j}])   = {dot:+.4f}")
    print(f"\n  logit   z  = {bias:+.4f} + {lin1:+.4f} + {lin2:+.4f} + {dot:+.4f}")
    print(f"             = {z:+.4f}")
    print(f"\n  probability   p = sigmoid({z:.4f}) = {p:.6f}")
    print(f"  BCE loss        = {loss:.6f}")

    # Gradients
    grads = model.gradients(idx1, idx2, y)
    r = p - y_true     # residual (single sample, B=1 so /1)
    print(f"\n  Residual  (p - y) = {r:+.6f}")
    print(f"\n  Analytical gradients")
    print(f"    dL/db      = {grads['b']:+.6f}")
    print(f"    dL/dw1[{i}]  = {grads['w1'][i]:+.6f}")
    print(f"    dL/dw2[{j}]  = {grads['w2'][j]:+.6f}")
    print(f"    dL/dv1[{i}]  = {grads['v1'][i]}")
    print(f"    dL/dv2[{j}]  = {grads['v2'][j]}")

    print("\n" + "=" * 60)
    print("NUMERICAL GRADIENT CHECK on toy model")
    print("=" * 60)
    numerical_gradient_check(model, idx1, idx2, y)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_toy_example()
