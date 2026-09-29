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


# ── inference ─────────────────────────────────────────────────────────────────

def predict_pair(
    model: "FM",
    drug1: str,
    drug2: str,
    drug_to_id: dict,
    threshold: float = 0.5,
) -> dict:
    """
    Given two drug names, return the model's prediction.

    Parameters
    ----------
    model      : a loaded FM instance (use FM.load to avoid retraining)
    drug1      : name of Drug 1  (must be in training vocabulary)
    drug2      : name of Drug 2
    drug_to_id : dict mapping drug name → integer ID  (from drug_mapping.json)
    threshold  : classification threshold (default 0.5; use val-tuned value)

    Returns
    -------
    dict with keys: drug1, drug2, logit, probability, predicted_class
    """
    if drug1 not in drug_to_id:
        raise KeyError(f"Drug '{drug1}' not in training vocabulary.")
    if drug2 not in drug_to_id:
        raise KeyError(f"Drug '{drug2}' not in training vocabulary.")

    i = drug_to_id[drug1]
    j = drug_to_id[drug2]

    idx1 = np.array([i], dtype=np.int32)
    idx2 = np.array([j], dtype=np.int32)

    logit = float(model.forward(idx1, idx2)[0])
    prob  = float(sigmoid(np.array([logit]))[0])

    return {
        "drug1":           drug1,
        "drug2":           drug2,
        "logit":           logit,
        "probability":     prob,
        "predicted_class": int(prob >= threshold),
    }


def explain_pair(
    model: "FM",
    drug1: str,
    drug2: str,
    drug_to_id: dict,
    threshold: float = 0.5,
) -> dict:
    """
    Break the FM score into its four additive components for one drug pair.

    z = b  +  w1[i]  +  w2[j]  +  dot(v1[i], v2[j])
    p = sigmoid(z)

    Returns a dict with every component plus the final logit and probability.
    """
    if drug1 not in drug_to_id:
        raise KeyError(f"Drug '{drug1}' not in training vocabulary.")
    if drug2 not in drug_to_id:
        raise KeyError(f"Drug '{drug2}' not in training vocabulary.")

    i = drug_to_id[drug1]
    j = drug_to_id[drug2]

    bias    = model.b
    w1_term = float(model.w1[i])
    w2_term = float(model.w2[j])
    dot     = float(np.dot(model.v1[i], model.v2[j]))
    logit   = bias + w1_term + w2_term + dot
    prob    = float(sigmoid(np.array([logit]))[0])

    return {
        "drug1":           drug1,
        "drug2":           drug2,
        "bias":            bias,
        "w1_term":         w1_term,
        "w2_term":         w2_term,
        "dot_product":     dot,
        "logit":           logit,
        "probability":     prob,
        "predicted_class": int(prob >= threshold),
    }


def print_explanation(explanation: dict, threshold: float = 0.5) -> None:
    """Pretty-print an explanation dict from explain_pair."""
    e = explanation
    print("=" * 52)
    print(f"  Drug 1 : {e['drug1']}")
    print(f"  Drug 2 : {e['drug2']}")
    print("=" * 52)
    print(f"  bias            b         = {e['bias']:+.6f}")
    print(f"  linear weight   w1[i]     = {e['w1_term']:+.6f}")
    print(f"  linear weight   w2[j]     = {e['w2_term']:+.6f}")
    print(f"  interaction     dot(v,v') = {e['dot_product']:+.6f}")
    print(f"  {'─'*44}")
    print(f"  logit           z         = {e['logit']:+.6f}")
    print(f"  probability     p=σ(z)    = {e['probability']:.6f}")
    print(f"  threshold                 = {threshold}")
    print(f"  prediction                = "
          f"{'Severe (1)' if e['predicted_class'] else 'Not Severe (0)'}")
    print("=" * 52)


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


# ── mini-batch training ───────────────────────────────────────────────────────

def train(
    model: FM,
    X_train: np.ndarray,        # shape (N, 2): [drug1_id, drug2_id]
    y_train: np.ndarray,        # shape (N,)
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 0.01,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """
    Mini-batch gradient descent.

    Each epoch:
      1. Shuffle the training set (reproducible with seed).
      2. Loop over mini-batches, compute gradients, update parameters.
      3. Compute full-dataset train loss, val loss, val F1.

    Returns history dict: {train_loss, val_loss, val_f1} — one value per epoch.
    """
    rng = np.random.default_rng(seed)
    N   = len(y_train)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # Shuffle training rows
        perm   = rng.permutation(N)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]

        # Mini-batch loop
        for start in range(0, N, batch_size):
            xb    = X_shuf[start : start + batch_size]
            yb    = y_shuf[start : start + batch_size]
            grads = model.gradients(xb[:, 0], xb[:, 1], yb)
            model.step(grads, lr)

        # ── epoch-end metrics ──────────────────────────────────────────────
        train_loss = model.loss(X_train[:, 0], X_train[:, 1], y_train)

        p_val    = model.predict_proba(X_val[:, 0], X_val[:, 1])
        val_loss = bce_loss(p_val, y_val)
        val_f1   = f1_score(y_val, (p_val >= 0.5).astype(np.float32))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if verbose:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  "
                  f"val_f1={val_f1:.4f}")

    return history


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 for the positive (Severe = 1) class."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


# ── mini-batch walkthrough ────────────────────────────────────────────────────

def show_minibatch_walkthrough(
    model: FM,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 4,
    lr: float = 0.01,
    seed: int = SEED,
) -> None:
    """
    Grab a tiny batch and print every step:
    forward pass → loss → gradients → parameter update.
    """
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(y), size=batch_size, replace=False)
    xb    = X[idx]
    yb    = y[idx]
    idx1b = xb[:, 0]
    idx2b = xb[:, 1]

    print("\n" + "=" * 60)
    print(f"MINI-BATCH WALKTHROUGH  (B = {batch_size}, lr = {lr})")
    print("=" * 60)

    # ── forward ───────────────────────────────────────────────────────────────
    z    = model.forward(idx1b, idx2b)
    p    = sigmoid(z)
    loss = bce_loss(p, yb)

    print(f"\n{'s':>3}  {'i':>5}  {'j':>5}  {'y':>3}  "
          f"{'z':>8}  {'p':>8}")
    print("-" * 40)
    for s in range(batch_size):
        print(f"  {s}  {idx1b[s]:>5}  {idx2b[s]:>5}  {yb[s]:>3.0f}  "
              f"{z[s]:>8.4f}  {p[s]:>8.4f}")
    print(f"\n  BCE loss = {loss:.6f}")

    # ── gradients ─────────────────────────────────────────────────────────────
    grads = model.gradients(idx1b, idx2b, yb)
    r     = (p - yb) / batch_size      # normalised residuals

    print(f"\n  Residuals (p - y) / B :")
    for s in range(batch_size):
        print(f"    s={s}  r={r[s]:+.6f}")

    print(f"\n  grad_b = {grads['b']:+.8f}")
    for s in range(batch_size):
        i, j = int(idx1b[s]), int(idx2b[s])
        print(f"  grad_w1[{i}] = {grads['w1'][i]:+.8f}   "
              f"grad_w2[{j}] = {grads['w2'][j]:+.8f}")

    # ── update ────────────────────────────────────────────────────────────────
    b_old  = model.b
    i0, j0 = int(idx1b[0]), int(idx2b[0])
    w1_old = model.w1[i0]
    v1_old = model.v1[i0].copy()

    model.step(grads, lr)

    print(f"\n  After update  (lr = {lr})")
    print(f"    b      : {b_old:+.6f}  →  {model.b:+.6f}  "
          f"Δ={model.b - b_old:+.2e}")
    print(f"    w1[{i0}] : {w1_old:+.6f}  →  {model.w1[i0]:+.6f}  "
          f"Δ={model.w1[i0] - w1_old:+.2e}")
    print(f"    v1[{i0}] before : {v1_old}")
    print(f"    v1[{i0}] after  : {model.v1[i0]}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, json, csv
    import pandas as pd

    # ── 1. Toy example + gradient check (toy model) ───────────────────────────
    run_toy_example()

    # ── 2. Load real splits ───────────────────────────────────────────────────
    ROOT = os.path.join(os.path.dirname(__file__), "..")
    DATA = os.path.join(ROOT, "data")

    print("\n\nLoading real splits ...")
    train_df = pd.read_csv(os.path.join(DATA, "split_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA, "split_val.csv"))

    with open(os.path.join(DATA, "drug_mapping.json")) as f:
        mapping = json.load(f)

    d2i    = mapping["drug_to_id"]
    offset = mapping["drug2_offset"]
    n_drugs = mapping["n_drugs"]

    def encode(df):
        # FM uses SEPARATE w1/w2 and v1/v2 tables, each indexed 0..n_drugs-1.
        # No offset needed here — the offset is only for DeepFM's shared table.
        i1 = df["Drug 1"].map(d2i).values.astype(np.int32)
        i2 = df["Drug 2"].map(d2i).values.astype(np.int32)
        X  = np.stack([i1, i2], axis=1)
        y  = df["Label"].values.astype(np.float32)
        return X, y

    X_train, y_train = encode(train_df)
    X_val,   y_val   = encode(val_df)
    print(f"  Train rows: {len(y_train)}   Val rows: {len(y_val)}")

    # ── 3. Gradient check on a real mini-batch ────────────────────────────────
    print("\n\nGradient check on real data  (k=4, B=32)")
    print("-" * 60)
    rng_c = np.random.default_rng(SEED)
    idx_c = rng_c.choice(len(y_train), 32, replace=False)
    model_check = FM(n_drugs, k=4)
    numerical_gradient_check(
        model_check,
        X_train[idx_c, 0], X_train[idx_c, 1], y_train[idx_c],
        n_samples=6,
    )

    # ── 4. Mini-batch walkthrough (k=8, fresh model) ─────────────────────────
    model_walk = FM(n_drugs, k=8)
    show_minibatch_walkthrough(model_walk, X_train, y_train,
                               batch_size=4, lr=0.01)

    # ── 5. Real training run (k=8, 20 epochs) ─────────────────────────────────
    print("\n\nTraining FM  (k=8, lr=0.01, batch_size=256, 20 epochs)")
    print("-" * 60)
    model = FM(n_drugs, k=8)
    history = train(model, X_train, y_train, X_val, y_val,
                    epochs=20, batch_size=256, lr=0.01)

    # ── 6. Save checkpoint and history ────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "models", "fm"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    ckpt_path = os.path.join(ROOT, "models", "fm", "fm_k8.npz")
    model.save(ckpt_path)

    hist_path = os.path.join(ROOT, "results", "fm_history_k8.csv")
    with open(hist_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1"])
        writer.writeheader()
        for ep, (tl, vl, vf) in enumerate(
                zip(history["train_loss"], history["val_loss"],
                    history["val_f1"]), 1):
            writer.writerow({"epoch": ep, "train_loss": tl,
                             "val_loss": vl, "val_f1": vf})

    best_ep  = int(np.argmax(history["val_f1"])) + 1
    best_f1  = max(history["val_f1"])
    print(f"\nCheckpoint → {ckpt_path}")
    print(f"History    → {hist_path}")
    print(f"Best val F1 = {best_f1:.4f}  at epoch {best_ep}")
