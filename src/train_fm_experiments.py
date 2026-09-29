"""
train_fm_experiments.py
=======================
Runs the four FM embedding-size experiments (k = 4, 8, 16, 32).

For each k:
  - Trains with the SAME hyperparameters (lr, epochs, batch_size, seed)
  - Sweeps thresholds [0.1 … 0.6] on the validation set and picks best F1
  - Records val loss / accuracy / precision / recall / F1 / param count
  - Saves the model checkpoint

Outputs
-------
  models/fm/fm_k{k}.npz
  results/fm_history_k{k}.csv          ← per-epoch history
  results/fm_experiments.csv           ← one row per k
  results/figures/fm_loss_curves.png
  results/figures/fm_val_f1_vs_k.png
"""

import os, sys, json, csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from any cwd
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from fm import FM, train, bce_loss, f1_score, sigmoid

# ── constants ─────────────────────────────────────────────────────────────────
SEED       = 42
EPOCHS     = 50
BATCH_SIZE = 256
LR         = 0.01
K_VALUES   = [4, 8, 16, 32]
THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.65, 0.05)]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data():
    DATA = os.path.join(ROOT, "data")
    train_df = pd.read_csv(os.path.join(DATA, "split_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA, "split_val.csv"))

    with open(os.path.join(DATA, "drug_mapping.json")) as f:
        mapping = json.load(f)

    d2i     = mapping["drug_to_id"]
    n_drugs = mapping["n_drugs"]

    def encode(df):
        # FM uses separate w1/w2 tables → no offset needed
        i1 = df["Drug 1"].map(d2i).values.astype(np.int32)
        i2 = df["Drug 2"].map(d2i).values.astype(np.int32)
        X  = np.stack([i1, i2], axis=1)
        y  = df["Label"].values.astype(np.float32)
        return X, y

    return encode(train_df), encode(val_df), n_drugs


def best_threshold(model: FM, X_val, y_val) -> tuple[float, float]:
    """Sweep thresholds; return (threshold, val_f1) with highest val F1."""
    p = model.predict_proba(X_val[:, 0], X_val[:, 1])
    best_t, best_f1 = 0.5, 0.0
    for t in THRESHOLDS:
        f1 = f1_score(y_val, (p >= t).astype(np.float32))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def eval_metrics(model: FM, X, y, threshold: float) -> dict:
    """Full metric set at a given threshold."""
    p    = model.predict_proba(X[:, 0], X[:, 1])
    yhat = (p >= threshold).astype(np.float32)
    loss = bce_loss(p, y)

    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())

    acc  = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"loss": loss, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


def param_count(model: FM) -> int:
    return (1                                   # bias
            + model.w1.size + model.w2.size     # linear weights
            + model.v1.size + model.v2.size)    # embeddings


def save_history(history: dict, k: int):
    path = os.path.join(ROOT, "results", f"fm_history_k{k}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_f1"])
        writer.writeheader()
        for ep, (tl, vl, vf) in enumerate(
                zip(history["train_loss"], history["val_loss"],
                    history["val_f1"]), 1):
            writer.writerow({"epoch": ep, "train_loss": tl,
                             "val_loss": vl, "val_f1": vf})


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_loss_curves(histories: dict):
    """One subplot per k: train vs val loss over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (k, hist) in zip(axes, sorted(histories.items())):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], label="Train loss", color="#1f77b4")
        ax.plot(epochs, hist["val_loss"],   label="Val loss",   color="#ff7f0e",
                linestyle="--")
        ax.set_title(f"FM  k={k}", fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("FM — Train vs Validation Loss  (lr=0.01, batch=256)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "fm_loss_curves.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_val_f1_vs_k(results: list[dict]):
    """Bar chart of best validation F1 for each k."""
    ks  = [r["k"] for r in results]
    f1s = [r["val_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([str(k) for k in ks], f1s, color="#2196F3", edgecolor="black",
                  width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
    ax.set_xlabel("Embedding size  k", fontsize=12)
    ax.set_ylabel("Validation F1  (Severe class)", fontsize=12)
    ax.set_title("FM — Validation F1 vs Embedding Dimension  k", fontsize=13)
    ax.set_ylim(0, max(f1s) * 1.25 + 0.02)
    ax.grid(axis="y", alpha=0.3)

    best_k = ks[int(np.argmax(f1s))]
    ax.set_title(
        f"FM — Validation F1 vs k   (best: k={best_k})", fontsize=13)

    plt.tight_layout()
    out = os.path.join(ROOT, "results", "figures", "fm_val_f1_vs_k.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── main experiment loop ──────────────────────────────────────────────────────

def run():
    os.makedirs(os.path.join(ROOT, "models", "fm"),       exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "figures"),  exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"),             exist_ok=True)

    (X_train, y_train), (X_val, y_val), n_drugs = load_data()
    print(f"Train: {len(y_train)}  Val: {len(y_val)}  n_drugs: {n_drugs}\n")

    all_results  = []
    all_histories = {}

    for k in K_VALUES:
        print(f"{'='*55}")
        print(f"  Training FM  k={k}  "
              f"({EPOCHS} epochs, lr={LR}, batch={BATCH_SIZE})")
        print(f"{'='*55}")

        model = FM(n_drugs, k=k, seed=SEED)

        history = train(
            model, X_train, y_train, X_val, y_val,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            seed=SEED, verbose=True,
        )

        # ── pick best threshold on validation ─────────────────────────────────
        threshold, _ = best_threshold(model, X_val, y_val)
        metrics = eval_metrics(model, X_val, y_val, threshold)
        n_params = param_count(model)

        row = {
            "k":           k,
            "threshold":   threshold,
            "val_loss":    round(metrics["loss"],      6),
            "val_accuracy":round(metrics["accuracy"],  6),
            "val_precision":round(metrics["precision"],6),
            "val_recall":  round(metrics["recall"],    6),
            "val_f1":      round(metrics["f1"],        6),
            "n_params":    n_params,
        }
        all_results.append(row)
        all_histories[k] = history

        print(f"\n  threshold={threshold}  "
              f"val_f1={metrics['f1']:.4f}  "
              f"val_acc={metrics['accuracy']:.4f}  "
              f"val_prec={metrics['precision']:.4f}  "
              f"val_rec={metrics['recall']:.4f}")
        print(f"  n_params={n_params}\n")

        # Save checkpoint and per-k history
        ckpt = os.path.join(ROOT, "models", "fm", f"fm_k{k}.npz")
        model.save(ckpt)
        print(f"  Checkpoint → {ckpt}")
        save_history(history, k)

    # ── save results CSV ───────────────────────────────────────────────────────
    csv_path = os.path.join(ROOT, "results", "fm_experiments.csv")
    fields   = ["k", "threshold", "val_loss", "val_accuracy",
                "val_precision", "val_recall", "val_f1", "n_params"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nExperiments saved → {csv_path}")

    # ── print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'k':>4}  {'thresh':>6}  {'val_loss':>9}  {'val_acc':>8}  "
          f"{'prec':>7}  {'rec':>7}  {'F1':>7}  {'params':>8}")
    print("-" * 65)
    best_k = max(all_results, key=lambda r: r["val_f1"])["k"]
    for r in all_results:
        marker = " ◄ BEST" if r["k"] == best_k else ""
        print(f"  {r['k']:>2}  {r['threshold']:>6}  {r['val_loss']:>9.6f}  "
              f"{r['val_accuracy']:>8.4f}  {r['val_precision']:>7.4f}  "
              f"{r['val_recall']:>7.4f}  {r['val_f1']:>7.4f}  "
              f"{r['n_params']:>8}{marker}")
    print("=" * 65)
    print(f"\nBest FM: k={best_k}  (val F1 = "
          f"{next(r['val_f1'] for r in all_results if r['k']==best_k):.4f})")

    # ── plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_loss_curves(all_histories)
    plot_val_f1_vs_k(all_results)
    print("Done.")


if __name__ == "__main__":
    run()
