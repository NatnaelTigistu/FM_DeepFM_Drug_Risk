"""
train_ablation.py
=================
Executes the ablation study using the selected optimal configuration:
- Embedding dimension: k=32
- MLP hidden layers: [64, 32]
- Dropout: 0.0
- SEED: 42

Compares:
1. FM-only (use_fm=True, use_deep=False)
2. Deep-only (use_fm=False, use_deep=True)
3. Full DeepFM (use_fm=True, use_deep=True)

Saves:
- results/ablation.csv
- results/figures/ablation_comparison.png

Does NOT evaluate the test set.
"""

import os
import sys
import json
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from deepfm import (
    DeepFM,
    set_seed,
    build_dataloaders,
    train_deepfm,
)

SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
LR = 0.001
K = 32
HIDDEN = (64, 32)
DROPOUT = 0.0


def run_ablation():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Ablation Study on device: {device} | SEED: {SEED}")

    data_dir = os.path.join(ROOT, "data")
    results_dir = os.path.join(ROOT, "results")
    figures_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "split_train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "split_val.csv"))

    with open(os.path.join(data_dir, "drug_mapping.json")) as f:
        drug_to_id = json.load(f)["drug_to_id"]
    n_drugs = len(drug_to_id)

    train_loader, val_loader = build_dataloaders(
        train_df, val_df, drug_to_id=drug_to_id, batch_size=BATCH_SIZE, seed=SEED
    )

    variants = [
        ("FM-only", True, False),
        ("Deep-only", False, True),
        ("Full DeepFM", True, True),
    ]

    ablation_results = []
    histories = {}

    for name, use_fm, use_deep in variants:
        print(f"\n---> Training {name} (use_fm={use_fm}, use_deep={use_deep})...")
        set_seed(SEED)
        model = DeepFM(
            num_drugs=n_drugs,
            embedding_dim=K,
            hidden_dims=HIDDEN,
            dropout=DROPOUT,
            use_fm=use_fm,
            use_deep=use_deep,
            shared_embeddings=True,
        )

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        res = train_deepfm(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LR,
            device=device,
            seed=SEED,
            verbose=False,
        )

        best_m = res["best_metrics"]
        row = {
            "variant": name,
            "val_loss": round(best_m["loss"], 6),
            "accuracy": round(best_m["accuracy"], 6),
            "precision": round(best_m["precision"], 6),
            "recall": round(best_m["recall"], 6),
            "f1": round(best_m["f1"], 6),
            "threshold": best_m["threshold"],
            "n_params": n_params,
            "best_epoch": best_m["epoch"],
        }
        ablation_results.append(row)
        histories[name] = res["history"]

        print(
            f"  {name:<12} | Val Loss: {best_m['loss']:.4f} | "
            f"Acc: {best_m['accuracy']:.4f} | Prec: {best_m['precision']:.4f} | "
            f"Rec: {best_m['recall']:.4f} | F1: {best_m['f1']:.4f} | Thresh: {best_m['threshold']}"
        )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "ablation.csv")
    fieldnames = [
        "variant",
        "val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "n_params",
        "best_epoch",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ablation_results)
    print(f"\n[OK] Saved ablation results to: {csv_path}")

    # ── Create Comparison Plot ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    names = [r["variant"] for r in ablation_results]
    f1_scores = [r["f1"] for r in ablation_results]
    val_losses = [r["val_loss"] for r in ablation_results]
    colors = ["#4285F4", "#FBBC05", "#34A853"]

    # Subplot 1: Validation F1
    bars1 = ax1.bar(names, f1_scores, color=colors, edgecolor="black", width=0.45)
    ax1.bar_label(bars1, fmt="%.4f", padding=3, fontsize=10, fontweight="bold")
    ax1.set_title("Ablation Study: Validation F1 Score", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Validation F1 (Severe Class)", fontsize=11)
    ax1.set_ylim(0, max(f1_scores) * 1.15)
    ax1.grid(axis="y", alpha=0.3)

    # Subplot 2: Validation Loss
    bars2 = ax2.bar(names, val_losses, color=colors, edgecolor="black", width=0.45)
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=10, fontweight="bold")
    ax2.set_title("Ablation Study: Validation BCE Loss", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Validation BCE Loss (lower is better)", fontsize=11)
    ax2.set_ylim(0, max(val_losses) * 1.2)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"DeepFM Component Ablation Study (k={K}, MLP={HIDDEN}, Dropout={DROPOUT})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    plot_path = os.path.join(figures_dir, "ablation_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved comparison plot to: {plot_path}")


if __name__ == "__main__":
    run_ablation()
