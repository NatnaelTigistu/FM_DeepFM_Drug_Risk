"""
train_deepfm_experiments.py
===========================
Executes the controlled DeepFM benchmark as specified in the assignment:

Benchmark Stages:
1. Embedding dimension: k in [4, 8, 16, 32] (MLP=[32], dropout=0.0)
2. MLP Architecture: [32], [64], [64, 32] (using best k, dropout=0.0)
3. Dropout rate: 0.0, 0.2, 0.5 (using best k and best MLP)

Tracks:
- training loss
- validation loss
- validation accuracy
- validation precision
- validation recall
- validation F1
- model parameter count

Saves:
- Best checkpoint to models/deepfm/deepfm_best.pt
- Experiment results to results/deepfm_experiments.csv
- Figures to results/figures/deepfm_loss_curves.png and results/figures/deepfm_benchmark_summary.png

Uses:
- Exact same data splits (split_train.csv, split_val.csv)
- Exact same drug mapping (drug_mapping.json)
- SEED = 42
- BCEWithLogitsLoss
- Does NOT evaluate the test set.
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

from deepfm import (
    DeepFM,
    TORCH_AVAILABLE,
    set_seed,
    build_dataloaders,
    train_deepfm,
    evaluate_metrics,
)

SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
LR = 0.001


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_benchmarks():
    if not TORCH_AVAILABLE:
        print("\n" + "=" * 65)
        print("NOTICE: PyTorch is not yet installed in this environment.")
        print("To run DeepFM benchmarks, install PyTorch or execute this script")
        print("in Google Colab via notebooks/DeepFM_template.ipynb.")
        print("=" * 65 + "\n")
        return

    import torch

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device} | SEED: {SEED}")

    data_dir = os.path.join(ROOT, "data")
    models_dir = os.path.join(ROOT, "models", "deepfm")
    results_dir = os.path.join(ROOT, "results")
    figures_dir = os.path.join(ROOT, "results", "figures")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ── 1. Load exact same data and mapping ───────────────────────────────────
    train_df = pd.read_csv(os.path.join(data_dir, "split_train.csv"))
    val_df   = pd.read_csv(os.path.join(data_dir, "split_val.csv"))

    with open(os.path.join(data_dir, "drug_mapping.json")) as f:
        mapping = json.load(f)
    drug_to_id = mapping["drug_to_id"]
    n_drugs = mapping["n_drugs"]

    print(f"Loaded train ({len(train_df)} rows) and val ({len(val_df)} rows). Drugs: {n_drugs}")

    train_loader, val_loader = build_dataloaders(
        train_df, val_df, drug_to_id=drug_to_id, batch_size=BATCH_SIZE, seed=SEED
    )

    all_results = []
    histories = {}
    best_overall_val_f1 = -1.0
    best_overall_state = None
    best_overall_info = None

    # ── STAGE 1: Embedding Dimension (k in [4, 8, 16, 32]) ───────────────────
    print("\n" + "=" * 65)
    print("STAGE 1: Embedding Dimension Benchmark (k in [4, 8, 16, 32])")
    print("=" * 65)

    stage1_k_values = [4, 8, 16, 32]
    stage1_results = {}

    for k in stage1_k_values:
        print(f"\n---> Training DeepFM (k={k}, MLP=[32], dropout=0.0)...")
        set_seed(SEED)
        model = DeepFM(
            num_drugs=n_drugs,
            embedding_dim=k,
            hidden_dims=(32,),
            dropout=0.0,
            use_fm=True,
            use_deep=True,
            shared_embeddings=True,
        )

        n_params = count_parameters(model)
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
            "stage": "1_embedding_dim",
            "k": k,
            "hidden_layers": "(32,)",
            "dropout": 0.0,
            "threshold": best_m["threshold"],
            "train_loss": round(res["history"]["train_loss"][best_m["epoch"] - 1], 6),
            "val_loss": round(best_m["loss"], 6),
            "val_accuracy": round(best_m["accuracy"], 6),
            "val_precision": round(best_m["precision"], 6),
            "val_recall": round(best_m["recall"], 6),
            "val_f1": round(best_m["f1"], 6),
            "n_params": n_params,
            "best_epoch": best_m["epoch"],
        }

        all_results.append(row)
        stage1_results[k] = row
        histories[f"k={k}"] = res["history"]

        print(
            f"  Result k={k:>2} | Best Epoch: {best_m['epoch']:>2} | "
            f"Val Loss: {best_m['loss']:.4f} | Val F1: {best_m['f1']:.4f} | "
            f"Acc: {best_m['accuracy']:.4f} | Rec: {best_m['recall']:.4f} | Params: {n_params}"
        )

        if best_m["f1"] > best_overall_val_f1:
            best_overall_val_f1 = best_m["f1"]
            best_overall_state = {k_param: v.cpu().clone() for k_param, v in res["model"].state_dict().items()}
            best_overall_info = row

    best_k = max(stage1_results.keys(), key=lambda x: stage1_results[x]["val_f1"])
    print(f"\n>> Best embedding size from Stage 1: k={best_k} (Val F1: {stage1_results[best_k]['val_f1']:.4f})")

    # ── STAGE 2: MLP Architecture ([32], [64], [64, 32]) ─────────────────────
    print("\n" + "=" * 65)
    print(f"STAGE 2: MLP Architecture Benchmark with best k={best_k}")
    print("=" * 65)

    mlp_configs = [(32,), (64,), (64, 32)]
    stage2_results = {}

    for hidden in mlp_configs:
        hidden_str = str(hidden)
        if hidden == (32,):
            # Reuse from stage 1
            row = stage1_results[best_k].copy()
            row["stage"] = "2_mlp_architecture"
            all_results.append(row)
            stage2_results[hidden_str] = row
            print(f"  MLP={hidden_str:<10} | (Reused from Stage 1) Val F1: {row['val_f1']:.4f}")
            continue

        print(f"\n---> Training DeepFM (k={best_k}, MLP={hidden_str}, dropout=0.0)...")
        set_seed(SEED)
        model = DeepFM(
            num_drugs=n_drugs,
            embedding_dim=best_k,
            hidden_dims=hidden,
            dropout=0.0,
            use_fm=True,
            use_deep=True,
            shared_embeddings=True,
        )

        n_params = count_parameters(model)
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
            "stage": "2_mlp_architecture",
            "k": best_k,
            "hidden_layers": hidden_str,
            "dropout": 0.0,
            "threshold": best_m["threshold"],
            "train_loss": round(res["history"]["train_loss"][best_m["epoch"] - 1], 6),
            "val_loss": round(best_m["loss"], 6),
            "val_accuracy": round(best_m["accuracy"], 6),
            "val_precision": round(best_m["precision"], 6),
            "val_recall": round(best_m["recall"], 6),
            "val_f1": round(best_m["f1"], 6),
            "n_params": n_params,
            "best_epoch": best_m["epoch"],
        }

        all_results.append(row)
        stage2_results[hidden_str] = row
        histories[f"mlp={hidden_str}"] = res["history"]

        print(
            f"  Result MLP={hidden_str:<10} | Best Epoch: {best_m['epoch']:>2} | "
            f"Val Loss: {best_m['loss']:.4f} | Val F1: {best_m['f1']:.4f} | "
            f"Acc: {best_m['accuracy']:.4f} | Rec: {best_m['recall']:.4f} | Params: {n_params}"
        )

        if best_m["f1"] > best_overall_val_f1:
            best_overall_val_f1 = best_m["f1"]
            best_overall_state = {k_param: v.cpu().clone() for k_param, v in res["model"].state_dict().items()}
            best_overall_info = row

    best_hidden_str = max(stage2_results.keys(), key=lambda x: stage2_results[x]["val_f1"])
    best_hidden = eval(best_hidden_str)
    print(f"\n>> Best MLP architecture from Stage 2: {best_hidden_str} (Val F1: {stage2_results[best_hidden_str]['val_f1']:.4f})")

    # ── STAGE 3: Dropout Rate (0.0, 0.2, 0.5) ─────────────────────────────────
    print("\n" + "=" * 65)
    print(f"STAGE 3: Dropout Benchmark with k={best_k}, MLP={best_hidden_str}")
    print("=" * 65)

    dropout_values = [0.0, 0.2, 0.5]
    stage3_results = {}

    for drop in dropout_values:
        if drop == 0.0:
            # Reuse from stage 2
            row = stage2_results[best_hidden_str].copy()
            row["stage"] = "3_dropout"
            all_results.append(row)
            stage3_results[drop] = row
            print(f"  Dropout={drop:<4} | (Reused from Stage 2) Val F1: {row['val_f1']:.4f}")
            continue

        print(f"\n---> Training DeepFM (k={best_k}, MLP={best_hidden_str}, dropout={drop})...")
        set_seed(SEED)
        model = DeepFM(
            num_drugs=n_drugs,
            embedding_dim=best_k,
            hidden_dims=best_hidden,
            dropout=drop,
            use_fm=True,
            use_deep=True,
            shared_embeddings=True,
        )

        n_params = count_parameters(model)
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
            "stage": "3_dropout",
            "k": best_k,
            "hidden_layers": best_hidden_str,
            "dropout": drop,
            "threshold": best_m["threshold"],
            "train_loss": round(res["history"]["train_loss"][best_m["epoch"] - 1], 6),
            "val_loss": round(best_m["loss"], 6),
            "val_accuracy": round(best_m["accuracy"], 6),
            "val_precision": round(best_m["precision"], 6),
            "val_recall": round(best_m["recall"], 6),
            "val_f1": round(best_m["f1"], 6),
            "n_params": n_params,
            "best_epoch": best_m["epoch"],
        }

        all_results.append(row)
        stage3_results[drop] = row
        histories[f"dropout={drop}"] = res["history"]

        print(
            f"  Result Dropout={drop:<4} | Best Epoch: {best_m['epoch']:>2} | "
            f"Val Loss: {best_m['loss']:.4f} | Val F1: {best_m['f1']:.4f} | "
            f"Acc: {best_m['accuracy']:.4f} | Rec: {best_m['recall']:.4f} | Params: {n_params}"
        )

        if best_m["f1"] > best_overall_val_f1:
            best_overall_val_f1 = best_m["f1"]
            best_overall_state = {k_param: v.cpu().clone() for k_param, v in res["model"].state_dict().items()}
            best_overall_info = row

    best_dropout = max(stage3_results.keys(), key=lambda x: stage3_results[x]["val_f1"])
    print(f"\n>> Best Dropout from Stage 3: {best_dropout} (Val F1: {stage3_results[best_dropout]['val_f1']:.4f})")

    # ── Save best checkpoint ──────────────────────────────────────────────────
    best_ckpt_path = os.path.join(models_dir, "deepfm_best.pt")
    torch.save(
        {
            "model_state_dict": best_overall_state,
            "config": best_overall_info,
            "best_val_f1": best_overall_val_f1,
            "n_drugs": n_drugs,
        },
        best_ckpt_path,
    )
    print(f"\n[OK] Saved best model checkpoint to: {best_ckpt_path}")
    print(f"     Best configuration: k={best_overall_info['k']}, MLP={best_overall_info['hidden_layers']}, "
          f"dropout={best_overall_info['dropout']} | Best Val F1: {best_overall_val_f1:.4f}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "deepfm_experiments.csv")
    fieldnames = [
        "stage",
        "k",
        "hidden_layers",
        "dropout",
        "threshold",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "n_params",
        "best_epoch",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[OK] Saved experiment benchmark results to: {csv_path}")

    # ── Generate Plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Plot 1: Loss curves for Stage 1 (k in [4, 8, 16, 32])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, k in enumerate(stage1_k_values):
        key = f"k={k}"
        h = histories[key]
        ep_range = range(1, len(h["train_loss"]) + 1)
        ax = axes[idx]
        ax.plot(ep_range, h["train_loss"], label="Train Loss", color="#1f77b4")
        ax.plot(ep_range, h["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--")
        ax.set_title(f"DeepFM (k={k}, MLP=[32])")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("DeepFM Training vs Validation Loss Across Embedding Sizes", fontsize=14)
    plt.tight_layout()
    loss_plot_path = os.path.join(figures_dir, "deepfm_loss_curves.png")
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    print(f"[OK] Saved: {loss_plot_path}")

    # Plot 2: Benchmark Summary Comparison (3 subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 1: Stage 1 - Embedding Dim
    k_labels = [str(k) for k in stage1_k_values]
    k_f1s = [stage1_results[k]["val_f1"] for k in stage1_k_values]
    bars1 = ax1.bar(k_labels, k_f1s, color="#4285F4", edgecolor="black")
    ax1.bar_label(bars1, fmt="%.4f", padding=3, fontsize=9)
    ax1.set_title("1. Embedding Dimension (k)")
    ax1.set_xlabel("Dimension k")
    ax1.set_ylabel("Validation F1")
    ax1.grid(axis="y", alpha=0.3)

    # Subplot 2: Stage 2 - MLP Architecture
    mlp_labels = [str(h) for h in mlp_configs]
    mlp_f1s = [stage2_results[str(h)]["val_f1"] for h in mlp_configs]
    bars2 = ax2.bar(mlp_labels, mlp_f1s, color="#34A853", edgecolor="black")
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=9)
    ax2.set_title(f"2. MLP Architecture (k={best_k})")
    ax2.set_xlabel("Hidden Layers")
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Stage 3 - Dropout
    drop_labels = [str(d) for d in dropout_values]
    drop_f1s = [stage3_results[d]["val_f1"] for d in dropout_values]
    bars3 = ax3.bar(drop_labels, drop_f1s, color="#FBBC05", edgecolor="black")
    ax3.bar_label(bars3, fmt="%.4f", padding=3, fontsize=9)
    ax3.set_title(f"3. Dropout Rate (MLP={best_hidden_str})")
    ax3.set_xlabel("Dropout")
    ax3.grid(axis="y", alpha=0.3)

    plt.suptitle("DeepFM Controlled Benchmark Summary (Validation F1)", fontsize=14, y=1.02)
    plt.tight_layout()
    summary_plot_path = os.path.join(figures_dir, "deepfm_benchmark_summary.png")
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {summary_plot_path}")

    print("\nBenchmark complete! Test set was NOT touched.")


if __name__ == "__main__":
    run_benchmarks()
