"""
evaluate_test.py
================
Final held-out test set evaluation for FM and DeepFM.

Rules applied:
1. Both models were selected strictly based on validation F1.
2. The checkpoints (models/fm/fm_k8.npz and models/deepfm/deepfm_best.pt) are loaded
   directly from disk without any retraining.
3. Thresholds (FM: 0.15, DeepFM: 0.50) were tuned on the validation set only.
4. Test set (data/split_test.csv) is evaluated exactly once.
5. Calculates: accuracy, precision, recall, F1, confusion matrix (TP, TN, FP, FN).
6. Generates:
   - results/final_comparison.csv
   - results/figures/test_confusion_matrices.png
   - results/figures/test_metrics_comparison.png
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
from fm import FM, bce_loss as fm_bce_loss
from deepfm import DeepFM

SEED = 42
DATA_DIR = os.path.join(ROOT, "data")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "results", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def evaluate_test():
    print("=" * 65)
    print("FINAL HELD-OUT TEST EVALUATION (FM vs DeepFM)")
    print("=" * 65)

    # ── 1. Load test data and mapping ─────────────────────────────────────────
    test_csv = os.path.join(DATA_DIR, "split_test.csv")
    mapping_json = os.path.join(DATA_DIR, "drug_mapping.json")

    test_df = pd.read_csv(test_csv)
    with open(mapping_json) as f:
        mapping = json.load(f)
    drug_to_id = mapping["drug_to_id"]
    n_drugs = mapping["n_drugs"]

    y_test = test_df["Label"].values.astype(np.float32)
    n_test = len(y_test)
    n_severe = int((y_test == 1).sum())

    print(f"Test Set: {n_test} samples | Severe (y=1): {n_severe} ({100 * n_severe / n_test:.2f}%)")

    # Encode test pairs
    idx1 = test_df["Drug 1"].map(drug_to_id).values.astype(np.int32)
    idx2 = test_df["Drug 2"].map(drug_to_id).values.astype(np.int32)

    # ── 2. Load Frozen FM Checkpoint ──────────────────────────────────────────
    fm_ckpt_path = os.path.join(MODELS_DIR, "fm", "fm_k8.npz")
    print(f"\n[1/2] Loading frozen FM checkpoint: {fm_ckpt_path}")
    fm_model = FM.load(fm_ckpt_path)
    fm_threshold = 0.15  # Chosen based on validation F1

    fm_probs = fm_model.predict_proba(idx1, idx2)
    fm_loss = fm_bce_loss(fm_probs, y_test)
    fm_preds = (fm_probs >= fm_threshold).astype(np.float32)

    # FM metrics
    fm_tp = int(((fm_preds == 1) & (y_test == 1)).sum())
    fm_fp = int(((fm_preds == 1) & (y_test == 0)).sum())
    fm_fn = int(((fm_preds == 0) & (y_test == 1)).sum())
    fm_tn = int(((fm_preds == 0) & (y_test == 0)).sum())

    fm_acc  = (fm_tp + fm_tn) / n_test
    fm_prec = fm_tp / (fm_tp + fm_fp) if (fm_tp + fm_fp) > 0 else 0.0
    fm_rec  = fm_tp / (fm_tp + fm_fn) if (fm_tp + fm_fn) > 0 else 0.0
    fm_f1   = 2 * fm_prec * fm_rec / (fm_prec + fm_rec) if (fm_prec + fm_rec) > 0 else 0.0

    print(f"  FM Test Metrics (Threshold={fm_threshold}):")
    print(f"    Loss: {fm_loss:.4f} | Acc: {fm_acc:.4f} | Prec: {fm_prec:.4f} | Rec: {fm_rec:.4f} | F1: {fm_f1:.4f}")
    print(f"    Confusion Matrix: TP={fm_tp}, FP={fm_fp}, FN={fm_fn}, TN={fm_tn}")

    # ── 3. Load Frozen DeepFM Checkpoint ──────────────────────────────────────
    deepfm_ckpt_path = os.path.join(MODELS_DIR, "deepfm", "deepfm_best.pt")
    print(f"\n[2/2] Loading frozen DeepFM checkpoint: {deepfm_ckpt_path}")
    deepfm_checkpoint = torch.load(deepfm_ckpt_path, map_location="cpu", weights_only=False)
    deepfm_config = deepfm_checkpoint["config"]

    deepfm_model = DeepFM(
        num_drugs=n_drugs,
        embedding_dim=deepfm_config["k"],
        hidden_dims=eval(deepfm_config["hidden_layers"]),
        dropout=deepfm_config["dropout"],
        use_fm=True,
        use_deep=True,
        shared_embeddings=True,
    )
    deepfm_model.load_state_dict(deepfm_checkpoint["model_state_dict"])
    deepfm_model.eval()

    deepfm_threshold = float(deepfm_config["threshold"])  # Tuned on validation F1 (0.50)

    t_idx1 = torch.tensor(idx1, dtype=torch.long)
    t_idx2 = torch.tensor(idx2, dtype=torch.long)
    t_y    = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        deepfm_logits = deepfm_model(t_idx1, t_idx2)
        deepfm_loss = float(torch.nn.BCEWithLogitsLoss()(deepfm_logits, t_y).item())
        deepfm_probs = torch.sigmoid(deepfm_logits).numpy()

    deepfm_preds = (deepfm_probs >= deepfm_threshold).astype(np.float32)

    # DeepFM metrics
    dfm_tp = int(((deepfm_preds == 1) & (y_test == 1)).sum())
    dfm_fp = int(((deepfm_preds == 1) & (y_test == 0)).sum())
    dfm_fn = int(((deepfm_preds == 0) & (y_test == 1)).sum())
    dfm_tn = int(((deepfm_preds == 0) & (y_test == 0)).sum())

    dfm_acc  = (dfm_tp + dfm_tn) / n_test
    dfm_prec = dfm_tp / (dfm_tp + dfm_fp) if (dfm_tp + dfm_fp) > 0 else 0.0
    dfm_rec  = dfm_tp / (dfm_tp + dfm_fn) if (dfm_tp + dfm_fn) > 0 else 0.0
    dfm_f1   = 2 * dfm_prec * dfm_rec / (dfm_prec + dfm_rec) if (dfm_prec + dfm_rec) > 0 else 0.0

    print(f"  DeepFM Test Metrics (Threshold={deepfm_threshold}):")
    print(f"    Loss: {deepfm_loss:.4f} | Acc: {dfm_acc:.4f} | Prec: {dfm_prec:.4f} | Rec: {dfm_rec:.4f} | F1: {dfm_f1:.4f}")
    print(f"    Confusion Matrix: TP={dfm_tp}, FP={dfm_fp}, FN={dfm_fn}, TN={dfm_tn}")

    # ── 4. Save results/final_comparison.csv ──────────────────────────────────
    comparison_rows = [
        {
            "model": "FM (from scratch)",
            "library": "NumPy",
            "k": 8,
            "architecture": "b + w1 + w2 + dot(v1, v2)",
            "threshold": fm_threshold,
            "test_loss": round(fm_loss, 6),
            "test_accuracy": round(fm_acc, 6),
            "test_precision": round(fm_prec, 6),
            "test_recall": round(fm_rec, 6),
            "test_f1": round(fm_f1, 6),
            "tp": fm_tp,
            "tn": fm_tn,
            "fp": fm_fp,
            "fn": fm_fn,
        },
        {
            "model": "DeepFM",
            "library": "PyTorch",
            "k": deepfm_config["k"],
            "architecture": f"FM + MLP{deepfm_config['hidden_layers']}",
            "threshold": deepfm_threshold,
            "test_loss": round(deepfm_loss, 6),
            "test_accuracy": round(dfm_acc, 6),
            "test_precision": round(dfm_prec, 6),
            "test_recall": round(dfm_rec, 6),
            "test_f1": round(dfm_f1, 6),
            "tp": dfm_tp,
            "tn": dfm_tn,
            "fp": dfm_fp,
            "fn": dfm_fn,
        },
    ]

    csv_path = os.path.join(RESULTS_DIR, "final_comparison.csv")
    fieldnames = list(comparison_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(f"\n[OK] Saved final comparison to: {csv_path}")

    # Also update fm_results.csv and deepfm_results.csv as required by the templates
    fm_results_dict = {
        "model": "FM (from scratch)",
        "split_ratio": "70/15/15",
        "embedding_dim_k": 8,
        "learning_rate": 0.01,
        "epochs": 50,
        "threshold": fm_threshold,
        "val_f1": 0.326333,
        "test_accuracy": round(fm_acc, 6),
        "test_precision": round(fm_prec, 6),
        "test_recall": round(fm_rec, 6),
        "test_f1": round(fm_f1, 6),
    }
    pd.DataFrame([fm_results_dict]).to_csv(os.path.join(RESULTS_DIR, "fm_results.csv"), index=False)

    deepfm_results_dict = {
        "model": "DeepFM",
        "library": "PyTorch",
        "split_ratio": "70/15/15",
        "embedding_dim_k": int(deepfm_config["k"]),
        "hidden_layers": str(deepfm_config["hidden_layers"]),
        "dropout": float(deepfm_config["dropout"]),
        "learning_rate": 0.001,
        "epochs": 20,
        "threshold": deepfm_threshold,
        "val_f1": round(deepfm_checkpoint["best_val_f1"], 6),
        "test_accuracy": round(dfm_acc, 6),
        "test_precision": round(dfm_prec, 6),
        "test_recall": round(dfm_rec, 6),
        "test_f1": round(dfm_f1, 6),
    }
    pd.DataFrame([deepfm_results_dict]).to_csv(os.path.join(RESULTS_DIR, "deepfm_results.csv"), index=False)

    # ── 5. Generate Confusion Matrix Plot ─────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def plot_cm(ax, tp, fp, fn, tn, title, color_map="Blues"):
        cm_matrix = np.array([[tn, fp], [fn, tp]])
        im = ax.imshow(cm_matrix, interpolation="nearest", cmap=color_map)
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        classes = ["Not Severe (0)", "Severe (1)"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

        thresh = cm_matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                val = cm_matrix[i, j]
                text_color = "white" if val > thresh else "black"
                tag = ""
                if i == 0 and j == 0: tag = " (TN)"
                elif i == 0 and j == 1: tag = " (FP)"
                elif i == 1 and j == 0: tag = " (FN)"
                elif i == 1 and j == 1: tag = " (TP)"
                ax.text(j, i, f"{val:,}{tag}", ha="center", va="center",
                        color=text_color, fontsize=11, fontweight="bold")

    plot_cm(ax1, fm_tp, fm_fp, fm_fn, fm_tn, f"FM from Scratch (Test F1={fm_f1:.4f})", "Blues")
    plot_cm(ax2, dfm_tp, dfm_fp, dfm_fn, dfm_tn, f"DeepFM (Test F1={dfm_f1:.4f})", "Greens")

    plt.suptitle("Final Test Confusion Matrices: FM from Scratch vs DeepFM", fontsize=14, y=1.02)
    plt.tight_layout()
    cm_plot_path = os.path.join(FIGURES_DIR, "test_confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved confusion matrices plot to: {cm_plot_path}")

    # ── 6. Generate Metrics Comparison Bar Plot ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fm_vals = [fm_acc, fm_prec, fm_rec, fm_f1]
    dfm_vals = [dfm_acc, dfm_prec, dfm_rec, dfm_f1]

    x = np.arange(len(metrics_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, fm_vals, width, label="FM (from scratch)", color="#4285F4", edgecolor="black")
    rects2 = ax.bar(x + width/2, dfm_vals, width, label="DeepFM (PyTorch)", color="#34A853", edgecolor="black")

    ax.bar_label(rects1, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(rects2, fmt="%.4f", padding=3, fontsize=9, fontweight="bold")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Test Performance Comparison: FM from Scratch vs DeepFM", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    metrics_plot_path = os.path.join(FIGURES_DIR, "test_metrics_comparison.png")
    plt.savefig(metrics_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved metrics comparison plot to: {metrics_plot_path}")

    print("\n" + "=" * 65)
    print("FINAL SUMMARY COMPARISON TABLE")
    print("=" * 65)
    print(pd.DataFrame(comparison_rows)[["model", "test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1", "tp", "tn", "fp", "fn"]].to_string(index=False))


if __name__ == "__main__":
    evaluate_test()
