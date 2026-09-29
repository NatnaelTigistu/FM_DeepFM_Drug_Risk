"""
deepfm.py  —  DeepFM architecture implemented in PyTorch
=========================================================

Architecture Overview
---------------------
DeepFM combines a Factorization Machine (FM) component and a Deep MLP component
sharing the same drug embeddings:

        drug_1 ID        drug_2 ID
            │                │
      [shared embeddings, size k]
            │                │
   FM part:  bias + w1[i] + w2[j] + <v_i, v_j>  ───┐
                                                    ├──► sum (final logit) ──► sigmoid ──► P(Severe)
   Deep part: concat(v_i, v_j) → MLP → 1 value  ───┘

Components:
1. Embedding:
   - Shared embedding table of size (num_drugs, k), mapping drug IDs to dense vectors.
   - For Drug 1 (index i) -> v1 = embedding(i)
   - For Drug 2 (index j) -> v2 = embedding(j)

2. FM Branch:
   - Global bias b (scalar parameter)
   - Linear term w1 for Drug 1 (embedding table of size (num_drugs, 1))
   - Linear term w2 for Drug 2 (embedding table of size (num_drugs, 1))
   - Second-order interaction: dot product <v1, v2>
   - FM logit = b + w1[i] + w2[j] + <v1, v2>

3. Deep Branch:
   - Input: concat([v1, v2]) of size 2*k
   - Small MLP: Linear(2k, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)
   - Deep logit = MLP output (scalar per sample)

4. Combined Prediction:
   - final logit = FM logit + Deep logit
   - probability = sigmoid(final_logit)
   - Loss function: nn.BCEWithLogitsLoss()
"""

from typing import Union, Sequence, Optional, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback placeholder so module import doesn't crash before torch is installed
    class nn:
        class Module:
            pass


if TORCH_AVAILABLE:
    class DeepFM(nn.Module):
        """
        DeepFM model for Drug-Drug Interaction Risk Prediction.

        Parameters
        ----------
        num_drugs : int
            Total number of unique drugs in the vocabulary.
        embedding_dim : int, default=8
            Dimension k of the latent drug embeddings.
        hidden_dims : tuple of int, default=(32,)
            Sizes of hidden layers in the deep MLP branch.
        dropout : float, default=0.0
            Dropout probability in the MLP branch.
        use_fm : bool, default=True
            Whether to include the FM component (used for ablation studies).
        use_deep : bool, default=True
            Whether to include the Deep component (used for ablation studies).
        shared_embeddings : bool, default=True
            Whether Drug 1 and Drug 2 share the embedding matrix.
        """

        def __init__(
            self,
            num_drugs: int,
            embedding_dim: int = 8,
            hidden_dims: Sequence[int] = (32,),
            dropout: float = 0.0,
            use_fm: bool = True,
            use_deep: bool = True,
            shared_embeddings: bool = True,
        ):
            super().__init__()
            self.num_drugs = num_drugs
            self.embedding_dim = embedding_dim
            self.hidden_dims = tuple(hidden_dims)
            self.dropout_rate = dropout
            self.use_fm = use_fm
            self.use_deep = use_deep
            self.shared_embeddings = shared_embeddings

            # ── 1. Embeddings ─────────────────────────────────────────────────
            # Shared or separate drug embedding tables
            if shared_embeddings:
                self.embedding = nn.Embedding(num_drugs, embedding_dim)
                self.drug1_embedding = self.embedding
                self.drug2_embedding = self.embedding
            else:
                self.drug1_embedding = nn.Embedding(num_drugs, embedding_dim)
                self.drug2_embedding = nn.Embedding(num_drugs, embedding_dim)

            # Initialize embeddings with uniform scale similar to NumPy FM
            scale = 1.0 / np.sqrt(embedding_dim)
            nn.init.uniform_(self.drug1_embedding.weight, -scale, scale)
            if not shared_embeddings:
                nn.init.uniform_(self.drug2_embedding.weight, -scale, scale)

            # ── 2. FM Branch Components ───────────────────────────────────────
            # Global bias scalar
            self.bias = nn.Parameter(torch.zeros(1))

            # Linear weights for Drug 1 and Drug 2
            self.w1 = nn.Embedding(num_drugs, 1)
            self.w2 = nn.Embedding(num_drugs, 1)
            nn.init.zeros_(self.w1.weight)
            nn.init.zeros_(self.w2.weight)

            # ── 3. Deep Branch Components ─────────────────────────────────────
            # Input dimension is concat(v1, v2) -> 2 * embedding_dim
            mlp_layers = []
            in_dim = 2 * embedding_dim
            for h_dim in hidden_dims:
                mlp_layers.append(nn.Linear(in_dim, h_dim))
                mlp_layers.append(nn.ReLU())
                if dropout > 0.0:
                    mlp_layers.append(nn.Dropout(p=dropout))
                in_dim = h_dim

            # Final linear projection to scalar logit
            mlp_layers.append(nn.Linear(in_dim, 1))
            self.mlp = nn.Sequential(*mlp_layers)

        def get_embeddings(self, idx1: torch.Tensor, idx2: torch.Tensor):
            """Fetch latent embedding vectors for Drug 1 and Drug 2."""
            v1 = self.drug1_embedding(idx1)  # (batch_size, k)
            v2 = self.drug2_embedding(idx2)  # (batch_size, k)
            return v1, v2

        def forward_fm(self, idx1: torch.Tensor, idx2: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
            """
            Compute FM branch logit:
            z_fm = bias + w1[idx1] + w2[idx2] + <v1, v2>
            """
            w1_term = self.w1(idx1).squeeze(-1)                   # (batch_size,)
            w2_term = self.w2(idx2).squeeze(-1)                   # (batch_size,)
            dot_term = torch.sum(v1 * v2, dim=-1)                 # (batch_size,)
            fm_logit = self.bias + w1_term + w2_term + dot_term   # (batch_size,)
            return fm_logit

        def forward_deep(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
            """
            Compute Deep branch logit:
            z_deep = MLP(concat([v1, v2]))
            """
            concat_emb = torch.cat([v1, v2], dim=-1)             # (batch_size, 2*k)
            deep_logit = self.mlp(concat_emb).squeeze(-1)        # (batch_size,)
            return deep_logit

        def forward(self, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
            """
            Forward pass returning the combined raw logits.

            Parameters
            ----------
            idx1 : torch.Tensor of shape (batch_size,), dtype long
            idx2 : torch.Tensor of shape (batch_size,), dtype long

            Returns
            -------
            torch.Tensor of shape (batch_size,) containing raw logits.
            Use torch.sigmoid(output) to obtain probabilities.
            """
            v1, v2 = self.get_embeddings(idx1, idx2)

            if self.use_fm and self.use_deep:
                fm_logit = self.forward_fm(idx1, idx2, v1, v2)
                deep_logit = self.forward_deep(v1, v2)
                return fm_logit + deep_logit
            elif self.use_fm:
                return self.forward_fm(idx1, idx2, v1, v2)
            elif self.use_deep:
                return self.forward_deep(v1, v2)
            else:
                raise ValueError("At least one of use_fm or use_deep must be True.")

        def forward_components(self, idx1: torch.Tensor, idx2: torch.Tensor) -> Dict[str, torch.Tensor]:
            """
            Expose internal components:
            - FM logit
            - Deep logit
            - Final combined logit
            - Probability (via sigmoid)
            """
            v1, v2 = self.get_embeddings(idx1, idx2)

            batch_size = idx1.shape[0]
            device = idx1.device

            if self.use_fm:
                fm_logit = self.forward_fm(idx1, idx2, v1, v2)
            else:
                fm_logit = torch.zeros(batch_size, device=device)

            if self.use_deep:
                deep_logit = self.forward_deep(v1, v2)
            else:
                deep_logit = torch.zeros(batch_size, device=device)

            if self.use_fm and self.use_deep:
                final_logit = fm_logit + deep_logit
            elif self.use_fm:
                final_logit = fm_logit
            else:
                final_logit = deep_logit

            probability = torch.sigmoid(final_logit)

            return {
                "fm_logit": fm_logit,
                "deep_logit": deep_logit,
                "final_logit": final_logit,
                "probability": probability,
            }

else:
    class DeepFM:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DeepFM. Please install torch.")


# ── Inspection and Inference Helpers ─────────────────────────────────────────

def explain_pair_deepfm(
    model: "DeepFM",
    drug1: str,
    drug2: str,
    drug_to_id: dict,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> dict:
    """
    Given two drug names, evaluate the trained DeepFM model without retraining
    and decompose the prediction into FM contribution, Deep contribution,
    final logit, and probability.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to run explain_pair_deepfm.")

    if drug1 not in drug_to_id:
        raise KeyError(f"Drug '{drug1}' not in training vocabulary.")
    if drug2 not in drug_to_id:
        raise KeyError(f"Drug '{drug2}' not in training vocabulary.")

    model.eval()

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    i = drug_to_id[drug1]
    j = drug_to_id[drug2]

    idx1_t = torch.tensor([i], dtype=torch.long, device=device)
    idx2_t = torch.tensor([j], dtype=torch.long, device=device)

    with torch.no_grad():
        v1, v2 = model.get_embeddings(idx1_t, idx2_t)

        bias_val = float(model.bias.item())
        w1_val = float(model.w1(idx1_t).item())
        w2_val = float(model.w2(idx2_t).item())
        dot_val = float(torch.sum(v1 * v2).item())

        fm_logit = bias_val + w1_val + w2_val + dot_val if model.use_fm else 0.0
        deep_logit = float(model.forward_deep(v1, v2).item()) if model.use_deep else 0.0

        if model.use_fm and model.use_deep:
            final_logit = fm_logit + deep_logit
        elif model.use_fm:
            final_logit = fm_logit
        else:
            final_logit = deep_logit

        prob = float(torch.sigmoid(torch.tensor(final_logit)).item())

    return {
        "drug1": drug1,
        "drug2": drug2,
        "bias": bias_val,
        "w1_term": w1_val,
        "w2_term": w2_val,
        "dot_product": dot_val,
        "fm_logit": fm_logit,
        "deep_logit": deep_logit,
        "final_logit": final_logit,
        "probability": prob,
        "predicted_class": int(prob >= threshold),
    }


def print_deepfm_explanation(explanation: dict, threshold: float = 0.5) -> None:
    """Pretty-print the component breakdown for a drug pair."""
    e = explanation
    print("=" * 56)
    print(f"  DeepFM Prediction Breakdown")
    print(f"  Drug 1 : {e['drug1']}")
    print(f"  Drug 2 : {e['drug2']}")
    print("=" * 56)
    print(f"  [FM Branch]")
    print(f"    bias              b         = {e['bias']:+.6f}")
    print(f"    linear weight     w1[i]     = {e['w1_term']:+.6f}")
    print(f"    linear weight     w2[j]     = {e['w2_term']:+.6f}")
    print(f"    dot product       <v1, v2>  = {e['dot_product']:+.6f}")
    print(f"    FM logit                    = {e['fm_logit']:+.6f}")
    print(f"  [Deep Branch]")
    print(f"    Deep MLP logit              = {e['deep_logit']:+.6f}")
    print(f"  {'─'*48}")
    print(f"  Final Logit  (FM + Deep)      = {e['final_logit']:+.6f}")
    print(f"  Probability  σ(Final Logit)   = {e['probability']:.6f}")
    print(f"  Threshold                     = {threshold}")
    print(f"  Prediction                    = "
          f"{'Severe (1)' if e['predicted_class'] else 'Not Severe (0)'}")
    print("=" * 56)


# ── Training and Evaluation Utilities ─────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set random seeds across standard library, numpy, and torch for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


if TORCH_AVAILABLE:
    class DrugPairDataset(torch.utils.data.Dataset):
        """Dataset of drug pairs and risk labels."""
        def __init__(self, idx1: np.ndarray, idx2: np.ndarray, labels: np.ndarray):
            self.idx1 = torch.tensor(idx1, dtype=torch.long)
            self.idx2 = torch.tensor(idx2, dtype=torch.long)
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int):
            return self.idx1[idx], self.idx2[idx], self.labels[idx]


def build_dataloaders(
    train_df,
    val_df,
    test_df=None,
    drug_to_id: Optional[dict] = None,
    batch_size: int = 256,
    seed: int = 42,
):
    """Create PyTorch DataLoaders for train, validation (and optionally test)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for build_dataloaders.")

    set_seed(seed)

    def encode(df):
        i1 = df["Drug 1"].map(drug_to_id).values.astype(np.int32)
        i2 = df["Drug 2"].map(drug_to_id).values.astype(np.int32)
        y = df["Label"].values.astype(np.float32)
        return DrugPairDataset(i1, i2, y)

    train_ds = encode(train_df)
    val_ds = encode(val_df)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    if test_df is not None:
        test_ds = encode(test_df)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False
        )
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def evaluate_metrics(
    model: "DeepFM",
    dataloader,
    criterion=None,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate DeepFM on a dataloader and compute loss, accuracy, precision, recall, and F1.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for evaluate_metrics.")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    model.eval()
    total_loss = 0.0
    total_samples = 0

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for idx1, idx2, y in dataloader:
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
            y = y.to(device)

            logits = model(idx1, idx2)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(y)
            total_samples += len(y)

            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_best_threshold(
    model: "DeepFM",
    dataloader,
    thresholds: Sequence[float] = tuple(round(t, 2) for t in np.arange(0.10, 0.65, 0.05)),
    device: Optional[str] = None,
) -> tuple[float, float]:
    """Find the classification threshold that maximizes validation F1."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for find_best_threshold.")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for idx1, idx2, y in dataloader:
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
            logits = model(idx1, idx2)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    best_thresh = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(np.float32)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh, best_f1


def train_deepfm(
    model: "DeepFM",
    train_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    threshold: Optional[float] = None,
    device: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Train DeepFM model with Adam and BCEWithLogitsLoss.
    Tracks training loss, validation loss, accuracy, precision, recall, and F1.
    Saves and returns the best model weights based on validation F1.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for train_deepfm.")

    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        total_train_samples = 0

        for idx1, idx2, y in train_loader:
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(idx1, idx2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            total_train_samples += len(y)

        avg_train_loss = train_loss / max(total_train_samples, 1)

        # Tune threshold dynamically if not fixed, or use fixed
        if threshold is None:
            curr_thresh, _ = find_best_threshold(model, val_loader, device=device)
        else:
            curr_thresh = threshold

        val_m = evaluate_metrics(model, val_loader, criterion=criterion, threshold=curr_thresh, device=device)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_m["loss"])
        history["val_accuracy"].append(val_m["accuracy"])
        history["val_precision"].append(val_m["precision"])
        history["val_recall"].append(val_m["recall"])
        history["val_f1"].append(val_m["f1"])

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = val_m.copy()
            best_metrics["threshold"] = curr_thresh
            best_metrics["epoch"] = epoch

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(
                f"  Epoch {epoch:>2}/{epochs} | "
                f"train_loss: {avg_train_loss:.4f} | "
                f"val_loss: {val_m['loss']:.4f} | "
                f"val_f1: {val_m['f1']:.4f} | "
                f"val_acc: {val_m['accuracy']:.4f} | "
                f"val_rec: {val_m['recall']:.4f}"
            )

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_metrics": best_metrics,
        "model": model,
    }

