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
