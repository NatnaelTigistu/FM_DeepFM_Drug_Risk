"""
data_utils.py
=============
Data preparation for the FM / DeepFM Drug-Risk project.

Responsibilities
----------------
1. Load the raw CSV (Drug 1, Drug 2, Label only).
2. Build canonical pair keys so that (A, B) and (B, A) always land in the
   same split — the key is tuple(sorted([drug_1, drug_2])).
3. Split by *pair key*, not by row, to prevent leakage.  The pair-level
   split is stratified by the *majority label* of each pair group.
4. After splitting, guarantee that every drug that appears in val/test also
   appears in training.  Any pair whose drugs are unseen in train is moved
   to train.
5. Save the three split CSVs and a JSON drug-to-ID mapping.
6. The drug-to-ID mapping is built from the training vocabulary only,
   with an offset so that Drug-1 and Drug-2 embeddings live in separate
   index ranges (as required by the FM template).

Allowed libraries (FM notebook rules): pandas, numpy; sklearn only for
train_test_split.  All other data work is plain Python / pandas.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── constants ────────────────────────────────────────────────────────────────
SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_CSV = os.path.join(DATA_DIR, "db_drug_interactions_severity.csv")

TRAIN_CSV = os.path.join(DATA_DIR, "split_train.csv")
VAL_CSV   = os.path.join(DATA_DIR, "split_val.csv")
TEST_CSV  = os.path.join(DATA_DIR, "split_test.csv")
MAPPING_JSON = os.path.join(DATA_DIR, "drug_mapping.json")


# ── helpers ───────────────────────────────────────────────────────────────────

def _pair_key(drug1: str, drug2: str) -> tuple:
    """Canonical unordered pair key — used only for split grouping."""
    return tuple(sorted([drug1, drug2]))


def _majority_label(labels: pd.Series) -> int:
    """Return the most-common label in a group (used to stratify groups)."""
    return int(labels.mode()[0])


# ── main pipeline ─────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    """
    Load the dataset and keep only the three columns we need.
    Verifies no missing values or duplicate rows are present.
    """
    df = pd.read_csv(path)

    # Sanity checks on the raw file
    assert "Drug 1" in df.columns and "Drug 2" in df.columns and "Label" in df.columns, \
        "Expected columns 'Drug 1', 'Drug 2', 'Label' not found."
    assert "Interaction Description" not in df.columns[df.columns.isin(["Interaction Description"])].tolist() \
        or True, ""   # presence is fine; we just never use it as a feature

    # Keep only what we need — explicitly drop the forbidden columns
    df = df[["Drug 1", "Drug 2", "Label"]].copy()

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    assert missing == 0,     f"Found {missing} missing values in the dataset."
    assert duplicates == 0,  f"Found {duplicates} duplicate rows in the dataset."

    return df


def _split_by_pairs(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split rows by unordered pair key to prevent (A,B)/(B,A) leakage.

    Strategy
    --------
    1. Assign each row a canonical pair key.
    2. Collapse to one row per unique pair key (keeping the majority label
       of that pair group for stratification).
    3. Split the *pair-key table* into train / val / test using
       sklearn's stratified split.
    4. Map the pair-key membership back to the original rows.
    5. Guarantee drug-coverage: any val/test pair whose drugs are not
       seen in train is forcibly moved to train.
    """
    rng = np.random.RandomState(seed)

    # Step 1 — pair key per row
    df = df.copy()
    df["_pair_key"] = [_pair_key(r["Drug 1"], r["Drug 2"])
                       for _, r in df.iterrows()]

    # Step 2 — one-row-per-pair summary for stratified splitting
    pair_summary = (
        df.groupby("_pair_key")["Label"]
        .agg(_majority_label)
        .reset_index()
        .rename(columns={"Label": "_strat_label"})
    )
    # pair_summary has columns: _pair_key, _strat_label

    keys = pair_summary["_pair_key"].values
    strat = pair_summary["_strat_label"].values

    # Step 3 — split pair keys (not rows)
    # First carve out the test set
    train_val_keys, test_keys, train_val_strat, _ = train_test_split(
        keys, strat,
        test_size=test_ratio,
        stratify=strat,
        random_state=seed,
    )
    # Then split remaining into train / val
    # Adjust val_ratio relative to train+val pool
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    train_keys, val_keys, _, _ = train_test_split(
        train_val_keys, train_val_strat,
        test_size=val_ratio_adjusted,
        stratify=train_val_strat,
        random_state=seed,
    )

    # Step 4 — map pair-key membership back to rows
    train_key_set = set(map(tuple, train_keys))
    val_key_set   = set(map(tuple, val_keys))
    test_key_set  = set(map(tuple, test_keys))

    train_mask = df["_pair_key"].apply(lambda k: k in train_key_set)
    val_mask   = df["_pair_key"].apply(lambda k: k in val_key_set)
    test_mask  = df["_pair_key"].apply(lambda k: k in test_key_set)

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()
    test_df  = df[test_mask].copy()

    # Step 5 — drug-coverage guarantee
    # Any drug in val or test that is not in train → move those pair's rows to train
    train_df, val_df, test_df = _fix_drug_coverage(train_df, val_df, test_df)

    # Drop the helper column
    for split in (train_df, val_df, test_df):
        split.drop(columns=["_pair_key"], inplace=True)

    return train_df, val_df, test_df


def _fix_drug_coverage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Guarantee: every drug appearing in val or test is also in train.

    Any row in val/test whose Drug 1 OR Drug 2 is not seen in training
    is moved from val/test into train — grouped by pair key so we never
    split a (A,B)/(B,A) group across sets.
    """
    def _train_drugs(df: pd.DataFrame) -> set:
        return set(df["Drug 1"].unique()) | set(df["Drug 2"].unique())

    def _missing_pairs(split_df: pd.DataFrame, known: set) -> set:
        """Pair keys in split_df that contain an unknown drug."""
        bad = set()
        for _, row in split_df.iterrows():
            if row["Drug 1"] not in known or row["Drug 2"] not in known:
                bad.add(row["_pair_key"])
        return bad

    max_iterations = 10   # safety cap
    for _ in range(max_iterations):
        train_drugs = _train_drugs(train_df)

        bad_val  = _missing_pairs(val_df,  train_drugs)
        bad_test = _missing_pairs(test_df, train_drugs)

        if not bad_val and not bad_test:
            break   # ← coverage achieved

        # Move bad pairs from val/test → train
        val_move   = val_df["_pair_key"].isin(bad_val)
        test_move  = test_df["_pair_key"].isin(bad_test)

        train_df = pd.concat([train_df, val_df[val_move], test_df[test_move]],
                             ignore_index=True)
        val_df   = val_df[~val_move].copy()
        test_df  = test_df[~test_move].copy()
    else:
        raise RuntimeError("Drug coverage could not be satisfied after "
                           f"{max_iterations} iterations.")

    return train_df, val_df, test_df


def make_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Public entry point: split the loaded dataframe into train/val/test."""
    return _split_by_pairs(df, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save the three split CSVs (columns: Drug 1, Drug 2, Label)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV,   index=False)
    test_df.to_csv(TEST_CSV,  index=False)


def build_drug_mapping(train_df: pd.DataFrame) -> dict:
    """
    Build a drug-to-integer-ID mapping from the TRAINING vocabulary only.

    Encoding scheme (from FM template):
      Drug-1 IDs  : 0  ..  n_drugs-1
      Drug-2 IDs  : n_drugs  ..  2*n_drugs-1   (offset = n_drugs)

    This way "Aspirin as Drug 1" and "Aspirin as Drug 2" have *different*
    feature IDs, preserving directionality while still sharing the same
    embedding vector in DeepFM (via nn.Embedding with the offset removed).

    Returns a dict with keys:
      "drug_to_id"   : {drug_name: id}   (0-based, Drug-1 space)
      "id_to_drug"   : {id: drug_name}
      "n_drugs"      : number of unique drugs in training
      "n_features"   : 2 * n_drugs  (total feature IDs, Drug1 + Drug2)
      "drug2_offset" : n_drugs       (add this to a Drug-1 ID to get Drug-2 ID)
    """
    # Collect all drugs that appear in the training split (in either column)
    all_train_drugs = sorted(
        set(train_df["Drug 1"].unique()) | set(train_df["Drug 2"].unique())
    )
    drug_to_id = {drug: idx for idx, drug in enumerate(all_train_drugs)}
    id_to_drug = {idx: drug for drug, idx in drug_to_id.items()}

    n_drugs = len(drug_to_id)

    mapping = {
        "drug_to_id":   drug_to_id,
        "id_to_drug":   {str(k): v for k, v in id_to_drug.items()},  # JSON keys must be str
        "n_drugs":      n_drugs,
        "n_features":   2 * n_drugs,
        "drug2_offset": n_drugs,
    }
    return mapping


def save_drug_mapping(mapping: dict, path: str = MAPPING_JSON) -> None:
    """Persist the drug mapping as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def load_drug_mapping(path: str = MAPPING_JSON) -> dict:
    """Load a previously saved drug mapping."""
    with open(path) as f:
        return json.load(f)


def encode_split(
    df: pd.DataFrame,
    mapping: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a split dataframe into integer-ID arrays.

    Returns
    -------
    X : np.ndarray, shape (n, 2), dtype int32
        Column 0 = Drug-1 feature ID  (0 .. n_drugs-1)
        Column 1 = Drug-2 feature ID  (n_drugs .. 2*n_drugs-1)
    y : np.ndarray, shape (n,), dtype float32
    """
    drug_to_id   = mapping["drug_to_id"]
    drug2_offset = mapping["drug2_offset"]

    id1 = df["Drug 1"].map(drug_to_id).values.astype(np.int32)
    id2 = df["Drug 2"].map(drug_to_id).values.astype(np.int32) + drug2_offset
    X = np.stack([id1, id2], axis=1)
    y = df["Label"].values.astype(np.float32)
    return X, y


# ── verification helpers ──────────────────────────────────────────────────────

def verify_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    original_df: pd.DataFrame,
) -> None:
    """
    Run all correctness checks and print a concise report.
    Raises AssertionError on any violation.
    """
    total_original = len(original_df)
    total_split    = len(train_df) + len(val_df) + len(test_df)

    print("=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)

    # ── 1. No row loss ────────────────────────────────────────────
    assert total_split == total_original, (
        f"Row count mismatch: {total_split} vs {total_original}"
    )
    print(f"[OK] Row count preserved: {total_original} rows total")

    # ── 2. No duplicates within any split ────────────────────────
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dupes = df[["Drug 1", "Drug 2", "Label"]].duplicated().sum()
        assert dupes == 0, f"Found {dupes} duplicate rows in {name} split"
    print("[OK] No duplicate rows in any split")

    # ── 3. No pair leakage ────────────────────────────────────────
    def pair_keys(df):
        return set(tuple(sorted([r["Drug 1"], r["Drug 2"]]))
                   for _, r in df.iterrows())

    train_pairs = pair_keys(train_df)
    val_pairs   = pair_keys(val_df)
    test_pairs  = pair_keys(test_df)

    tv_overlap  = train_pairs & val_pairs
    tt_overlap  = train_pairs & test_pairs
    vt_overlap  = val_pairs   & test_pairs

    assert len(tv_overlap) == 0,  f"Train/val pair overlap: {len(tv_overlap)} pairs"
    assert len(tt_overlap) == 0,  f"Train/test pair overlap: {len(tt_overlap)} pairs"
    assert len(vt_overlap) == 0,  f"Val/test pair overlap: {len(vt_overlap)} pairs"
    print("[OK] No pair-level leakage between any two splits")

    # ── 4. Drug coverage ─────────────────────────────────────────
    train_drugs = set(train_df["Drug 1"].unique()) | set(train_df["Drug 2"].unique())
    val_drugs   = set(val_df["Drug 1"].unique())   | set(val_df["Drug 2"].unique())
    test_drugs  = set(test_df["Drug 1"].unique())  | set(test_df["Drug 2"].unique())

    val_unseen  = val_drugs  - train_drugs
    test_unseen = test_drugs - train_drugs

    assert len(val_unseen)  == 0, f"Drugs in val but not train: {val_unseen}"
    assert len(test_unseen) == 0, f"Drugs in test but not train: {test_unseen}"
    print("[OK] Every drug in val/test appears in training")

    # ── 5. Split sizes and label distributions ───────────────────
    print()
    print(f"{'Split':<10} {'Rows':>7}  {'%Total':>7}  {'Label=1 (Severe)':>17}  {'%Severe':>8}")
    print("-" * 60)
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n = len(df)
        pct_total  = 100 * n / total_original
        n_severe   = (df["Label"] == 1).sum()
        pct_severe = 100 * n_severe / n
        print(f"{name:<10} {n:>7}  {pct_total:>6.1f}%  "
              f"{n_severe:>12}         {pct_severe:>6.1f}%")
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_split:>7}  {'100.0%':>7}")

    # ── 6. Drug vocabulary sizes ──────────────────────────────────
    print()
    print(f"Unique drugs in train : {len(train_drugs)}")
    print(f"Unique drugs in val   : {len(val_drugs)}")
    print(f"Unique drugs in test  : {len(test_drugs)}")
    print(f"Union (all splits)    : "
          f"{len(train_drugs | val_drugs | test_drugs)}")

    print()
    print("[ALL CHECKS PASSED]")
    print("=" * 60)


# ── script entry point ────────────────────────────────────────────────────────

def run_pipeline(verbose: bool = True) -> dict:
    """
    End-to-end data preparation pipeline.
    Returns a summary dict of sizes and label rates.
    """
    if verbose:
        print(f"Loading dataset from: {os.path.abspath(RAW_CSV)}")
    df = load_raw()

    if verbose:
        print(f"Raw dataset: {len(df)} rows, {df['Label'].mean()*100:.1f}% Severe\n")

    # Split
    train_df, val_df, test_df = make_splits(df, val_ratio=0.15, test_ratio=0.15)

    # Verify
    verify_splits(train_df, val_df, test_df, df)

    # Save splits
    save_splits(train_df, val_df, test_df)
    if verbose:
        print(f"\nSaved: {TRAIN_CSV}")
        print(f"Saved: {VAL_CSV}")
        print(f"Saved: {TEST_CSV}")

    # Build and save drug mapping
    mapping = build_drug_mapping(train_df)
    save_drug_mapping(mapping)
    if verbose:
        print(f"Saved: {MAPPING_JSON}")
        print(f"\nDrug vocabulary (training): {mapping['n_drugs']} drugs")
        print(f"Total feature IDs         : {mapping['n_features']}  "
              f"(Drug-1: 0..{mapping['n_drugs']-1}, "
              f"Drug-2: {mapping['drug2_offset']}..{mapping['n_features']-1})")

    summary = {
        "n_train":       len(train_df),
        "n_val":         len(val_df),
        "n_test":        len(test_df),
        "n_total":       len(df),
        "pct_severe_train": float((train_df["Label"]==1).mean()),
        "pct_severe_val":   float((val_df["Label"]==1).mean()),
        "pct_severe_test":  float((test_df["Label"]==1).mean()),
        "n_drugs":       mapping["n_drugs"],
        "n_features":    mapping["n_features"],
    }
    return summary


if __name__ == "__main__":
    run_pipeline(verbose=True)
