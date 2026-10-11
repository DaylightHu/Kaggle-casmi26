from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import check_rdkit, scorer_key


def check_submission(submission, expected_ids, allow_mismatch=False):
    check_rdkit(allow_mismatch)
    frame = pd.read_csv(submission, dtype=str, keep_default_na=False) if isinstance(submission, (str, Path)) else submission
    if not {"molecule_id", "smiles"}.issubset(frame.columns):
        raise ValueError("Submission needs molecule_id and smiles columns")
    if frame.empty or frame[["molecule_id", "smiles"]].isna().any().any():
        raise ValueError("Submission is empty or contains nulls")
    if frame["molecule_id"].duplicated().any():
        raise ValueError("Duplicate molecule_id")
    ids = frame["molecule_id"].astype(str)
    if any(not v.strip() for v in ids) or set(ids) != set(map(str, expected_ids)):
        raise ValueError("Submission molecule IDs do not exactly match test molecule IDs")
    for row in frame.itertuples(index=False):
        guesses = str(row.smiles).split(";")
        if not 1 <= len(guesses) <= 25 or any(not x.strip() for x in guesses):
            raise ValueError(f"Invalid candidate list for {row.molecule_id}")
        keys = [scorer_key(x) for x in guesses]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Equivalent duplicate structures waste ranks: {row.molecule_id}")
    return frame


def evaluate_submission(submission, truth, allow_mismatch=False):
    check_rdkit(allow_mismatch)
    if truth["molecule_id"].duplicated().any() or truth.empty:
        raise ValueError("Truth must have one nonempty row per molecule")
    submission = check_submission(submission, truth["molecule_id"], allow_mismatch)
    ordered = submission.set_index("molecule_id").loc[truth["molecule_id"]]
    ranks = []
    for correct, guesses in zip(truth["smiles"], ordered["smiles"]):
        key = scorer_key(correct)
        keys = [scorer_key(s) for s in guesses.split(";")][:25]
        ranks.append(keys.index(key)+1 if key in keys else 0)
    ranks = np.asarray(ranks)
    reciprocal = np.zeros(len(ranks), dtype=float)
    good = ranks > 0
    reciprocal[good] = 1.0/ranks[good]
    return {"molecules": len(ranks), "mrr_at_25": float(reciprocal.mean()),
            "top1": float((ranks == 1).mean()), "recall_at_25": float(good.mean())}
