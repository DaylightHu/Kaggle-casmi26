import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .cache import ReferenceCache
from .candidates import CandidatePool
from .chemistry import check_rdkit, neutral_mass
from .common import write_json
from .metric import check_submission
from .spectra import clean_spectrum, mean_energy, mode_code, query_quality, reference_scores, timstof


def build_queries(frame, cache):
    required = {"molecule_id", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct", "ionization_mode"}
    if not required.issubset(frame.columns) or frame.empty or frame["molecule_id"].isna().any():
        raise ValueError("Invalid or empty query table")
    queries = []
    for row in frame.to_dict("records"):
        mz, weights = clean_spectrum(row["ms2_mzs"], row["ms2_normalized_intensities"], float(row["precursor_mz"]), cache.manifest["preprocess"])
        # Unknown test adducts fail explicitly instead of silently assuming [M+H]+.
        mass = neutral_mass(row["precursor_mz"], row["adduct"])
        queries.append({"mz": mz, "weights": weights, "precursor": float(row["precursor_mz"]),
                        "mass": mass, "adduct": cache.adduct_map.get(row["adduct"], 0),
                        "mode": mode_code(row["ionization_mode"]), "ce": mean_energy(row.get("collision_energy_ev")),
                        "timstof": timstof(row.get("instrument_type")),
                        "quality": query_quality(mz, weights, row.get("base_peak_intensity"))})
    return queries


def weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(values)[order[np.searchsorted(cumulative, cumulative[-1]/2)]])


class Ranker:
    def __init__(self, config, cache, pool, checkpoint=None, device="auto", blocked_sids=None, blocked_rows=None):
        self.config, self.cache, self.pool = config, cache, pool
        self.blocked_sids = set(blocked_sids or [])
        self.blocked_rows = np.zeros(len(cache.meta), dtype=bool)
        if blocked_rows:
            self.blocked_rows[np.asarray(blocked_rows, dtype=int)] = True
        self.predictor = None
        if checkpoint:
            from .fingerprint import FingerprintPredictor
            self.predictor = FingerprintPredictor(checkpoint, cache, device)

    def choose_references(self, sid, query):
        if sid in self.blocked_sids:
            return np.empty(0, dtype=np.int64)
        rows = np.asarray(self.cache.rows_for_sid(sid))
        if len(rows) == 0:
            return rows
        meta = self.cache.meta
        keep = ~self.blocked_rows[rows]
        if query["mode"]:
            keep &= (meta["mode"][rows] == query["mode"]) | (meta["mode"][rows] == 0)
        rows = rows[keep]
        limit = self.config["retrieval"]["max_references_per_structure"]
        if len(rows) > limit:
            priority = 2 * (meta["adduct"][rows] == query["adduct"]).astype(float)
            priority += meta["timstof"][rows].astype(float)
            if np.isfinite(query["ce"]):
                distance = np.abs(meta["ce"][rows] - query["ce"])
                priority += np.exp(-np.nan_to_num(distance, nan=1000)/25.0)
            priority += np.minimum(meta["n"][rows]/1000.0, 0.1)
            rows = rows[np.argsort(-priority, kind="stable")[:limit]]
        return rows

    def spectral_score(self, ids, queries):
        scores = np.zeros((len(queries), len(ids)), dtype=np.float32)
        cfg = self.config["retrieval"]
        for j, query in enumerate(queries):
            parts, destinations = [], []
            for i, candidate in enumerate(ids):
                if candidate >= self.pool.train_size:
                    continue
                rows = self.choose_references(int(candidate), query)
                if len(rows):
                    parts.append(rows)
                    destinations.append(np.full(len(rows), i, dtype=np.int64))
            if not parts:
                continue
            refs, owners = np.concatenate(parts), np.concatenate(destinations)
            # Bound scratch memory even for a large same-mass candidate set.
            for start in range(0, len(refs), 2048):
                rows = refs[start:start+2048]
                meta = self.cache.meta[rows]
                result = reference_scores(query["mz"], query["weights"], query["precursor"],
                         query["adduct"], query["mode"], query["ce"],
                         self.cache.mzs[rows], self.cache.weights[rows].astype(np.float32),
                         meta["n"], meta["precursor"], meta["adduct"], meta["mode"], meta["ce"],
                         meta["timstof"], cfg["fragment_da"], cfg["fragment_ppm"])
                np.maximum.at(scores[j], owners[start:start+2048], result)
        # First aggregate within an adduct+mode, so repeated acquisitions don't dominate.
        groups = {}
        for i, q in enumerate(queries):
            groups.setdefault((q["adduct"], q["mode"]), []).append(i)
        grouped = []
        for indices in groups.values():
            weights = np.asarray([queries[i]["quality"] for i in indices])
            group = 0.75 * np.average(scores[indices], axis=0, weights=weights) + 0.25 * scores[indices].max(axis=0)
            grouped.append(group)
        return np.mean(grouped, axis=0)

    def rank(self, frame):
        queries = build_queries(frame, self.cache)
        cfg = self.config["retrieval"]
        mass = weighted_median([q["mass"] for q in queries], [q["quality"] for q in queries])
        ids = self.pool.window(mass, cfg["mass_ppm"], cfg["mass_da"])
        fallback = "none"
        if not len(ids):
            ids = self.pool.window(mass, cfg["fallback_ppm"], cfg["fallback_da"])
            fallback = "widened"
        if not len(ids):
            # Nonempty submission only. These candidates are explicitly flagged as outside the mass window.
            ids = self.pool.nearest(mass, cfg["top_k"])
            fallback = "nearest_outside_window"
        original_count = len(ids)
        if len(ids) > cfg["max_candidates"]:
            ids = ids[np.argsort(np.abs(self.pool.masses[ids]-mass), kind="stable")[:cfg["max_candidates"]]]
        lib_score = self.spectral_score(ids, queries)
        fp_score = np.zeros(len(ids), dtype=np.float32)
        if self.predictor is not None:
            probabilities = self.predictor.predict(queries)
            fp = self.pool.fingerprints(ids)
            intersection = fp @ probabilities
            fp_score = intersection / np.maximum(fp.sum(axis=1) + probabilities.sum() - intersection, 1e-8)
            alpha = cfg["fp_weight_strong"] if lib_score.max(initial=0) >= 0.60 else cfg["fp_weight_weak"]
            score = (1-alpha)*lib_score + alpha*fp_score
        else:
            score = lib_score.copy()
        ppm = np.abs(self.pool.masses[ids]-mass)/mass*1e6
        mass_prior = np.exp(-0.5*(ppm/10.0)**2)
        score += 0.02*mass_prior
        # Stable ties are resolved by mass agreement and then canonical identity.
        records = self.pool.table.iloc[ids].copy()
        records["score"], records["library_score"], records["fingerprint_score"] = score, lib_score, fp_score
        records["error_ppm"] = ppm
        ranked = records.sort_values(["score", "error_ppm", "key"], ascending=[False, True, True], kind="stable").head(cfg["top_k"])
        dispersion = max(abs(q["mass"]-mass)/mass*1e6 for q in queries)
        diagnostic = {"neutral_mass": mass, "mass_dispersion_ppm": dispersion, "spectra": len(queries),
                      "candidate_count": original_count, "candidate_count_scored": len(ids),
                      "candidate_truncated": original_count > len(ids), "mass_fallback": fallback,
                      "best_library_score": float(lib_score.max(initial=0)),
                      "candidate_keys": self.pool.table.iloc[ids]["key"].tolist()}
        return ranked, diagnostic


def predict(config, test_path=None, sample_path=None, output="submission.csv", checkpoint=None,
            device="auto", allow_mismatch=False, blocked_sids=None, blocked_rows=None,
            require_full_cache=True):
    started = time.monotonic()
    version = check_rdkit(allow_mismatch)
    work = Path(config["work_dir"])
    cache = ReferenceCache(work / "cache")
    cache.require_config(config)
    if version != cache.manifest["rdkit"]:
        raise ValueError("Runtime RDKit differs from the reference cache")
    if require_full_cache and cache.manifest["partial_cache"]:
        raise ValueError("A --max-rows cache is a development cache; rebuild the full training cache before submission")
    pool = CandidatePool(cache, work / "candidates")
    frame = pd.read_parquet(test_path or Path(config["data_dir"]) / "test.parquet")
    if frame.empty or "molecule_id" not in frame or frame["molecule_id"].isna().any():
        raise ValueError("Empty test table or invalid molecule_id")
    frame["molecule_id"] = frame["molecule_id"].astype(str)
    expected = frame["molecule_id"].drop_duplicates().tolist()
    if sample_path:
        sample = pd.read_csv(sample_path, dtype=str)
        if "molecule_id" not in sample or sample["molecule_id"].isna().any() or sample["molecule_id"].duplicated().any():
            raise ValueError("Invalid sample molecule IDs")
        if set(sample["molecule_id"]) == set(expected):
            expected = sample["molecule_id"].tolist()
        else:
            print("Sample molecule IDs differ from the current test; deriving output IDs from the current test file.")
    ranker = Ranker(config, cache, pool, checkpoint, device, blocked_sids, blocked_rows)
    rows, details, diagnostics = [], [], []
    grouped = frame.groupby("molecule_id", sort=False)
    for molecule in tqdm(expected, desc="molecule inference"):
        ranked, diag = ranker.rank(grouped.get_group(molecule))
        rows.append({"molecule_id": molecule, "smiles": ";".join(ranked["smiles"])})
        diag["molecule_id"] = molecule
        diagnostics.append(diag)
        ranked.insert(0, "molecule_id", molecule)
        ranked.insert(1, "rank", np.arange(1, len(ranked)+1))
        details.append(ranked)
    submission = pd.DataFrame(rows)
    check_submission(submission, expected, allow_mismatch)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    pd.concat(details, ignore_index=True).to_csv(output.with_suffix(".candidates.csv"), index=False)
    summary = {"seconds": time.monotonic()-started, "molecules": len(rows), "spectra": len(frame),
               "rdkit": version, "checkpoint": str(checkpoint) if checkpoint else None,
               "fingerprint_training": ranker.predictor.training_info if ranker.predictor else None,
               "queries": diagnostics}
    write_json(output.with_suffix(".diagnostics.json"), summary)
    print(f"Saved {output}; {len(rows)} molecules; {summary['seconds']:.1f} seconds")
    return submission, summary
