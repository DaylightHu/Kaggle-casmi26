from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cache import ReferenceCache
from .common import read_json, stable_fraction, write_json

QUERY_COLUMNS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
                 "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
                 "precursor_mz", "collision_energy_ev", "collision_energy_orig", "collision_energy_orig_units"]


def make_split(config, kind="identity", fraction=0.1, molecules=400, spectra_per_molecule=8, train_path=None):
    if not 0 < fraction < 1 or molecules < 1 or spectra_per_molecule < 1:
        raise ValueError("Invalid split size")
    work = Path(config["work_dir"])
    cache = ReferenceCache(work / "cache")
    table, meta = cache.structures, cache.meta
    if kind == "identity":
        held = [i for i, key in enumerate(table["key"]) if stable_fraction(key, config["seed"]) < fraction]
    elif kind == "np-panel":
        held = np.flatnonzero(table["np_count"].to_numpy() > 0).tolist()
    else:
        raise ValueError("Unknown split kind")
    # Keep every held identity out of fingerprint fitting, including other libraries.
    candidates = [i for i in held if len(cache.rows_for_sid(i))]
    if not candidates or len(held) == len(table):
        raise ValueError("Split has no usable validation molecules or no training molecules")
    rng = np.random.default_rng(config["seed"])
    selected = sorted(rng.choice(candidates, min(molecules, len(candidates)), replace=False).tolist())
    np_lib = cache.manifest["libraries"].index("enveda-np-examples") if "enveda-np-examples" in cache.manifest["libraries"] else -1
    query_rows = []
    for sid in selected:
        rows = np.asarray(cache.rows_for_sid(sid))
        if kind == "np-panel":
            rows = rows[meta["lib"][rows] == np_lib]
        priority = meta["timstof"][rows].astype(float) * 10 + np.log1p(meta["n"][rows])
        order = rows[np.argsort(-priority, kind="stable")]
        # Prefer distinct adduct / collision-energy views before repeated acquisitions.
        seen, diverse, extra = set(), [], []
        for row in order:
            energy = float(meta["ce"][row])
            tag = (int(meta["adduct"][row]), round(energy / 5) if np.isfinite(energy) else -1)
            if tag in seen:
                extra.append(int(row))
            else:
                diverse.append(int(row))
                seen.add(tag)
        query_rows.extend((diverse + extra)[:spectra_per_molecule])
    if not query_rows:
        raise ValueError("No query spectra for this split")
    query_rows = sorted(query_rows)
    # In the class-1 panel, withhold every NP-panel acquisition, not only the selected queries.
    if kind == "np-panel":
        blocked = np.flatnonzero(meta["lib"] == np_lib).tolist()
    else:
        blocked = query_rows
    source = Path(train_path or config["data_dir"] + "/train.parquet")
    pf = pq.ParquetFile(source)
    if source.stat().st_size != cache.manifest["train_size"] or pf.metadata.num_rows != cache.manifest["total_source_rows"]:
        raise ValueError("Validation source does not match the cache's training file")
    parts, cursor = [], 0
    requested = np.asarray(query_rows)
    columns = [c for c in QUERY_COLUMNS if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(batch_size=4096, columns=columns):
        positions = requested[(requested >= cursor) & (requested < cursor + len(batch))] - cursor
        if len(positions):
            frame = batch.to_pandas().iloc[positions].copy()
            sids = meta["sid"][cursor+positions]
            frame["molecule_id"] = ["val_" + table.iloc[int(sid)]["key"] for sid in sids]
            parts.append(frame)
        cursor += len(batch)
        if cursor > requested[-1]:
            break
    queries = pd.concat(parts, ignore_index=True)
    present = [int(i) for i in selected if "val_" + table.iloc[i]["key"] in set(queries["molecule_id"])]
    truth = pd.DataFrame({"molecule_id": ["val_" + table.iloc[i]["key"] for i in present],
                          "smiles": [table.iloc[i]["smiles"] for i in present]})
    dest = work / "validation"
    dest.mkdir(parents=True, exist_ok=True)
    queries.to_parquet(dest / "queries.parquet", index=False)
    truth.to_csv(dest / "truth.csv", index=False)
    write_json(dest / "split.json", {"kind": kind, "seed": config["seed"], "fraction": fraction,
               "cache_rows": len(meta), "cache_structures": len(table), "held_sids": held,
               "query_rows": query_rows, "blocked_rows": blocked, "selected_sids": present,
               "held_keys": table.iloc[held]["key"].tolist(),
               "queries": str((dest / "queries.parquet").resolve()), "truth": str((dest / "truth.csv").resolve())})
    print(f"Split: {kind}, {len(held)} held identities, {len(truth)} queries, {len(queries)} spectra")
    return dest / "split.json"


def load_split(path, cache):
    split = read_json(path)
    if split["cache_rows"] != len(cache.meta) or split["cache_structures"] != len(cache.structures):
        raise ValueError("Split and cache differ")
    keys = cache.structures.iloc[split["held_sids"]]["key"].tolist()
    if keys != split["held_keys"]:
        raise ValueError("Split identities differ from cache")
    return split
