"""Disk-backed reference library. No whole-corpus peak dataframe is loaded."""
import collections
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .chemistry import TEST_ADDUCTS, check_rdkit, describe_structure
from .common import read_json, write_json
from .spectra import clean_spectrum, mean_energy, mode_code, timstof

META_DTYPE = np.dtype([("sid", "<i4"), ("precursor", "<f8"), ("adduct", "<i2"),
                       ("mode", "i1"), ("ce", "<f4"), ("timstof", "?"),
                       ("lib", "<i2"), ("n", "<i2"), ("error_ppm", "<f4")])
TRAIN_COLUMNS = ["normalized_smiles", "precursor_mz", "ms2_mzs",
                 "ms2_normalized_intensities", "adduct", "ionization_mode",
                 "collision_energy_ev", "instrument_type", "ingest_lib", "precursor_error_ppm"]


def prepare_cache(config, train_path=None, allow_mismatch=False, max_rows=None):
    version = check_rdkit(allow_mismatch)
    path = Path(train_path or Path(config["data_dir"]) / "train.parquet")
    dest = Path(config["work_dir"]) / "cache"
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Cache directory is not empty: {dest}. Choose another --work-dir to rebuild.")
    dest.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(path)
    missing = set(TRAIN_COLUMNS[:6]) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")
    rows = min(max_rows or pf.metadata.num_rows, pf.metadata.num_rows)
    top_n = int(config["preprocess"]["top_n"])
    if rows < 1 or not 1 <= top_n <= 32767:
        raise ValueError("Empty training file or invalid top_n")
    bits = int(config["fingerprint"]["bits"])
    if bits % 8:
        raise ValueError("Fingerprint bits must be divisible by 8")
    mzout = np.lib.format.open_memmap(dest / "mzs.npy", mode="w+", dtype="float32", shape=(rows, top_n))
    wout = np.lib.format.open_memmap(dest / "weights.npy", mode="w+", dtype="float16", shape=(rows, top_n))
    meta = np.lib.format.open_memmap(dest / "meta.npy", mode="w+", dtype=META_DTYPE, shape=(rows,))
    adducts = ["unknown"] + TEST_ADDUCTS.copy()
    libraries = ["unknown"]
    adduct_map = {v: i for i, v in enumerate(adducts)}
    library_map = {"unknown": 0}
    structures, fps, by_key, by_smiles = [], [], {}, {}
    audit = collections.Counter()
    columns = [c for c in TRAIN_COLUMNS if c in pf.schema_arrow.names]
    cursor = 0
    for batch in tqdm(pf.iter_batches(batch_size=2048, columns=columns), total=(rows+2047)//2048, desc="prepare"):
        values = batch.to_pydict()
        length = min(len(batch), rows-cursor)
        mzout[cursor:cursor+length] = 0
        wout[cursor:cursor+length] = 0
        meta[cursor:cursor+length] = np.zeros(length, dtype=META_DTYPE)
        meta["sid"][cursor:cursor+length] = -1
        meta["ce"][cursor:cursor+length] = np.nan
        meta["error_ppm"][cursor:cursor+length] = np.nan
        for j in range(length):
            i = cursor+j
            label = values["normalized_smiles"][j]
            if not isinstance(label, str) or not label:
                audit["invalid_label"] += 1
                continue
            if label not in by_smiles:
                try:
                    record, fp = describe_structure(label, **config["fingerprint"])
                    key = record["key"]
                    if key not in by_key:
                        sid = len(structures)
                        by_key[key] = sid
                        record.update(sid=sid, count=0, np_count=0, source="train")
                        structures.append(record)
                        fps.append(fp)
                    by_smiles[label] = by_key[key]
                except (ValueError, RuntimeError, KeyError):
                    by_smiles[label] = -1
            sid = by_smiles[label]
            if sid < 0:
                audit["invalid_label"] += 1
                continue
            get = lambda column, default=None: values[column][j] if column in values else default
            try:
                precursor = float(get("precursor_mz"))
                if not np.isfinite(precursor) or precursor <= 0:
                    raise ValueError("Invalid precursor")
                mz, weights = clean_spectrum(get("ms2_mzs"), get("ms2_normalized_intensities"), precursor, config["preprocess"])
            except (ValueError, TypeError):
                audit["invalid_spectrum"] += 1
                continue
            adduct = str(get("adduct") or "unknown")
            library = str(get("ingest_lib") or "unknown")
            for value, mapping, labels in [(adduct, adduct_map, adducts), (library, library_map, libraries)]:
                if value not in mapping:
                    mapping[value] = len(labels)
                    labels.append(value)
            if len(adducts) > 32767 or len(libraries) > 32767:
                raise ValueError("Too many distinct metadata labels")
            error = get("precursor_error_ppm")
            try:
                error = float(error) if error is not None else np.nan
            except (TypeError, ValueError):
                error = np.nan
            mzout[i, :len(mz)] = mz
            wout[i, :len(weights)] = weights
            meta[i] = (sid, precursor, adduct_map[adduct], mode_code(get("ionization_mode")),
                       mean_energy(get("collision_energy_ev")), timstof(get("instrument_type"), library),
                       library_map[library], len(mz), error)
            structures[sid]["count"] += 1
            structures[sid]["np_count"] += int(library == "enveda-np-examples")
            audit["valid_spectra"] += 1
            if not len(mz):
                audit["empty_after_cleaning"] += 1
        cursor += length
        if cursor == rows:
            break
    if not structures:
        raise ValueError("No valid structures found")
    for array in (mzout, wout, meta):
        array.flush()
    table = pd.DataFrame(structures)
    table.to_parquet(dest / "structures.parquet", index=False)
    np.save(dest / "fingerprints.npy", np.stack(fps))
    valid = (meta["sid"] >= 0) & (meta["n"] > 0)
    valid &= ~np.isfinite(meta["error_ppm"]) | (np.abs(meta["error_ppm"]) <= config["preprocess"]["max_precursor_error_ppm"])
    valid_rows = np.flatnonzero(valid)
    ordered = valid_rows[np.argsort(meta["sid"][valid_rows], kind="stable")]
    counts = np.bincount(meta["sid"][ordered], minlength=len(structures))
    np.save(dest / "reference_rows.npy", ordered)
    np.save(dest / "reference_offsets.npy", np.r_[0, np.cumsum(counts)])
    np.save(dest / "mass_order.npy", np.argsort(table["mass"].to_numpy(), kind="stable"))
    write_json(dest / "manifest.json", {"format_version": 1, "rdkit": version,
               "train_path": str(path.resolve()), "train_size": path.stat().st_size,
               "total_source_rows": pf.metadata.num_rows, "rows": rows, "structures": len(structures),
               "partial_cache": rows != pf.metadata.num_rows, "adducts": adducts, "libraries": libraries,
               "preprocess": config["preprocess"], "fingerprint": config["fingerprint"],
               "reference_spectra": len(ordered), "audit": dict(audit)})
    print("Cache ready:", dest, dict(audit))
    return dest


class ReferenceCache:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = read_json(self.path / "manifest.json")
        if self.manifest["format_version"] != 1:
            raise ValueError("Unsupported cache format")
        self.structures = pd.read_parquet(self.path / "structures.parquet")
        self.mzs = np.load(self.path / "mzs.npy", mmap_mode="r")
        self.weights = np.load(self.path / "weights.npy", mmap_mode="r")
        self.meta = np.load(self.path / "meta.npy", mmap_mode="r")
        self.fps = np.load(self.path / "fingerprints.npy", mmap_mode="r")
        self.reference_rows = np.load(self.path / "reference_rows.npy", mmap_mode="r")
        self.offsets = np.load(self.path / "reference_offsets.npy", mmap_mode="r")
        self.adduct_map = {a: i for i, a in enumerate(self.manifest["adducts"])}

    def rows_for_sid(self, sid):
        return self.reference_rows[self.offsets[sid]:self.offsets[sid+1]]

    def require_config(self, config):
        for name in ("preprocess", "fingerprint"):
            if config[name] != self.manifest[name]:
                raise ValueError(f"{name} config differs from cache. Rebuild in a new work directory.")
