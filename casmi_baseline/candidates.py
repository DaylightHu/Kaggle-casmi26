from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .chemistry import describe_structure, check_rdkit
from .common import read_json, write_json


def inspect_candidates(path):
    path = Path(path)
    paths = sorted(path.rglob("*")) if path.is_dir() else [path]
    output = []
    for item in paths:
        if not item.is_file():
            continue
        record = {"file": str(item), "bytes": item.stat().st_size}
        try:
            if item.suffix == ".parquet":
                f = pq.ParquetFile(item)
                record.update(rows=f.metadata.num_rows, columns=f.schema_arrow.names)
            elif item.suffix in (".csv", ".tsv"):
                frame = pd.read_csv(item, sep="\t" if item.suffix == ".tsv" else ",", nrows=2)
                record.update(columns=frame.columns.tolist())
            elif item.suffix == ".npy":
                array = np.load(item, mmap_mode="r", allow_pickle=False)
                record.update(shape=list(array.shape), dtype=str(array.dtype))
        except Exception as e:
            record["inspection_error"] = str(e)
        output.append(record)
    print(__import__("json").dumps(output, ensure_ascii=False, indent=2))
    return output


def iter_smiles(path, smiles_column):
    path = Path(path)
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        if smiles_column not in pf.schema_arrow.names:
            raise ValueError(f"No column {smiles_column}. Available: {pf.schema_arrow.names}")
        for batch in pf.iter_batches(batch_size=4096, columns=[smiles_column]):
            yield from batch.column(0).to_pylist()
    elif path.suffix in (".csv", ".tsv"):
        for frame in pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", usecols=[smiles_column],
                                 dtype={smiles_column: str}, chunksize=4096):
            yield from frame[smiles_column]
    elif path.suffix in (".smi", ".smiles", ".txt"):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    yield line.split()[0]
    else:
        raise ValueError("Candidate input must be CSV, TSV, Parquet, or a line-based SMILES file")


def import_candidates(config, path, smiles_column="smiles", name="external", allow_mismatch=False):
    version = check_rdkit(allow_mismatch)
    from .cache import ReferenceCache
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    cache.require_config(config)
    if version != cache.manifest["rdkit"]:
        raise ValueError("Candidate import and cache must use the same RDKit version")
    dest = Path(config["work_dir"]) / "candidates"
    dest.mkdir(parents=True, exist_ok=True)
    records, fps, seen = [], [], set(cache.structures["key"])
    existing = dest / "structures.parquet"
    if existing.exists():
        manifest = read_json(dest / "manifest.json")
        if manifest["fingerprint"] != config["fingerprint"] or manifest["rdkit"] != version:
            raise ValueError("Existing external catalog was created with another chemistry configuration")
        records = pd.read_parquet(existing).to_dict("records")
        fps = list(np.load(dest / "fingerprints.npy"))
        seen.update(r["key"] for r in records)
    added = invalid = duplicate = 0
    for smiles in iter_smiles(path, smiles_column):
        if not isinstance(smiles, str) or not smiles.strip():
            invalid += 1
            continue
        try:
            record, fp = describe_structure(smiles, **config["fingerprint"])
        except (ValueError, RuntimeError, KeyError):
            invalid += 1
            continue
        if record["key"] in seen:
            duplicate += 1
            continue
        seen.add(record["key"])
        record.update(sid=-1, count=0, np_count=0, source=name)
        records.append(record)
        fps.append(fp)
        added += 1
    if not records:
        raise ValueError("No new valid structures; catalog was not created")
    table = pd.DataFrame(records)
    table.to_parquet(dest / "structures.parquet", index=False)
    np.save(dest / "fingerprints.npy", np.stack(fps))
    write_json(dest / "manifest.json", {"format_version": 1, "rdkit": version,
               "fingerprint": config["fingerprint"], "structures": len(records),
               "last_import": str(Path(path).resolve()), "added": added, "invalid": invalid, "duplicates": duplicate})
    print(f"External candidates: {len(records)}; added={added}, invalid={invalid}, duplicate={duplicate}")


class CandidatePool:
    def __init__(self, cache, external_path=None):
        tables = [cache.structures]
        self.fps = [cache.fps]
        self.train_size = len(cache.structures)
        if external_path and (Path(external_path) / "structures.parquet").exists():
            manifest = read_json(Path(external_path) / "manifest.json")
            if manifest["fingerprint"] != cache.manifest["fingerprint"] or manifest["rdkit"] != cache.manifest["rdkit"]:
                raise ValueError("External candidate chemistry does not match the reference cache")
            external = pd.read_parquet(Path(external_path) / "structures.parquet")
            raw_fps = np.load(Path(external_path) / "fingerprints.npy", mmap_mode="r")
            keep = ~external["key"].isin(cache.structures["key"])
            tables.append(external.loc[keep])
            self.fps.append(raw_fps[np.flatnonzero(keep.to_numpy())])
        self.table = pd.concat(tables, ignore_index=True)
        self.masses = self.table["mass"].to_numpy(dtype=np.float64)
        self.mass_order = np.argsort(self.masses, kind="stable")
        self.sorted_mass = self.masses[self.mass_order]

    def window(self, mass, ppm, absolute):
        delta = max(mass * ppm * 1e-6, absolute)
        left = np.searchsorted(self.sorted_mass, mass-delta, side="left")
        right = np.searchsorted(self.sorted_mass, mass+delta, side="right")
        return self.mass_order[left:right]

    def nearest(self, mass, count=25):
        pos = np.searchsorted(self.sorted_mass, mass)
        ids = self.mass_order[max(0, pos-count):min(len(self.mass_order), pos+count)]
        return ids[np.argsort(np.abs(self.masses[ids]-mass), kind="stable")[:count]]

    def fingerprints(self, ids):
        ids = np.asarray(ids)
        packed = np.empty((len(ids), self.fps[0].shape[1]), dtype=np.uint8)
        mask = ids < self.train_size
        packed[mask] = self.fps[0][ids[mask]]
        if (~mask).any():
            packed[~mask] = self.fps[1][ids[~mask]-self.train_size]
        return np.unpackbits(packed, axis=1, bitorder="little").astype(np.float32)
