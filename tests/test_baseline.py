"""Synthetic execution tests only. No training and no competition score claims."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from casmi_baseline.cache import ReferenceCache, prepare_cache
from casmi_baseline.candidates import CandidatePool, import_candidates
from casmi_baseline.chemistry import TEST_ADDUCTS, describe_structure, ion_mz, neutral_mass, scorer_key
from casmi_baseline.common import load_config, read_json
from casmi_baseline.metric import check_submission, evaluate_submission
from casmi_baseline.retrieval import Ranker, predict
from casmi_baseline.spectra import clean_spectrum, cosine_scan
from casmi_baseline.validation import make_split, load_split


@pytest.mark.parametrize("adduct", TEST_ADDUCTS + ["[M+2H]2+", "[2M+H]+", "[M+ACN+H]+", "[M]+"])
def test_adduct_roundtrip(adduct):
    assert neutral_mass(ion_mz(348.123456, adduct), adduct) == pytest.approx(348.123456, abs=1e-10)


def test_mass_constants_and_unknown_adduct():
    assert ion_mz(100, "[M+H]+") == pytest.approx(101.0072764523, abs=1e-9)
    assert ion_mz(100, "[M+CH2O2-H]-") == pytest.approx(144.9982028513, abs=1e-9)
    with pytest.raises(ValueError):
        neutral_mass(500, "unrecognised")


def test_metric_identity():
    assert scorer_key("C[C@H](O)C(=O)O") == scorer_key("C[C@@H](O)C(=O)O")
    assert scorer_key("Oc1ccccn1") == scorer_key("O=c1cccc[nH]1")
    assert scorer_key("CCCO") != scorer_key("CCOC")
    with pytest.raises(ValueError):
        scorer_key("not_a_smiles")


def test_cleaning():
    cfg = load_config()["preprocess"]
    mz, w = clean_spectrum([20, 20, 30, 101, np.nan, 15], [1, .25, .01, 1, 1, -1], 100, cfg)
    assert mz.tolist() == [20, 30]
    assert np.linalg.norm(w) == pytest.approx(1)
    assert cosine_scan(mz, w, mz, w, .02, 10) == pytest.approx(1, abs=1e-6)
    with pytest.raises(ValueError):
        clean_spectrum([10, 20], [1], 100, cfg)


@pytest.fixture
def tiny_project(tmp_path):
    config = load_config(data_dir=tmp_path / "data", work_dir=tmp_path / "work")
    data = Path(config["data_dir"])
    data.mkdir()
    labels = ["CCCO", "CCOC", "CCCCO", "CC(=O)O", "c1ccccc1", "CCN", "CC(=O)N", "CC(C)O"]
    rows = []
    for sid, smiles in enumerate(labels):
        description, _ = describe_structure(smiles)
        for library in ["enveda-np-examples", "massbank"]:
            for energy in [20, 40]:
                peaks = [10+sid*.2, 15+sid*.2, 20+sid*.2, 25+sid*.2, 30+sid*.2, 35+sid*.2]
                rows.append({"molecule_id": f"m_{sid}", "spectrum_id": f"s_{len(rows)}",
                    "normalized_smiles": smiles, "ms2_mzs": peaks,
                    "ms2_normalized_intensities": [1, .8, .7, .5, .3, .1],
                    "base_peak_intensity": 20000.0, "adduct": "[M+H]+", "ionization_mode": "positive",
                    "instrument_type": "timsTOF" if library.startswith("enveda") else "Orbitrap",
                    "precursor_mz": ion_mz(description["mass"], "[M+H]+"), "collision_energy_ev": [float(energy)],
                    "ingest_lib": library, "precursor_error_ppm": 0.0})
    train = pd.DataFrame(rows)
    train.to_parquet(data / "train.parquet", index=False)
    test = train[train["ingest_lib"] == "enveda-np-examples"].drop(columns=["normalized_smiles", "ingest_lib", "precursor_error_ppm"])
    test.to_parquet(data / "test.parquet", index=False)
    pd.DataFrame({"molecule_id": [f"m_{i}" for i in reversed(range(len(labels)))], "smiles": "CCO"}).to_csv(data / "sample_submission.csv", index=False)
    prepare_cache(config)
    return config, labels, train


def test_end_to_end_retrieval_and_live_ids(tiny_project):
    config, labels, train = tiny_project
    data = Path(config["data_dir"])
    output = Path(config["work_dir"]) / "submission.csv"
    sub, diagnostic = predict(config, sample_path=data / "sample_submission.csv", output=output)
    assert sub["molecule_id"].tolist() == [f"m_{i}" for i in reversed(range(len(labels)))]
    truth = pd.DataFrame({"molecule_id": [f"m_{i}" for i in range(len(labels))], "smiles": labels})
    assert evaluate_submission(sub, truth)["top1"] == 1
    assert len(diagnostic["queries"]) == len(labels)
    # Visible IDs in sample_submission may be stale during a hidden re-run.
    changed = pd.read_parquet(data / "test.parquet")
    changed["molecule_id"] = "hidden_" + changed["molecule_id"]
    changed.to_parquet(data / "hidden.parquet", index=False)
    result, _ = predict(config, test_path=data / "hidden.parquet", sample_path=data / "sample_submission.csv", output=output)
    assert all(s.startswith("hidden_") for s in result["molecule_id"])


def test_split_blocks_whole_identities(tiny_project):
    config, _, _ = tiny_project
    path = make_split(config, fraction=.5, molecules=3)
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    split = load_split(path, cache)
    assert set(cache.meta["sid"][split["query_rows"]]).issubset(set(split["held_sids"]))
    pool = CandidatePool(cache)
    ranker = Ranker(config, cache, pool, blocked_sids=split["held_sids"])
    frame = pd.read_parquet(Path(path).parent / "queries.parquet")
    first = frame.groupby("molecule_id").get_group(frame.iloc[0]["molecule_id"])
    _, diagnostics = ranker.rank(first)
    assert diagnostics["best_library_score"] < 1
    from casmi_baseline.retrieval import build_queries
    for sid in split["held_sids"]:
        assert len(ranker.choose_references(sid, build_queries(first, cache)[0])) == 0


def test_np_panel_withholds_all_panel_rows(tiny_project):
    config, _, _ = tiny_project
    # Smaller identity panel for this test, so at least one training identity remains.
    cache_path = Path(config["work_dir"]) / "cache/structures.parquet"
    table = pd.read_parquet(cache_path)
    table.loc[table["sid"] >= 3, "np_count"] = 0
    table.to_parquet(cache_path, index=False)
    path = make_split(config, kind="np-panel", molecules=2, spectra_per_molecule=1)
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    split = load_split(path, cache)
    np_code = cache.manifest["libraries"].index("enveda-np-examples")
    assert set(split["blocked_rows"]) == set(np.flatnonzero(cache.meta["lib"] == np_code))
    assert len(split["query_rows"]) == 2


def test_external_candidates_deduplicate(tiny_project, tmp_path):
    config, _, _ = tiny_project
    candidate = tmp_path / "extra.csv"
    pd.DataFrame({"canonical_smiles": ["CCCO", "CCO", "OCC", "bad"]}).to_csv(candidate, index=False)
    import_candidates(config, candidate, "canonical_smiles")
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    pool = CandidatePool(cache, Path(config["work_dir"]) / "candidates")
    assert len(pool.table) == len(cache.structures)+1
    assert pool.fingerprints([0, len(cache.structures)]).shape == (2, 2048)


def test_partial_cache_cannot_submit(tiny_project, tmp_path):
    config, _, _ = tiny_project
    small = copy.deepcopy(config)
    small["work_dir"] = str(tmp_path / "small")
    prepare_cache(small, max_rows=3)
    with pytest.raises(ValueError, match="development cache"):
        predict(small)


@pytest.mark.parametrize("row", [
    {"molecule_id": "x", "smiles": ""},
    {"molecule_id": "x", "smiles": ";".join(["CCO"]*26)},
    {"molecule_id": "x", "smiles": "CCO;OCC"},
    {"molecule_id": "x", "smiles": "invalid"},
    {"molecule_id": "other", "smiles": "CCO"},
])
def test_reject_bad_submissions(row):
    with pytest.raises(ValueError):
        check_submission(pd.DataFrame([row]), ["x"])


def test_mrr_formula():
    truth = pd.DataFrame({"molecule_id": ["a", "b", "c"], "smiles": ["CCO", "CCO", "CCO"]})
    sub = pd.DataFrame({"molecule_id": ["a", "b", "c"], "smiles": ["CCO", "CCN;CCC;CC(=O)O;CCO", "CCC"]})
    assert evaluate_submission(sub, truth)["mrr_at_25"] == pytest.approx((1+.25)/3)


def test_fingerprint_forward_no_training(tiny_project, tmp_path):
    torch = pytest.importorskip("torch")
    from casmi_baseline.fingerprint import SpectrumFingerprintNet, FingerprintPredictor, collate, tokenize, query_from_cache
    config, _, _ = tiny_project
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    model_config = {"bin_width": .1, "max_mz": 1500, "embedding_dim": 32, "hidden_dim": 64,
                    "dropout": 0, "bits": 2048, "num_adducts": len(cache.manifest["adducts"])}
    queries = [query_from_cache(cache, 0), query_from_cache(cache, 1)]
    model = SpectrumFingerprintNet(model_config).eval()
    with torch.inference_mode():
        assert model(collate([tokenize(q, model_config) for q in queries])).shape == (2, 2048)
    checkpoint = tmp_path / "untrained.pt"
    torch.save({"format_version": 1, "model": model.state_dict(), "model_config": model_config,
                "adducts": cache.manifest["adducts"], "fingerprint": config["fingerprint"],
                "preprocess": config["preprocess"], "rdkit": cache.manifest["rdkit"],
                "training_info": {"full": False, "held_keys": [], "partial_cache": False}}, checkpoint)
    predictor = FingerprintPredictor(checkpoint, cache, "cpu")
    p = predictor.predict(queries)
    assert p.shape == (2048,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()


def test_exported_offline_entrypoint(tiny_project, tmp_path):
    from casmi_baseline.cli import export_assets
    config, labels, _ = tiny_project
    assets = tmp_path / "inference-assets"
    export_assets(config, assets)
    output = tmp_path / "offline_submission.csv"
    env = os.environ.copy()
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    result = subprocess.run([sys.executable, str(assets / "infer.py"), "--assets", str(assets),
                            "--data-dir", config["data_dir"], "--output", str(output)],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    frame = check_submission(output, [f"m_{i}" for i in range(len(labels))])
    assert len(frame) == len(labels)


def test_validation_refuses_leaky_model(tiny_project, tmp_path):
    torch = pytest.importorskip("torch")
    from casmi_baseline.cli import main
    config, _, _ = tiny_project
    make_split(config, fraction=.5, molecules=2)
    checkpoint = tmp_path / "leaky.pt"
    torch.save({"format_version": 1, "training_info": {"full": True, "held_keys": []}}, checkpoint)
    with pytest.raises(ValueError, match="seen validation identities"):
        main(["--data-dir", config["data_dir"], "--work-dir", config["work_dir"],
              "validate", "--checkpoint", str(checkpoint)])
