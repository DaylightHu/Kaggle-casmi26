"""Deterministic tests for the independent GLACIER re-ranker."""

import numpy as np
import pandas as pd
import pytest

from casmi_baseline import glacier


def test_entropy_similarity_identity_and_disjoint():
    mz = np.array([50., 75., 100.])
    weights = np.sqrt(np.array([.2, .3, .5]))
    weights /= np.linalg.norm(weights)
    assert glacier.entropy_similarity(mz, weights, mz, weights) == pytest.approx(1.)
    assert glacier.entropy_similarity(mz, weights, mz + 10, weights) == 0.


def test_prediction_postprocess_merges_by_max():
    mz, weights = glacier.postprocess_prediction(
        [10.00001, 10.00002, 20., 30.], [.3, .8, 1., -1.], top_n=100)
    assert mz.tolist() == pytest.approx([10., 20.])
    probabilities = weights.astype(float) ** 2
    probabilities /= probabilities.sum()
    assert probabilities.tolist() == pytest.approx([.8 / 1.8, 1. / 1.8])


def test_formula_slot_margin_fusion_preserves_formula_slots():
    frame = pd.DataFrame({
        'key': ['a', 'b', 'c', 'd', 'e'],
        'smiles': ['CCO', 'COC', 'CCC', 'CCN', 'CNC'],
        'formula': ['C2H6O', 'C2H6O', 'C3H8', '', ''],
        'score': [.50, .48, .60, .40, .30],
        'glacier_raw': [.10, .40, .90, .01, .99],
        'error_ppm': [0., 0., 0., 0., 0.],
    })
    result = glacier.formula_slot_rerank(frame, .75)
    assert result['key'].tolist() == ['b', 'a', 'c', 'd', 'e']
    assert result['formula'].tolist() == frame['formula'].tolist()
    # Blank/unknown formulae are never treated as one shared formula group.
    assert result.iloc[3:]['key'].tolist() == ['d', 'e']
    assert result.loc[0, 'glacier_confidence'] == pytest.approx(1.)


def test_formula_slot_weak_evidence_is_exact_base_fallback():
    frame = pd.DataFrame({
        'key': ['a', 'b'], 'smiles': ['CCO', 'COC'],
        'formula': ['C2H6O', 'C2H6O'], 'score': [.50, .49],
        'glacier_raw': [.1000, .1001], 'error_ppm': [0., 0.],
    })
    result = glacier.formula_slot_rerank(frame, .75)
    assert result['key'].tolist() == ['a', 'b']
    assert result['score'].tolist() == frame['score'].tolist()


def test_job_builder_deduplicates_forward_predictions(monkeypatch):
    frame = pd.DataFrame([
        {'molecule_id': 'm1', 'adduct': '[M+H]+', 'ionization_mode': 'positive',
         'collision_energy_ev': [20.], 'instrument_type': 'Orbitrap'},
        {'molecule_id': 'm1', 'adduct': '[M+H]+', 'ionization_mode': 'positive',
         'collision_energy_ev': [21.], 'instrument_type': 'Orbitrap'},
    ])
    queries = [
        {'mz': np.array([50.], np.float32), 'weights': np.array([1.], np.float32),
         'quality': .5},
        {'mz': np.array([60.], np.float32), 'weights': np.array([1.], np.float32),
         'quality': .8},
    ]
    monkeypatch.setattr(glacier, 'build_queries', lambda group, cache: queries)
    candidates = {'m1': pd.DataFrame([
        {'key': 'a', 'smiles': 'CCO'},
        # Disconnected structures are outside GLACIER's molecular-graph domain.
        {'key': 'salt', 'smiles': 'CC.[Na+]'},
    ])}
    jobs, observed, mapping = glacier.build_jobs(frame, candidates, object())
    assert len(observed) == 2
    assert len(jobs) == 1
    assert jobs[0]['queries'] == [0, 1]
    assert len(mapping) == 2


def test_quality_weighted_aggregation_then_equal_adduct_weight():
    records = [
        {'adduct': '[M+H]+', 'mode': 'positive', 'score': .2, 'quality': 1.},
        {'adduct': '[M+H]+', 'mode': 'positive', 'score': .8, 'quality': 3.},
        {'adduct': '[M+Na]+', 'mode': 'positive', 'score': .4, 'quality': 1.},
    ]
    assert glacier.aggregate_job_scores(records) == pytest.approx((.65 + .4) / 2)
