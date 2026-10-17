"""GLACIER forward-spectrum re-ranking for the CASMI candidate engine.

This module deliberately owns candidate selection, spectrum scoring, evidence
aggregation, and re-ranking.  The external bundle contributes only the official
MassSpecGym GLACIER weights, the pinned ``ms-pred`` source subset, and small
pure-PyTorch compatibility shims needed by an offline Kaggle runtime.

``predict`` first runs the existing DreaMS-head ranker and keeps its top-K
unique structures.  Eligible positive-mode spectra are then predicted in one
or more isolated GPU workers.  GLACIER evidence only reorders structures inside
the formula slots already selected by the base model by default, so an absent,
failed, or uninformative forward model is an exact base-ranking fallback.
"""

import argparse
import functools
import gc
import math
import os
import subprocess
import sys
import time
import types
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import diversity, dreams_head
from .common import ROOT, file_digest, load_config, read_json, write_json
from .chemistry import scorer_key
from .metric import check_submission
from .retrieval import build_queries
from .spectra import mean_energy


VERSION = 1
BUNDLE_NAME = 'casmi26-glacier'
CHECKPOINT_SHA256 = '9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12'
SOURCE_CHECKPOINT_SHA256 = '5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11'
MS_PRED_COMMIT = '708148c2a8eb0c32120644436fefd2fe90cd2214'
CHECKPOINT_TENSORS = 264
CHECKPOINT_PARAMETERS = 15_108_154
SUPPORTED_ADDUCTS = frozenset(('[M+H]+', '[M+Na]+'))
SUPPORTED_ATOMIC_NUMBERS = frozenset((1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17,
                                      19, 26, 27, 33, 34, 35, 53))
# Only files imported by this module are trusted.  gl_runner.py and gl_fuse.py
# are intentionally neither validated nor imported.
EXECUTED_FILE_SHA256 = {
    'shim/LinSATNet/__init__.py': 'a0ffdf64f610ad6a86ef560d3ff62f06a9c2b53ed13473d85eae87953f39e6af',
    'shim/dgl/__init__.py': '35dd2028bbb1fb14442d7a4cbbd9cb260c1063ab05dd556ddaf1f77aa2f3d2e2',
    'shim/dgl/backend/__init__.py': 'f85e782b33669b9567ba24c46931db87506637870d723c8696e66515ff0f9ee7',
    'shim/dgl/backend/pytorch.py': 'e7bf3d120f37ebc40fb8dec000f637ec516b75610b21717a4d7abe3cb40f37c7',
    'shim/dgl/function.py': '4dd78301a49129085e3234ff902999c7921979a5e09e5415a3aaf3a0aa9d27ff',
    'shim/dgl/nn/__init__.py': 'cd78a6aaee83a58cadb14a9a41f8c8e92a07c3c2bba52f6a83b4eb8ac4fd62a2',
    'shim/torch_scatter/__init__.py': 'd8f0b42adc06bb499514f68378c826900cbcb2e9764a4ea403bc160497c842a6',
    'src/ms_pred/__init__.py': 'e4a927730a084dd169e994b1fa712c5e17709e0a869f00aadb4709df9855c487',
    'src/ms_pred/common/__init__.py': '521b8cd399abda6a52cd1cc3304d46557035c70f3332be50122c9e68032d1351',
    'src/ms_pred/common/chem_utils.py': 'c9842ef76eb831394c53fdd5050d87842edaebd46427a722d43983286495f96e',
    'src/ms_pred/glacier/dataset.py': '547a09d8629a45c8fad43dcb60277a36906b1d382520c959950562d109014360',
    'src/ms_pred/glacier/joint_model.py': '9481602423011e830c94b8fcd857ac2be3cb31b2c24d714a3ae0c1edd235c293',
    'src/ms_pred/graphormer/__init__.py': 'b31b131343597bb736ee984291cbe0688383b507cd6b22d4d9ca01159ac440f2',
    'src/ms_pred/graphormer/graphormer_graph_encoder.py': '074a59296c0a2903fd5018c27705ea1da2393f09c84ba30fb32acbf3827729ea',
    'src/ms_pred/graphormer/graphormer_graph_encoder_layer.py': 'd98ba908616e231187afe2409409c06269e5d2017bd12c0d85c6ef093e8272f7',
    'src/ms_pred/graphormer/graphormer_layers.py': '26c8d693bebc8904a975aeb11cf343b9c7a0fd9a120f828ed597bd8180bc606d',
    'src/ms_pred/graphormer/multihead_attention.py': 'f11b79e3e4f8147cbbe97638e27b7046dfb8c2ed714b35409925f9bb4c5f0815',
    'src/ms_pred/iceberg/dag_data.py': '80a4f2abd57a59077bf988c9a3f975de36cf856c189aadf9f871382054494814',
    'src/ms_pred/iceberg/graph_mutate.py': '06c343312d44eedc1498c5513b33bcf5b1bc950f05dcb87e18827a0a7a81c55d',
    'src/ms_pred/magma/fragmentation.py': '5fc6ac347a09ea8486154ce827eea0ea810a0c7f6a277279ffd4c5abd1e93744',
    'src/ms_pred/nn_utils/__init__.py': '2f7f5b3d2537b6bde6a28be84908826901cb2306756b680a57934c39831ade57',
    'src/ms_pred/nn_utils/dgl_graph_ops.py': 'd65b239032eae4ec8cb2d0b9850fd773542722ff953c96827ab59000d13eba9f',
    'src/ms_pred/nn_utils/dgl_modules.py': 'b98dfc02dad06a67e2c9ee61670ebc0c96a8b2a62859335888f8189d5677e2e4',
    'src/ms_pred/nn_utils/form_embedder.py': '30bb3670c8b21e01ebb3fb2548dac7f8e0cac589fd4cfccc51589ddf55c1ea4b',
    'src/ms_pred/nn_utils/mol_graph.py': '62d11349b0c9d06335384b6fabfcddffaf78ece4c1d6f5c18ac426746d1e7d89',
    'src/ms_pred/nn_utils/nn_utils.py': 'f07a9018515ed0204b305741c5b13db9285d400ee7ff3e5bfa12db756a87146f',
    'src/ms_pred/nn_utils/transformer_layer.py': '8884d91afd479bef9b05d4a4527953ade8fd9353332ca3c6eddb8e1b048d2a92',
}


def validate_bundle(path):
    """Validate the pinned transport bundle without importing or executing it."""
    root = Path(path).resolve()
    manifest_path = root / 'MANIFEST.json'
    checkpoint = root / 'ckpt/glacier.pt'
    source = root / 'src/ms_pred'
    shim = root / 'shim'
    missing = [str(p) for p in (manifest_path, checkpoint, source, shim) if not p.exists()]
    if missing:
        raise FileNotFoundError('Incomplete GLACIER bundle: ' + ', '.join(missing))
    manifest = read_json(manifest_path)
    expected = {
        'name': BUNDLE_NAME,
        'ckpt_source_sha256': SOURCE_CHECKPOINT_SHA256,
        'ckpt_tensors': CHECKPOINT_TENSORS,
        'ckpt_params': CHECKPOINT_PARAMETERS,
        'ms_pred_commit': MS_PRED_COMMIT,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Unexpected GLACIER manifest {key}: {manifest.get(key)!r}')
    entry = manifest.get('files', {}).get('ckpt/glacier.pt', {})
    if entry.get('sha256') != CHECKPOINT_SHA256:
        raise ValueError('GLACIER manifest does not pin the expected inference checkpoint.')
    if checkpoint.stat().st_size != entry.get('bytes'):
        raise ValueError('GLACIER checkpoint size differs from its manifest.')
    if file_digest(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('GLACIER checkpoint SHA256 mismatch.')
    for relative, expected_hash in EXECUTED_FILE_SHA256.items():
        item = root / relative
        declared = manifest.get('files', {}).get(relative, {}).get('sha256')
        if not item.is_file() or declared != expected_hash or file_digest(item) != expected_hash:
            raise ValueError(f'GLACIER executable asset mismatch: {relative}')
    return root, checkpoint, manifest


def _standard_score(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values
    scale = float(values.std())
    return np.zeros(len(values), dtype=np.float64) if scale < 1e-12 else (values - values.mean()) / scale


def formula_slot_rerank(frame, weight):
    """Confidence-gated absolute-margin fusion inside existing formula slots.

    This deliberately does not use the bundle author's z-score fusion.  Absolute
    entropy similarity and separation determine how much the forward model may
    alter the base score.  Weak or nearly tied predictions therefore leave the
    DreaMS-head order unchanged.
    """
    result = frame.reset_index(drop=True).copy()
    result['glacier_fused'] = result['score'].astype(float)
    result['glacier_confidence'] = 0.
    if weight <= 0 or 'glacier_raw' not in result:
        return result
    formulas = result['formula'].fillna('').astype(str)
    for formula in formulas.drop_duplicates():
        if not formula:
            continue
        positions = np.flatnonzero(formulas.to_numpy() == formula)
        valid = positions[np.isfinite(result.loc[positions, 'glacier_raw'].to_numpy(float))]
        if len(valid) < 2:
            continue
        base = result.loc[valid, 'score'].to_numpy(float)
        forward = result.loc[valid, 'glacier_raw'].to_numpy(float)
        spread = float(np.ptp(forward))
        absolute = float(np.clip((float(forward.max()) - .05) / .35, 0., 1.))
        separation = float(np.clip(spread / .10, 0., 1.))
        confidence = math.sqrt(absolute * separation)
        combined = base + float(weight) * confidence * (forward - np.median(forward))
        chosen_order = np.argsort(-combined, kind='stable')
        chosen = valid[chosen_order]
        # Fill the same formula's original slots with its newly ordered rows.
        replacement = result.iloc[chosen].copy().reset_index(drop=True)
        result.iloc[valid] = replacement.to_numpy()
        result.loc[valid, 'glacier_fused'] = combined[chosen_order]
        result.loc[valid, 'glacier_confidence'] = confidence
    return result


def global_rerank(frame, weight):
    """Optional experimental cross-formula fusion; formula-slot mode is safer."""
    result = frame.reset_index(drop=True).copy()
    raw = result['glacier_raw'].to_numpy(float)
    valid = np.isfinite(raw)
    fused = _standard_score(result['score'].to_numpy(float))
    if valid.sum() >= 2 and np.ptp(raw[valid]) > 1e-12:
        component = np.zeros(len(result), dtype=np.float64)
        component[valid] = _standard_score(raw[valid])
        fused += float(weight) * component
    result['glacier_fused'] = fused
    result['glacier_confidence'] = float(np.isfinite(raw).mean())
    return result.sort_values(['glacier_fused', 'error_ppm', 'key'],
                              ascending=[False, True, True], kind='stable').reset_index(drop=True)


def entropy_probabilities(weights):
    p = np.asarray(weights, dtype=np.float64) ** 2
    total = float(p.sum())
    if total <= 0:
        return p
    p /= total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    if entropy < 3.0:
        p **= .25 + .25 * entropy
        p /= max(float(p.sum()), 1e-30)
    return p


def entropy_similarity(mz_a, weight_a, mz_b, weight_b, tolerance_da=.02, tolerance_ppm=10.):
    """One-to-one spectral entropy similarity on sorted peak lists."""
    mz_a, mz_b = np.asarray(mz_a, dtype=np.float64), np.asarray(mz_b, dtype=np.float64)
    pa, pb = entropy_probabilities(weight_a), entropy_probabilities(weight_b)
    i = j = 0
    score = 0.
    while i < len(mz_a) and j < len(mz_b):
        tolerance = max(float(tolerance_da), min(mz_a[i], mz_b[j]) * float(tolerance_ppm) * 1e-6)
        delta = mz_a[i] - mz_b[j]
        if abs(delta) <= tolerance:
            x, y = pa[i], pb[j]
            if x > 0 and y > 0:
                score += ((x + y) * math.log2(x + y) - x * math.log2(x) - y * math.log2(y)) / 2
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return float(np.clip(score, 0., 1.))


def postprocess_prediction(masses, intensities, top_n=100):
    """Apply the bundle's documented deterministic GLACIER post-processing."""
    masses = np.asarray(masses, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    keep = np.isfinite(masses) & np.isfinite(intensities) & (masses > 0) & (intensities > 0)
    masses, intensities = masses[keep], intensities[keep]
    if not len(masses):
        return np.empty(0, np.float32), np.empty(0, np.float32)
    rounded = np.round(masses, 4)
    unique, inverse = np.unique(rounded, return_inverse=True)
    merged = np.zeros(len(unique), dtype=np.float64)
    np.maximum.at(merged, inverse, intensities)
    if len(unique) > top_n:
        chosen = np.argpartition(merged, -top_n)[-top_n:]
        unique, merged = unique[chosen], merged[chosen]
    order = np.argsort(unique, kind='stable')
    unique, merged = unique[order], merged[order]
    merged /= max(float(merged.max()), 1e-30)
    # entropy_probabilities squares its input; sqrt preserves the model's intensities.
    weights = np.sqrt(merged)
    weights /= max(float(np.linalg.norm(weights)), 1e-30)
    return unique.astype(np.float32), weights.astype(np.float32)


def aggregate_job_scores(records):
    """Quality-weight CE replicates, then weight each covered adduct equally."""
    groups = defaultdict(list)
    for record in records:
        if np.isfinite(record['score']):
            groups[(record['adduct'], record['mode'])].append(record)
    values = []
    for group in groups.values():
        weights = np.asarray([max(float(x['quality']), 1e-6) for x in group])
        scores = np.asarray([float(x['score']) for x in group])
        values.append(float(np.average(scores, weights=weights)))
    return float(np.mean(values)) if values else np.nan


def instrument_name(value):
    try:
        missing = value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    text = str('' if missing else value).lower().replace(' ', '')
    if 'tof' in text:
        return 'QTOF'
    if 'orbitrap' in text or 'ft' in text:
        return 'Orbitrap'
    return 'Unknown'


def collision_bucket(value):
    value = mean_energy(value)
    if not np.isfinite(value):
        return np.nan
    return float(5 * math.floor(abs(value) / 5 + .5))


def _install_import_stubs():
    """Provide only training-time imports that inference never calls."""
    if 'pytorch_lightning' not in sys.modules:
        lightning = types.ModuleType('pytorch_lightning')

        class LightningModule(nn.Module):
            def save_hyperparameters(self, *args, **kwargs):
                return None

            @property
            def device(self):
                parameter = next(self.parameters(), None)
                return parameter.device if parameter is not None else torch.device('cpu')

            def freeze(self):
                self.requires_grad_(False)
                self.eval()

            def log(self, *args, **kwargs):
                return None

        lightning.LightningModule = LightningModule
        loggers = types.ModuleType('pytorch_lightning.loggers')
        loggers_base = types.ModuleType('pytorch_lightning.loggers.base')
        utilities = types.ModuleType('pytorch_lightning.utilities')

        class Logger:
            pass

        def identity_decorator(function):
            return function

        loggers.LightningLoggerBase = Logger
        loggers.Logger = Logger
        loggers_base.rank_zero_experiment = identity_decorator
        utilities.rank_zero_only = identity_decorator
        sys.modules['pytorch_lightning'] = lightning
        sys.modules['pytorch_lightning.loggers'] = loggers
        sys.modules['pytorch_lightning.loggers.base'] = loggers_base
        sys.modules['pytorch_lightning.loggers.logger'] = loggers
        sys.modules['pytorch_lightning.utilities'] = utilities

    if 'pygmtools' not in sys.modules:
        pygm = types.ModuleType('pygmtools')

        def unavailable(*args, **kwargs):
            raise RuntimeError('A training-only pygmtools operation was reached during GLACIER inference.')

        pygm.sinkhorn = unavailable
        pygm.hungarian = unavailable
        sys.modules['pygmtools'] = pygm

    # ms_pred.common imports several database/training utilities at package import
    # time.  The forward path only needs chem_utils, so keep those optional stacks
    # (h5py, multiprocess/pathos, dataset splitters) out of the Kaggle runtime.
    for name in ('ms_pred.common.fingerprint', 'ms_pred.common.misc_utils',
                 'ms_pred.common.parallel_utils', 'ms_pred.common.splitter'):
        sys.modules.setdefault(name, types.ModuleType(name))
    denoising = types.ModuleType('ms_pred.common.denoising_utils')
    denoising.electronic_denoising = lambda spectrum: spectrum
    sys.modules.setdefault('ms_pred.common.denoising_utils', denoising)
    for name in ('ms_pred.nn_utils.tune_utils', 'ms_pred.nn_utils.base_hyperopt'):
        sys.modules.setdefault(name, types.ModuleType(name))
    plot = types.ModuleType('ms_pred.common.plot_utils')
    plot.plot_mol_as_vector = lambda *args, **kwargs: None
    plot.plot_compare_ms = lambda *args, **kwargs: None
    plot.plot_ms = lambda *args, **kwargs: None
    sys.modules.setdefault('ms_pred.common.plot_utils', plot)


class GlacierRuntime:
    def __init__(self, bundle, device):
        root, checkpoint, _ = validate_bundle(bundle)
        sys.path.insert(0, str(root / 'src'))
        sys.path.insert(0, str(root / 'shim'))
        _install_import_stubs()
        from ms_pred.glacier.dataset import IntenDataset, TreeProcessor
        from ms_pred.glacier.joint_model import JointModel
        from ms_pred import common

        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if not {'hyper_parameters', 'state_dict'}.issubset(payload):
            raise ValueError('Expected a stripped {hyper_parameters, state_dict} GLACIER checkpoint.')
        if len(payload['state_dict']) != CHECKPOINT_TENSORS:
            raise ValueError('Unexpected number of GLACIER state tensors.')
        state_values = list(payload['state_dict'].values())
        if (not all(isinstance(value, torch.Tensor) for value in state_values)
                or sum(value.numel() for value in state_values) != CHECKPOINT_PARAMETERS):
            raise ValueError('Unexpected GLACIER checkpoint tensor payload.')
        self.model = JointModel(**payload['hyper_parameters'])
        self.model.load_state_dict(payload['state_dict'], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = TreeProcessor(pe_embed_k=self.model.pe_embed_k,
                                       root_encode='graphormer',
                                       embed_elem_group=self.model.embed_elem_group,
                                       multi_hop_max_dist=self.model.multi_hop_max_dist)
        self.dataset_class = IntenDataset
        self.common = common
        self.device = torch.device(device)

    def move_model(self):
        self.model.to(self.device)
        return self


class GlacierJobDataset(Dataset):
    def __init__(self, jobs, processor, adduct_map, instrument_map, cache_size=1024):
        self.jobs = sorted(jobs, key=lambda x: (x.get('atoms', 0), x['smiles'], x['adduct'], x['id']))
        self.processor = processor
        self.adduct_map = adduct_map
        self.instrument_map = instrument_map
        self.cache_size = max(0, int(cache_size))
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, index):
        job = self.jobs[index]
        try:
            key = (job['smiles'], job['adduct'])
            feat = self.cache.get(key)
            if feat is None:
                tree = {'root_canonical_smiles': job['smiles'], 'adduct': job['adduct'],
                        'collision_energy': job['ce']}
                feat = self.processor.featurize_tree(
                    tree, include_inten_targs=False, include_frag_targs=False)
                if self.cache_size:
                    self.cache[key] = feat
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
            elif self.cache_size:
                self.cache.move_to_end(key)
            feat = dict(feat)
            feat['collision_energy'] = job['ce']
            item = {
                'name': str(job['id']),
                'adduct': self.adduct_map[job['adduct']],
                'precursor': 0.,
                'instrument': self.instrument_map.get(job['instrument'],
                                                      self.instrument_map['Unknown']),
                '_job_id': int(job['id']),
            }
            item.update(feat)
            return item
        except Exception as error:
            return {'_job_id': int(job['id']),
                    '_error': f'{type(error).__name__}: {error}'}


def glacier_collate(batch, dataset_class):
    errors = [(item['_job_id'], item['_error']) for item in batch if '_error' in item]
    valid = [item for item in batch if '_error' not in item]
    if not valid:
        return {'_job_ids': [], '_feature_errors': errors}
    result = dataset_class.collate_fn(valid)
    result['_job_ids'] = [item['_job_id'] for item in valid]
    result['_feature_errors'] = errors
    return result


def move_batch(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
        elif key == 'graphormer_input' and value is not None:
            for inner in value:
                value[inner] = value[inner].to(device, non_blocking=True)
    return batch


def run_worker(args):
    started = time.monotonic()
    payload = read_json(args.jobs)
    queries = payload['queries']
    jobs = payload['jobs']
    scores, errors = [], []
    completed = set()

    def persist():
        write_json(args.worker_output, {
            'version': VERSION, 'scores': scores, 'jobs': len(jobs),
            'completed_predictions': len(completed), 'comparisons': len(scores),
            'seconds': time.monotonic() - started, 'errors': errors[:20],
        })

    # A hard model/import failure still leaves an explicit empty partial result.
    persist()
    if not jobs:
        return
    runtime = GlacierRuntime(args.bundle, args.device)
    workers = min(int(args.workers), len(jobs))
    cache_size = int(args.feature_cache)
    if workers > 1 and cache_size:
        cache_size = max(64, math.ceil(cache_size / workers))
    dataset = GlacierJobDataset(jobs, runtime.processor, runtime.common.ion2onehot_pos,
                                runtime.common.instrument2onehot_pos, cache_size)
    loader_args = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': workers,
        'collate_fn': functools.partial(glacier_collate, dataset_class=runtime.dataset_class),
    }
    if workers:
        loader_args.update(multiprocessing_context='fork', prefetch_factor=2,
                           persistent_workers=False, pin_memory=False)
    loader = DataLoader(**loader_args)
    iterator = iter(loader)  # fork CPU workers before CUDA initialization
    runtime.move_model()
    job_by_id = {int(job['id']): job for job in jobs}
    consecutive_failures = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(
                tqdm(iterator, total=len(loader), desc=f'GLACIER {args.device}'), start=1):
            if time.monotonic() - started > args.time_budget:
                errors.append('time_budget_exhausted')
                break
            for job_id, message in batch.pop('_feature_errors'):
                errors.append(f'job {job_id} feature: {message}')
            ids = batch.pop('_job_ids')
            if not ids:
                if batch_number % 32 == 0:
                    persist()
                continue
            try:
                batch = move_batch(batch, runtime.device)
                predicted = runtime.model.predict_inten_frag_batch_sparse(batch, args.sparse_peaks)
                for row, job_id in enumerate(ids):
                    count = int(predicted['counts'][row])
                    mz, weights = postprocess_prediction(
                        predicted['masses'][row, :count], predicted['intens'][row, :count],
                        args.sparse_peaks)
                    for query_index in job_by_id[int(job_id)]['queries']:
                        query = queries[query_index]
                        score = entropy_similarity(query['mz'], query['weights'], mz, weights,
                                                   args.tolerance_da, args.tolerance_ppm)
                        scores.append([int(job_id), int(query_index), score])
                    completed.add(int(job_id))
                consecutive_failures = 0
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
                consecutive_failures += 1
                if runtime.device.type == 'cuda':
                    torch.cuda.empty_cache()
                if consecutive_failures >= 3:
                    errors.append('aborted_after_three_consecutive_batch_failures')
                    break
            if batch_number % 32 == 0:
                persist()
    persist()


def unique_candidates(frame, limit):
    seen = set()
    rows = []
    for index, row in frame.iterrows():
        try:
            key = scorer_key(str(row['smiles']))
        except (ValueError, RuntimeError):
            continue
        if key in seen:
            continue
        seen.add(key)
        rows.append(index)
        if len(rows) == limit:
            break
    result = frame.loc[rows].copy().reset_index(drop=True)
    result['base_rank'] = np.arange(1, len(result) + 1)
    return result


def build_jobs(frame, candidates, cache):
    jobs, queries = [], []
    mapping = []
    next_id = 0
    for molecule, group in frame.groupby('molecule_id', sort=False):
        ranked = candidates[str(molecule)]
        prepared = []
        for _, candidate in ranked.iterrows():
            mol = Chem.MolFromSmiles(str(candidate['smiles']))
            if (mol is None or len(Chem.GetMolFrags(mol)) != 1
                    or any(atom.GetAtomicNum() not in SUPPORTED_ATOMIC_NUMBERS
                           for atom in mol.GetAtoms())):
                continue
            mol = Chem.RemoveHs(mol)
            if not mol.GetNumAtoms() or mol.GetNumAtoms() > 100:
                continue
            prepared.append({'key': str(candidate['key']),
                             'smiles': Chem.MolToSmiles(mol),
                             'atoms': int(mol.GetNumAtoms())})
        signatures = {}
        built = build_queries(group, cache)
        for row, query in zip(group.to_dict('records'), built):
            adduct = str(row['adduct'])
            mode = str(row.get('ionization_mode', '')).lower()
            ce = collision_bucket(row.get('collision_energy_ev'))
            if (adduct not in SUPPORTED_ADDUCTS or mode != 'positive' or not np.isfinite(ce)
                    or not len(query['mz'])):
                continue
            query_index = len(queries)
            queries.append({'mz': query['mz'].astype(float).tolist(),
                            'weights': query['weights'].astype(float).tolist()})
            instrument = instrument_name(row.get('instrument_type'))
            for candidate in prepared:
                signature = (candidate['key'], candidate['smiles'], adduct, instrument, ce)
                job_id = signatures.get(signature)
                if job_id is None:
                    job_id = next_id
                    signatures[signature] = job_id
                    jobs.append({'id': job_id, 'molecule_id': str(molecule),
                                 **candidate, 'adduct': adduct, 'instrument': instrument,
                                 'ce': ce, 'queries': []})
                    next_id += 1
                job = jobs[job_id]
                job['queries'].append(query_index)
                mapping.append({'id': job_id, 'molecule_id': str(molecule),
                                'key': candidate['key'], 'query': query_index,
                                'adduct': adduct, 'mode': mode,
                                'quality': float(query['quality'])})
    return jobs, queries, mapping


def shard_jobs(jobs, shards):
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['molecule_id'], job['key'])].append(job)
    result = [[] for _ in range(shards)]
    loads = [0] * shards
    for group in sorted(groups.values(), key=len, reverse=True):
        target = int(np.argmin(loads))
        result[target].extend(group)
        loads[target] += len(group)
    return result


def launch_workers(args, jobs, queries, runtime_dir):
    if not jobs:
        return {}, [], []
    devices = [part.strip() for part in args.devices.split(',') if part.strip()]
    if not devices or devices == ['auto']:
        devices = [str(i) for i in range(torch.cuda.device_count())] or ['cpu']
    partitions = shard_jobs(jobs, len(devices))
    processes, outputs = [], []
    for index, (device, partition) in enumerate(zip(devices, partitions)):
        if not partition:
            continue
        job_path = runtime_dir / f'jobs_{index}.json'
        output_path = runtime_dir / f'scores_{index}.json'
        output_path.unlink(missing_ok=True)
        write_json(job_path, {'queries': queries, 'jobs': partition})
        command = [sys.executable, '-m', 'casmi_baseline.glacier',
                   '--config', str(args.config), 'worker',
                   '--bundle', str(args.bundle), '--jobs', str(job_path),
                   '--worker-output', str(output_path), '--device', 'cpu' if device == 'cpu' else 'cuda:0',
                   '--batch-size', str(args.glacier_batch_size), '--workers', str(args.glacier_workers),
                   '--feature-cache', str(args.feature_cache), '--sparse-peaks', str(args.sparse_peaks),
                   '--time-budget', str(args.time_budget), '--tolerance-da', str(args.tolerance_da),
                   '--tolerance-ppm', str(args.tolerance_ppm)]
        env = os.environ.copy()
        if device != 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = device
        processes.append(subprocess.Popen(command, env=env))
        outputs.append(output_path)
    failures = []
    deadline = time.monotonic() + float(args.time_budget) + 300.
    for process in processes:
        try:
            code = process.wait(timeout=max(1., deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            failures.append('parent_time_budget_exhausted')
            continue
        if code:
            failures.append(code)
    scores, worker_meta = {}, []
    for output in outputs:
        if not output.exists():
            continue
        record = read_json(output)
        worker_meta.append(record)
        scores.update({(int(job), int(query)): float(score)
                       for job, query, score in record['scores']})
    return scores, worker_meta, failures


def build_base_ranker(config, args, cache, pool):
    bags, peaks, fusion = diversity.resolve_models(config, args)
    checkpoint = Path(args.head_checkpoint or Path(config['work_dir']) / 'models/dreams_head.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not payload.get('training_info', {}).get('full'):
        raise ValueError('Submission requires a full DreaMS-head checkpoint.')
    bound = type('GlacierBaseRanker', (dreams_head.DreamsHeadRanker,), {
        'index_dir': args.index_dir, 'weight': args.analog_weight,
        'dreams_path': args.dreams_dir, 'direct_weight': args.direct_weight,
        'propagation_weight': args.propagation_weight, 'min_similarity': args.min_similarity,
        'neighbor_limit': args.neighbors, 'encoder_batch': args.base_batch_size,
        'scan_batch': args.scan_batch, 'head_path': checkpoint, 'head_weight': args.head_weight})
    return bound(config, cache, pool, bags, peaks, args.base_device, fusion), checkpoint


def predict(config, args):
    started = time.monotonic()
    bundle, _, manifest = validate_bundle(args.bundle)
    args.bundle = bundle
    cache, pool = diversity.open_pool(config)
    frame = pd.read_parquet(args.test_path or Path(config['data_dir']) / 'test.parquet')
    frame['molecule_id'] = frame['molecule_id'].astype(str)
    expected = frame['molecule_id'].drop_duplicates().tolist()
    ranker, head_checkpoint = build_base_ranker(config, args, cache, pool)
    candidates, base_diagnostics = {}, []
    grouped = frame.groupby('molecule_id', sort=False)
    for molecule in tqdm(expected, desc='base top-K candidates'):
        ranked, diagnostic = ranker.rank(grouped.get_group(molecule))
        candidates[molecule] = unique_candidates(ranked, args.top_candidates)
        base_diagnostics.append({'molecule_id': molecule, **diagnostic})
    del ranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    jobs, queries, mapping = build_jobs(frame, candidates, cache)
    runtime_dir = Path(args.runtime_dir or Path(args.output or 'submission.csv').with_suffix('.glacier'))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_json(runtime_dir / 'jobs.meta.json', {
        'version': VERSION, 'prediction_jobs': len(jobs), 'comparisons': len(mapping),
        'queries': len(queries),
        'top_candidates': args.top_candidates, 'bundle_manifest': manifest,
    })
    scores, worker_meta, failures = launch_workers(args, jobs, queries, runtime_dir)
    records = defaultdict(list)
    mapping_by_pair = {(entry['id'], entry['query']): entry for entry in mapping}
    for pair, score in scores.items():
        meta = mapping_by_pair[pair]
        records[(meta['molecule_id'], meta['key'])].append({**meta, 'score': score})

    submissions, base_submissions, details, diagnostics = [], [], [], []
    for base in base_diagnostics:
        molecule = base['molecule_id']
        ranked = candidates[molecule].copy()
        base_selected = ranked.iloc[:25]
        base_submissions.append({'molecule_id': molecule,
                                 'smiles': ';'.join(base_selected['smiles'])})
        ranked['glacier_raw'] = [aggregate_job_scores(records[(molecule, str(key))])
                                 for key in ranked['key']]
        ranked['glacier_covered'] = np.isfinite(ranked['glacier_raw'])
        if args.fusion_mode == 'formula':
            ranked = formula_slot_rerank(ranked, args.glacier_weight)
        else:
            ranked = global_rerank(ranked, args.glacier_weight)
        ranked['molecule_id'] = molecule
        ranked['rank'] = np.arange(1, len(ranked) + 1)
        selected = ranked.iloc[:25].copy()
        submissions.append({'molecule_id': molecule, 'smiles': ';'.join(selected['smiles'])})
        details.append(selected)
        base_keys = base_selected['key'].astype(str).tolist()
        selected_keys = selected['key'].astype(str).tolist()
        diagnostics.append({**base,
                            'glacier_candidates': int(ranked['glacier_covered'].sum()),
                            'glacier_top1_changed': bool(selected_keys and base_keys
                                                         and selected_keys[0] != base_keys[0]),
                            'glacier_top25_order_changed': selected_keys != base_keys,
                            'glacier_best': float(np.nanmax(ranked['glacier_raw']))
                            if ranked['glacier_covered'].any() else None})
    submission = pd.DataFrame(submissions)
    base_submission = pd.DataFrame(base_submissions)
    check_submission(submission, expected)
    check_submission(base_submission, expected)
    output = Path(args.output or 'submission.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    base_output = output.with_suffix('.base.csv')
    base_submission.to_csv(base_output, index=False)
    pd.concat(details, ignore_index=True).to_csv(output.with_suffix('.candidates.csv'), index=False)
    write_json(output.with_suffix('.diagnostics.json'), {
        'version': VERSION, 'engine': 'glacier-forward-reranker',
        'seconds': time.monotonic() - started, 'molecules': len(submission),
        'spectra': len(frame), 'prediction_jobs': len(jobs),
        'completed_predictions': len({job for job, _ in scores}),
        'comparisons': len(mapping), 'completed_comparisons': len(scores),
        'worker_failures': failures, 'workers': worker_meta,
        'fusion_mode': args.fusion_mode, 'glacier_weight': args.glacier_weight,
        'top_candidates': args.top_candidates, 'bundle': str(bundle),
        'base_output': str(base_output),
        'top1_changed': sum(item['glacier_top1_changed'] for item in diagnostics),
        'top25_order_changed': sum(item['glacier_top25_order_changed'] for item in diagnostics),
        'checkpoint_sha256': CHECKPOINT_SHA256, 'source_checkpoint_sha256': SOURCE_CHECKPOINT_SHA256,
        'ms_pred_commit': MS_PRED_COMMIT, 'head_checkpoint': str(head_checkpoint),
        'queries': diagnostics,
    })
    print(f'Saved {output}; {len(submission)} molecules; '
          f'{len({job for job, _ in scores})}/{len(jobs)} GLACIER predictions; '
          f'{time.monotonic() - started:.1f}s')
    return submission


def parser():
    p = argparse.ArgumentParser(description='Independent GLACIER forward-spectrum re-ranker')
    p.add_argument('--config', default=str(ROOT / 'configs/local.yaml'))
    p.add_argument('--data-dir')
    p.add_argument('--work-dir')
    sub = p.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check-bundle', help='Verify every executable asset and the checkpoint')
    check.add_argument('--bundle', required=True)
    item = sub.add_parser('predict')
    item.add_argument('--test-path')
    item.add_argument('--output', default='submission.csv')
    item.add_argument('--bundle', required=True)
    item.add_argument('--runtime-dir')
    item.add_argument('--devices', default='auto')
    item.add_argument('--top-candidates', type=int, default=64)
    item.add_argument('--glacier-weight', type=float, default=.75)
    item.add_argument('--fusion-mode', choices=['formula', 'global'], default='formula')
    item.add_argument('--glacier-batch-size', type=int, default=8)
    item.add_argument('--glacier-workers', type=int, default=2)
    item.add_argument('--feature-cache', type=int, default=1024)
    item.add_argument('--sparse-peaks', type=int, default=100)
    item.add_argument('--time-budget', type=float, default=24000.)
    item.add_argument('--tolerance-da', type=float, default=.02)
    item.add_argument('--tolerance-ppm', type=float, default=10.)
    item.add_argument('--checkpoint', action='append')
    item.add_argument('--peak-checkpoints', nargs='+')
    item.add_argument('--fusion')
    item.add_argument('--base-device', default='auto')
    item.add_argument('--index-dir')
    item.add_argument('--analog-weight', type=float, default=.18)
    item.add_argument('--dreams-dir')
    item.add_argument('--direct-weight', type=float, default=.10)
    item.add_argument('--propagation-weight', type=float, default=.10)
    item.add_argument('--min-similarity', type=float, default=.60)
    item.add_argument('--neighbors', type=int, default=64)
    item.add_argument('--base-batch-size', type=int, default=8)
    item.add_argument('--scan-batch', type=int, default=16384)
    item.add_argument('--head-checkpoint')
    item.add_argument('--head-weight', type=float, default=.16)
    item.add_argument('--threads', type=int, default=2)

    worker = sub.add_parser('worker', help=argparse.SUPPRESS)
    worker.add_argument('--bundle', required=True)
    worker.add_argument('--jobs', required=True)
    worker.add_argument('--worker-output', required=True)
    worker.add_argument('--device', required=True)
    worker.add_argument('--batch-size', type=int, default=8)
    worker.add_argument('--workers', type=int, default=2)
    worker.add_argument('--feature-cache', type=int, default=1024)
    worker.add_argument('--sparse-peaks', type=int, default=100)
    worker.add_argument('--time-budget', type=float, default=24000.)
    worker.add_argument('--tolerance-da', type=float, default=.02)
    worker.add_argument('--tolerance-ppm', type=float, default=10.)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'check-bundle':
        root, checkpoint, manifest = validate_bundle(args.bundle)
        print(f'Validated {root}; {checkpoint.stat().st_size} bytes; '
              f'{manifest["ckpt_params"]} parameters')
        return root
    if args.command == 'worker':
        if min(args.batch_size, args.sparse_peaks) < 1 or args.workers < 0:
            raise ValueError('Invalid GLACIER worker settings.')
        return run_worker(args)
    if (args.top_candidates < 25 or args.glacier_weight < 0 or args.glacier_batch_size < 1
            or args.glacier_workers < 0 or args.feature_cache < 0 or args.sparse_peaks < 1
            or args.time_budget <= 0 or args.tolerance_da < 0 or args.tolerance_ppm < 0):
        raise ValueError('Invalid GLACIER prediction settings.')
    torch.set_num_threads(args.threads)
    config = load_config(args.config, args.data_dir, args.work_dir)
    return predict(config, args)


if __name__ == '__main__':
    main()
