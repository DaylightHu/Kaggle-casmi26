"""Compatible CASMI extension: peak-set model, public candidates and rank fusion.

Add this file to casmi_baseline/. Run `python -m casmi_baseline.diversity --help`.
Existing caches, fingerprint checkpoints and CLI are unchanged. Only `train`
and `fit-fusion` fit parameters; prepare/import/export/predict never train.

Entropy similarity is an independent implementation of the published formula:
https://doi.org/10.1038/s41592-021-01331-z
Public natural-product catalog: https://coconut.naturalproducts.net/download
This module does not contain pretrained weights or external data.
"""
import argparse
import contextlib
import gzip
import hashlib
import itertools
import json
import math
import random
import re
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .cache import ReferenceCache
from .candidates import CandidatePool
from .chemistry import check_rdkit, describe_structure, neutral_mass, scorer_key
from .common import ROOT, file_digest, load_config, read_json, stable_fraction, write_json
from .fingerprint import (FingerprintPredictor, TrainingSpectra, choose_device,
                          query_from_cache, seed_worker)
from .retrieval import Ranker, build_queries, weighted_median
from .spectra import njit
from .validation import load_split

VERSION = 1
MACCS_BITS, PATH_BITS = 167, 512
ELEMENTS = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Na', 'K', 'Se', 'other']
FEATURES = ['cosine', 'entropy', 'loss_entropy', 'bag_tanimoto', 'bag_evidence',
            'peak_morgan', 'peak_maccs', 'peak_path', 'embedding', 'formula', 'mass']
WEAK = np.array([.04, .04, .02, .18, .06, .18, .10, .10, .18, .08, .02], dtype=np.float32)
STRONG = np.array([.40, .25, .10, .05, .02, .05, .025, .025, .05, .02, .01], dtype=np.float32)


def open_pool(config):
    work = Path(config['work_dir'])
    cache = ReferenceCache(work / 'cache')
    cache.require_config(config)
    check_rdkit()
    return cache, CandidatePool(cache, work / 'candidates')


def key_digest(table):
    return hashlib.sha256('\n'.join(table['key'].astype(str)).encode()).hexdigest()


def descriptor(smiles):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import MACCSkeys, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles)
    maccs = np.zeros(MACCS_BITS, dtype=np.uint8)
    path = np.zeros(PATH_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
    DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol, fpSize=PATH_BITS), path)
    counts = np.zeros(len(ELEMENTS), dtype=np.uint16)
    for element, count in re.findall(r'([A-Z][a-z]?)(\d*)', rdMolDescriptors.CalcMolFormula(mol)):
        index = ELEMENTS.index(element) if element in ELEMENTS else len(ELEMENTS) - 1
        counts[index] += int(count or '1')
    return np.packbits(maccs, bitorder='little'), np.packbits(path, bitorder='little'), counts


def prepare(config, workers=4):
    cache, pool = open_pool(config)
    dest = Path(config['work_dir']) / 'diversity'
    dest.mkdir(parents=True, exist_ok=True)
    size = len(pool.table)
    names = [('maccs.npy', np.uint8, (size, 21)), ('paths.npy', np.uint8, (size, PATH_BITS // 8)),
             ('counts.npy', np.uint16, (size, len(ELEMENTS)))]
    arrays = [np.lib.format.open_memmap(dest / (name + '.tmp'), mode='w+', dtype=dtype, shape=shape)
              for name, dtype, shape in names]
    start = 0
    keys = pool.table['key'].astype(str).tolist()
    if (dest / 'manifest.json').exists():
        old = read_json(dest / 'manifest.json')
        if (old['version'] == VERSION and old['rdkit'] == cache.manifest['rdkit']
                and old['keys'] == keys[:len(old['keys'])]):
            start = len(old['keys'])
            for array, (name, _, _) in zip(arrays, names):
                array[:start] = np.load(dest / name, mmap_mode='r')
    context = ProcessPoolExecutor(max_workers=workers) if workers > 1 else contextlib.nullcontext()
    with context as executor:
        for pos in tqdm(range(start, size, 1024), desc='MACCS/path/formula cache'):
            smiles = pool.table.iloc[pos:pos + 1024]['smiles'].tolist()
            result = list(executor.map(descriptor, smiles, chunksize=32)) if executor else list(map(descriptor, smiles))
            for channel, array in enumerate(arrays):
                array[pos:pos + len(result)] = np.stack([r[channel] for r in result])
    for array in arrays:
        array.flush()
    del arrays
    for name, _, _ in names:
        (dest / (name + '.tmp')).replace(dest / name)
    write_json(dest / 'manifest.json', {'version': VERSION, 'rdkit': cache.manifest['rdkit'],
               'fingerprint': cache.manifest['fingerprint'], 'key_digest': key_digest(pool.table),
               'keys': keys, 'structures': size, 'elements': ELEMENTS})
    print(f'Auxiliary cache: {size} structures; {size - start} newly computed; {dest}')


class AuxiliaryPool:
    def __init__(self, config, cache, pool):
        self.pool = pool
        path = Path(config['work_dir']) / 'diversity'
        manifest = read_json(path / 'manifest.json')
        if (manifest['version'] != VERSION or manifest['key_digest'] != key_digest(pool.table)
                or manifest['rdkit'] != cache.manifest['rdkit']
                or manifest['fingerprint'] != cache.manifest['fingerprint']):
            raise ValueError('Candidate catalog changed: run diversity prepare again.')
        self.maccs = np.load(path / 'maccs.npy', mmap_mode='r')
        self.paths = np.load(path / 'paths.npy', mmap_mode='r')
        self.counts = np.load(path / 'counts.npy', mmap_mode='r')
        self.bits = cache.manifest['fingerprint']['bits']

    def targets(self, ids):
        ids = np.atleast_1d(ids).astype(np.int64)
        return np.concatenate([self.pool.fingerprints(ids),
            np.unpackbits(self.maccs[ids], axis=1, bitorder='little')[:, :MACCS_BITS],
            np.unpackbits(self.paths[ids], axis=1, bitorder='little')], axis=1).astype(np.float32)

    def features(self, ids):
        return np.concatenate([self.targets(ids), np.log1p(self.counts[np.atleast_1d(ids)]) / 4.0], axis=1).astype(np.float32)


def iter_external(path, column=None):
    """Stream CSV/TSV/Parquet/SDF/SMI, including a single CSV or SDF in ZIP."""
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        name = column or next(c for c in ['canonical_smiles', 'smiles', 'SMILES', 'normalized_smiles'] if c in pf.schema.names)
        for batch in pf.iter_batches(batch_size=4096, columns=[name]):
            yield from batch.column(0).to_pylist()
        return
    with contextlib.ExitStack() as stack:
        filename = path.name.lower()
        if path.suffix == '.zip':
            archive = stack.enter_context(zipfile.ZipFile(path))
            members = [n for n in archive.namelist() if n.lower().endswith(('.csv', '.tsv', '.sdf', '.smi'))]
            if len(members) != 1:
                raise ValueError(f'ZIP must contain exactly one data file; members={members}')
            filename = members[0].lower()
            handle = stack.enter_context(archive.open(members[0]))
        elif path.suffix == '.gz':
            filename = filename[:-3]
            handle = stack.enter_context(gzip.open(path, 'rb'))
        else:
            handle = stack.enter_context(path.open('rb'))
        if filename.endswith('.sdf'):
            from rdkit import Chem
            for mol in Chem.ForwardSDMolSupplier(handle):
                if mol is not None:
                    yield Chem.MolToSmiles(mol)
        elif filename.endswith(('.csv', '.tsv')):
            for frame in pd.read_csv(handle, sep='\t' if filename.endswith('.tsv') else ',', chunksize=4096):
                name = column or next(c for c in ['canonical_smiles', 'smiles', 'SMILES', 'normalized_smiles'] if c in frame)
                yield from frame[name]
        else:
            for raw in handle:
                line = raw.decode('utf-8').strip()
                if line and not line.startswith('#'):
                    yield line.split()[0]


def import_worker(item):
    smiles, fp_config, minimum, maximum = item
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog('rdApp.warning')
    if not isinstance(smiles, str) or not smiles.strip():
        return 'invalid', None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'invalid', None
    if not minimum <= rdMolDescriptors.CalcExactMolWt(mol) <= maximum:
        return 'outside_mass_range', None
    try:
        return 'valid', describe_structure(smiles, **fp_config)
    except (ValueError, RuntimeError, KeyError):
        return 'invalid', None


def import_catalog(config, args):
    """Filter a broad mass RANGE, never the visible test's individual masses."""
    cache, _ = open_pool(config)
    dest = Path(config['work_dir']) / 'candidates'
    dest.mkdir(parents=True, exist_ok=True)
    records, fps, seen = [], [], set(cache.structures['key'])
    if (dest / 'structures.parquet').exists():
        old = read_json(dest / 'manifest.json')
        if old['rdkit'] != cache.manifest['rdkit'] or old['fingerprint'] != config['fingerprint']:
            raise ValueError('Existing external candidates use another chemistry configuration.')
        records = pd.read_parquet(dest / 'structures.parquet').to_dict('records')
        fps = list(np.load(dest / 'fingerprints.npy'))
        seen.update(r['key'] for r in records)
    stats = {'added': 0, 'duplicate': 0, 'invalid': 0, 'outside_mass_range': 0}
    context = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else contextlib.nullcontext()
    with context as executor:
        for path in args.path:
            stream = iter_external(path, args.smiles_column)
            bar = tqdm(desc=f'import {Path(path).name}', unit='structures')
            while True:
                items = list(itertools.islice(stream, 1024))
                if not items:
                    break
                requests = [(s, config['fingerprint'], args.min_mass, args.max_mass) for s in items]
                result = executor.map(import_worker, requests, chunksize=32) if executor else map(import_worker, requests)
                for status, value in result:
                    if status != 'valid':
                        stats[status] += 1
                        continue
                    record, fp = value
                    if record['key'] in seen:
                        stats['duplicate'] += 1
                        continue
                    seen.add(record['key'])
                    record.update(sid=-1, count=0, np_count=0, source=args.name)
                    records.append(record)
                    fps.append(fp)
                    stats['added'] += 1
                bar.update(len(items))
            bar.close()
    if not records:
        print('No additional valid external structures:', stats)
        return
    pd.DataFrame(records).to_parquet(dest / 'structures.parquet', index=False)
    np.save(dest / 'fingerprints.npy', np.stack(fps))
    write_json(dest / 'manifest.json', {'format_version': 1, 'rdkit': cache.manifest['rdkit'],
        'fingerprint': config['fingerprint'], 'structures': len(records), 'last_import': args.path, **stats})
    print('External catalog:', len(records), stats)


def peak_item(query):
    mz = np.asarray(query['mz'], dtype=np.float32)
    weights = np.asarray(query['weights'], dtype=np.float32)
    # The cache stores sqrt(intensity), L2-normalized. Recover raw relative intensity.
    intensity = weights ** 2
    intensity /= max(float(intensity.max(initial=0)), 1e-8)
    peaks = np.stack([mz, np.maximum(query['precursor'] - mz, 0), intensity], axis=1)
    ce = query['ce']
    meta = np.array([query['precursor']/1500, query['mass']/1500,
        min(ce/100, 5) if np.isfinite(ce) else 0, float(not np.isfinite(ce)),
        float(query['mode'] == 1), float(query['mode'] == -1), float(query['timstof']), len(mz)/128], dtype=np.float32)
    return {'peaks': peaks, 'metadata': meta, 'adduct': query['adduct']}


def collate_peaks(items):
    length = max(1, max(len(v['peaks']) for v in items))
    peaks = np.zeros((len(items), length, 3), dtype=np.float32)
    mask = np.ones((len(items), length), dtype=bool)
    for i, item in enumerate(items):
        n = len(item['peaks'])
        peaks[i, :n], mask[i, :n] = item['peaks'], False
    batch = {'peaks': torch.from_numpy(peaks), 'mask': torch.from_numpy(mask),
             'metadata': torch.from_numpy(np.stack([v['metadata'] for v in items])),
             'adduct': torch.tensor([v['adduct'] for v in items], dtype=torch.long)}
    if 'target' in items[0]:
        for key in ['target', 'counts', 'molecule', 'negatives']:
            batch[key] = torch.from_numpy(np.stack([v[key] for v in items]))
        batch['sid'] = torch.tensor([v['sid'] for v in items], dtype=torch.long)
    return batch


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class PeakSetNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dimension = cfg['dimension']
        bits = cfg['bits'] + MACCS_BITS + PATH_BITS
        self.register_buffer('frequencies', torch.logspace(-3, math.log10(64), 24))
        self.peak_projection = nn.Sequential(nn.Linear(102, dimension), nn.GELU(), nn.LayerNorm(dimension))
        self.adduct = nn.Embedding(cfg['num_adducts'], 24)
        self.metadata = nn.Sequential(nn.Linear(32, dimension), nn.GELU(), nn.Linear(dimension, dimension))
        layer = nn.TransformerEncoderLayer(dimension, 6, dimension * 3, cfg['dropout'],
                                           batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(layer, cfg['layers'], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)
        self.fingerprint = nn.Linear(dimension, bits)
        self.formula = nn.Linear(dimension, len(ELEMENTS))
        self.spectrum_projection = nn.Sequential(nn.Linear(dimension, 192), nn.GELU(), nn.Linear(192, 128))
        self.structure_projection = nn.Sequential(nn.Linear(bits + len(ELEMENTS), 384),
                                                  nn.GELU(), nn.Linear(384, 128))

    def structure_embedding(self, features):
        return nn.functional.normalize(self.structure_projection(features), dim=-1)

    def forward(self, batch):
        peaks = batch['peaks']
        masses = peaks[:, :, :2]
        phase = masses.unsqueeze(-1) * self.frequencies * (2 * math.pi)
        fourier = torch.cat([phase.sin(), phase.cos()], dim=-1).flatten(2)
        intensity = peaks[:, :, 2:3]
        raw = torch.cat([masses/1500, intensity, torch.log1p(100*intensity)/math.log(101),
                         (masses - masses.round()) * 2], dim=-1)
        x = self.peak_projection(torch.cat([raw, fourier], dim=-1))
        cls = self.metadata(torch.cat([batch['metadata'], self.adduct(batch['adduct'])], dim=-1)).unsqueeze(1)
        x = torch.cat([cls, x], dim=1)
        mask = torch.cat([torch.zeros((len(x), 1), device=x.device, dtype=torch.bool), batch['mask']], dim=1)
        x = self.encoder(x, src_key_padding_mask=mask)
        h = self.norm(x[:, 0])
        return {'logits': self.fingerprint(h), 'counts': nn.functional.softplus(self.formula(h)),
                'embedding': nn.functional.normalize(self.spectrum_projection(h), dim=-1)}


class PeakTrainingData(Dataset):
    def __init__(self, cache, aux, config, held_sids, profile, hard_negatives, extra_views):
        self.cache, self.aux = cache, aux
        sids = np.setdiff1d(np.arange(len(cache.structures)), held_sids)
        base = TrainingSpectra(cache, sids, {}, config['training'])
        self.sids, self.rows, self.offsets = base.sids, base.rows, base.offsets
        if extra_views < 0:
            raise ValueError('extra_views must be nonnegative.')
        self.length = len(self.sids) + round(len(self.sids) * extra_views)
        self.profile, self.hard_negatives = profile, hard_negatives
        libraries = cache.manifest['libraries']
        np_codes = [i for i, v in enumerate(libraries) if v in ['riken', 'gnps', 'massbank', 'mona', 'msdial', 'enveda-np-examples']]
        self.np_rows = np.isin(cache.meta['lib'], np_codes)
        self.np_sids = np.intersect1d(self.sids, cache.meta['sid'][self.rows[self.np_rows[self.rows]]])
        self.tims_sids = np.intersect1d(self.sids, cache.meta['sid'][self.rows[cache.meta['timstof'][self.rows]]])
        self.mass_order = self.sids[np.argsort(aux.pool.masses[self.sids])]
        self.training_masses = aux.pool.masses[self.mass_order]
        self.mass_position = {int(s): i for i, s in enumerate(self.mass_order)}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < len(self.sids):
            sid, branch = int(self.sids[index]), 0
        else:
            probabilities = [.40, .25, .35] if self.profile == 'balanced' else [.20, .50, .30]
            branch = int(np.random.choice(3, p=probabilities))
            population = [self.sids, self.np_sids, self.tims_sids][branch]
            if not len(population):
                population = self.sids
            sid = int(population[np.random.randint(len(population))])
        rows = self.rows[self.offsets[sid]:self.offsets[sid+1]]
        preferred = rows[self.np_rows[rows]] if branch == 1 else rows[self.cache.meta['timstof'][rows]] if branch == 2 else rows
        if len(preferred):
            rows = preferred
        # First select a library, then one spectrum: large libraries do not dominate each molecule.
        lib = np.random.choice(np.unique(self.cache.meta['lib'][rows]))
        rows = rows[self.cache.meta['lib'][rows] == lib]
        query = query_from_cache(self.cache, int(np.random.choice(rows)))
        n = len(query['mz'])
        keep = np.random.random(n) >= .10
        if keep.sum() < min(5, n):
            keep[np.argsort(query['weights'])[-min(5, n):]] = True
        mz = query['mz'][keep]
        query['mz'] = np.maximum(mz + np.random.normal(0, mz*2e-6), .001).astype(np.float32)
        w = query['weights'][keep] * np.exp(np.random.normal(0, .10, keep.sum()))
        query['weights'] = (w/max(np.linalg.norm(w), 1e-8)).astype(np.float32)
        item = peak_item(query)
        pos = self.mass_position[sid]
        mass = self.aux.pool.masses[sid]
        delta = max(mass*50e-6, .02)
        left = np.searchsorted(self.training_masses, mass-delta, side='left')
        right = np.searchsorted(self.training_masses, mass+delta, side='right')
        nearby = self.mass_order[left:right]
        nearby = nearby[nearby != sid]
        if not len(nearby):
            nearby = self.mass_order[max(0, pos-32):min(len(self.mass_order), pos+33)]
            nearby = nearby[nearby != sid]
        if not len(nearby):
            raise ValueError('At least two training identities are required.')
        negative = np.random.choice(nearby, self.hard_negatives, replace=len(nearby) < self.hard_negatives)
        molecule = self.aux.features([sid])[0]
        item.update(sid=sid, target=molecule[:-len(ELEMENTS)], counts=molecule[-len(ELEMENTS):],
                    molecule=molecule, negatives=self.aux.features(negative))
        return item


def peak_loss(model, output, batch):
    sections = [model.cfg['bits'], MACCS_BITS, PATH_BITS]
    logits = output['logits'].split(sections, dim=1)
    targets = batch['target'].split(sections, dim=1)
    loss = output['logits'].sum() * 0
    for weight, pred, target in zip([.60, .20, .20], logits, targets):
        p = pred.float().sigmoid()
        dice = 1 - ((2*(p*target).sum(1)+1)/(p.sum(1)+target.sum(1)+1)).mean()
        loss = loss + weight*(nn.functional.binary_cross_entropy_with_logits(pred, target) + .20*dice)
    loss = loss + .20*nn.functional.smooth_l1_loss(output['counts'].float(), batch['counts'])
    positive = model.structure_embedding(batch['molecule'])
    negative = model.structure_embedding(batch['negatives'])
    q = output['embedding']
    hard_logits = torch.cat([(q*positive).sum(1, keepdim=True), torch.einsum('bd,bkd->bk', q, negative)], dim=1)/.10
    loss = loss + .10*nn.functional.cross_entropy(hard_logits.float(), torch.zeros(len(q), dtype=torch.long, device=q.device))
    # Duplicate identities in a batch are positives, never false negatives.
    logits = q @ positive.T / .10
    positives = batch['sid'][:, None] == batch['sid'][None, :]
    contrastive = torch.logsumexp(logits.float(), dim=1) - torch.logsumexp(logits.float().masked_fill(~positives, -torch.inf), dim=1)
    return loss + .05*contrastive.mean()


def aggregate_views(values, queries):
    groups = {}
    for i, q in enumerate(queries):
        groups.setdefault((q['adduct'], q['mode']), []).append(i)
    return np.mean([np.average(values[rows], axis=0, weights=[queries[i]['quality'] for i in rows])
                    for rows in groups.values()], axis=0)


class PeakPredictor:
    def __init__(self, path, cache, device):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('kind') != 'casmi-peak-set' or payload.get('version') != VERSION:
            raise ValueError('Not a diversity peak-set checkpoint.')
        for key in ['rdkit', 'preprocess', 'fingerprint', 'adducts']:
            if payload[key] != cache.manifest[key]:
                raise ValueError(f'Peak-set checkpoint differs from cache in {key}.')
        self.device = choose_device(device)
        self.model = PeakSetNet(payload['model_config']).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval()
        self.training_info = payload['training_info']

    @classmethod
    def live(cls, model, device, info):
        value = object.__new__(cls)
        value.model, value.device, value.training_info = model, device, info
        return value

    @torch.inference_mode()
    def predict(self, queries):
        output = self.model(move(collate_peaks([peak_item(q) for q in queries]), self.device))
        p = aggregate_views(output['logits'].sigmoid().float().cpu().numpy(), queries)
        counts = aggregate_views(output['counts'].float().cpu().numpy(), queries)
        embedding = aggregate_views(output['embedding'].float().cpu().numpy(), queries)
        embedding /= max(np.linalg.norm(embedding), 1e-8)
        return p, counts, embedding

    @torch.inference_mode()
    def similarities(self, features, embedding):
        vector = self.model.structure_embedding(torch.from_numpy(features).to(self.device))
        return vector.float().cpu().numpy() @ embedding


@njit(cache=True)
def entropy_probabilities(weights):
    p = weights.astype(np.float64) ** 2
    total = p.sum()
    if total <= 0:
        return p
    p /= total
    entropy = 0.0
    for x in p:
        if x > 0:
            entropy -= x * math.log(x)
    if entropy < 3.0:
        p = p ** (.25 + .25 * entropy)
        p /= p.sum()
    return p


@njit(cache=True)
def entropy_scan(a, pa, b, pb, tolerance_da, tolerance_ppm):
    i, j, result = 0, 0, 0.0
    while i < len(a) and j < len(b):
        tolerance = max(tolerance_da, min(a[i], b[j])*tolerance_ppm*1e-6)
        delta = a[i] - b[j]
        if abs(delta) <= tolerance:
            x, y = pa[i], pb[j]
            if x > 0 and y > 0:
                result += ((x+y)*math.log2(x+y)-x*math.log2(x)-y*math.log2(y))/2
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return min(max(result, 0.0), 1.0)


@njit(cache=True)
def entropy_references(qmz, qw, precursor, adduct, rmz, rw, n, rprecursor, radduct, da, ppm):
    direct = np.zeros(len(rmz), dtype=np.float32)
    losses = direct.copy()
    qp = entropy_probabilities(qw)
    qloss = precursor - qmz[::-1]
    for i in range(len(rmz)):
        count = n[i]
        if count == 0 or len(qmz) == 0:
            continue
        mz = rmz[i, :count]
        p = entropy_probabilities(rw[i, :count])
        same = adduct == radduct[i]
        direct[i] = entropy_scan(qmz, qp, mz, p, da, ppm) * (1.0 if same else .85)
        losses[i] = entropy_scan(qloss, qp[::-1], rprecursor[i]-mz[::-1], p[::-1], da, ppm) * (1.0 if same else .90)
    return direct, losses


def rank_column(values):
    """Average tied ranks; constant/absent engines contribute exactly zero."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0 or np.ptp(values) < 1e-9:
        return np.zeros(len(values), dtype=np.float32)
    order = np.argsort(-values, kind='stable')
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float32)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end + 1)/2
    return 6.0/(5.0 + ranks)


def fusion_score(features, gate, weights=None):
    weak = np.asarray(weights['weak'], dtype=np.float32) if weights else WEAK
    strong = np.asarray(weights['strong'], dtype=np.float32) if weights else STRONG
    active = np.any(features != 0, axis=0)
    w = ((1-gate)*weak + gate*strong) * active
    return features @ (w/max(w.sum(), 1e-8))


def tanimoto(fp, probability):
    intersection = fp @ probability
    return intersection/np.maximum(fp.sum(1)+probability.sum()-intersection, 1e-8)


class DiverseRanker(Ranker):
    def __init__(self, config, cache, pool, checkpoints=(), peak_checkpoints=(), device='auto',
                 fusion=None, blocked_sids=None, blocked_rows=None, peak_predictors=None, use_library=True):
        super().__init__(config, cache, pool, None, device, blocked_sids, blocked_rows)
        self.bags = [FingerprintPredictor(p, cache, device) for p in checkpoints]
        self.peaks = list(peak_predictors or []) + [PeakPredictor(p, cache, device) for p in peak_checkpoints]
        self.aux = AuxiliaryPool(config, cache, pool) if self.peaks else None
        self.fusion = read_json(fusion) if fusion else None
        if self.fusion and self.fusion['features'] != FEATURES:
            raise ValueError('Fusion feature order differs from this module.')
        self.use_library = use_library
        self.prior = None
        if self.bags:
            withheld = set(itertools.chain.from_iterable(p.training_info.get('held_keys', []) for p in self.bags))
            sids = np.flatnonzero(~cache.structures['key'].isin(withheld).to_numpy())
            counts = np.zeros(cache.manifest['fingerprint']['bits'], dtype=np.float64)
            for start in range(0, len(sids), 4096):
                counts += np.unpackbits(cache.fps[sids[start:start+4096]], axis=1, bitorder='little').sum(0)
            self.prior = np.clip(counts/max(len(sids), 1), 1e-4, 1-1e-4).astype(np.float32)

    def entropy_views(self, ids, queries):
        direct = np.zeros((len(queries), len(ids)), dtype=np.float32)
        loss = direct.copy()
        cfg = self.config['retrieval']
        for qindex, q in enumerate(queries):
            refs, owners = [], []
            for column, sid in enumerate(ids):
                if sid >= self.pool.train_size:
                    continue
                rows = self.choose_references(int(sid), q)
                refs.extend(rows)
                owners.extend([column]*len(rows))
            refs, owners = np.asarray(refs, dtype=np.int64), np.asarray(owners, dtype=np.int64)
            for start in range(0, len(refs), 1024):
                rows = refs[start:start+1024]
                meta = self.cache.meta[rows]
                d, l = entropy_references(q['mz'], q['weights'], q['precursor'], q['adduct'],
                    self.cache.mzs[rows], self.cache.weights[rows].astype(np.float32),
                    meta['n'], meta['precursor'], meta['adduct'], cfg['fragment_da'], cfg['fragment_ppm'])
                np.maximum.at(direct[qindex], owners[start:start+1024], d)
                np.maximum.at(loss[qindex], owners[start:start+1024], l)
        return aggregate_views(direct, queries), aggregate_views(loss, queries)

    def rank_queries(self, queries):
        cfg = self.config['retrieval']
        mass = weighted_median([q['mass'] for q in queries], [q['quality'] for q in queries])
        ids = self.pool.window(mass, cfg['mass_ppm'], cfg['mass_da'])
        fallback = 'none'
        if not len(ids):
            ids = self.pool.window(mass, cfg['fallback_ppm'], cfg['fallback_da'])
            fallback = 'widened'
        if not len(ids):
            ids = self.pool.nearest(mass, 25)
            fallback = 'nearest_outside_window'
        # Score ALL mass-window candidates in chunks, rather than discard isomers by mass proximity.
        raw = np.zeros((len(ids), len(FEATURES)), dtype=np.float32)
        if self.use_library:
            raw[:, 0] = self.spectral_score(ids, queries)
            raw[:, 1], raw[:, 2] = self.entropy_views(ids, queries)
        bag = np.mean([p.predict(queries) for p in self.bags], axis=0) if self.bags else None
        peak_outputs = [p.predict(queries) for p in self.peaks]
        if peak_outputs:
            probability = np.mean([v[0] for v in peak_outputs], axis=0)
            predicted_counts = np.mean([v[1] for v in peak_outputs], axis=0)
        for start in range(0, len(ids), 1024):
            selected = ids[start:start+1024]
            fp = self.pool.fingerprints(selected)
            if bag is not None:
                raw[start:start+len(selected), 3] = tanimoto(fp, bag)
                p = np.clip(bag, 1e-4, 1-1e-4)
                evidence = np.log(p/(1-p)) - np.log(self.prior/(1-self.prior))
                raw[start:start+len(selected), 4] = fp @ evidence
            if peak_outputs:
                features = self.aux.features(selected)
                bits = self.aux.bits
                boundaries = [0, bits, bits+MACCS_BITS, bits+MACCS_BITS+PATH_BITS]
                for channel in range(3):
                    left, right = boundaries[channel:channel+2]
                    raw[start:start+len(selected), 5+channel] = tanimoto(features[:, left:right], probability[left:right])
                raw[start:start+len(selected), 8] = np.mean([
                    model.similarities(features, result[2]) for model, result in zip(self.peaks, peak_outputs)], axis=0)
                error = np.abs(features[:, -len(ELEMENTS):] - predicted_counts)
                element_weights = np.array([.25, .25] + [1.0]*(len(ELEMENTS)-2))
                raw[start:start+len(selected), 9] = -(error*element_weights).sum(1)
        ppm = np.abs(self.pool.masses[ids]-mass)/mass*1e6
        mass_views = np.stack([np.exp(-.5*((self.pool.masses[ids]-q['mass'])/q['mass']*1e6/10.0)**2) for q in queries])
        raw[:, 10] = aggregate_views(mass_views, queries)
        normalized = np.stack([rank_column(raw[:, i]) for i in range(len(FEATURES))], axis=1)
        # Preserve absolute library evidence instead of ranking tiny noise matches as strong hits.
        for channel in [0, 1, 2]:
            normalized[:, channel] = .65*raw[:, channel] + .35*normalized[:, channel]
        gate = float(np.clip((max(raw[:, 0].max(initial=0), raw[:, 1].max(initial=0))-.45)/.35, 0, 1))
        score = fusion_score(normalized, gate, self.fusion)
        result = self.pool.table.iloc[ids].copy()
        result['candidate_id'], result['score'], result['error_ppm'], result['gate'] = ids, score, ppm, gate
        for i, name in enumerate(FEATURES):
            result[name], result['raw_'+name] = normalized[:, i], raw[:, i]
        result = result.sort_values(['score', 'error_ppm', 'key'], ascending=[False, True, True], kind='stable')
        diagnostic = {'neutral_mass': mass, 'spectra': len(queries), 'candidate_count': len(ids),
            'mass_fallback': fallback, 'best_cosine': float(raw[:, 0].max(initial=0)),
            'best_entropy': float(raw[:, 1].max(initial=0)), 'library_gate': gate}
        return result, diagnostic

    def rank(self, frame):
        return self.rank_queries(build_queries(frame, self.cache))


def validation_rank(model, config, cache, pool, split, info, device, limit):
    """Model selection by molecule-level candidate MRR, with all held references disabled."""
    grouped = {}
    for row in split['query_rows']:
        sid = int(cache.meta['sid'][row])
        grouped.setdefault(sid, []).append(row)
    sids = sorted(grouped, key=lambda s: stable_fraction(cache.structures.iloc[s]['key'], 101))[:limit]
    predictor = PeakPredictor.live(model, device, info)
    ranker = DiverseRanker(config, cache, pool, peak_predictors=[predictor],
                          blocked_sids=split['held_sids'], blocked_rows=split['blocked_rows'], use_library=False)
    scores, covered = [], []
    model.eval()
    for sid in sids:
        queries = [query_from_cache(cache, row) for row in grouped[sid]]
        ranked, _ = ranker.rank_queries(queries)
        correct = ranked['key'].to_numpy() == cache.structures.iloc[sid]['key']
        positions = np.flatnonzero(correct)
        covered.append(bool(len(positions)))
        scores.append(1.0/(positions[0]+1) if len(positions) and positions[0] < 25 else 0.0)
    return {'mrr_at_25': float(np.mean(scores)), 'candidate_coverage': float(np.mean(covered)), 'molecules': len(scores)}


def train(config, args):
    cache, pool = open_pool(config)
    aux = AuxiliaryPool(config, cache, pool)
    split_path = args.split or Path(config['work_dir']) / 'validation/split.json'
    split = None if args.full else load_split(split_path, cache)
    held = [] if args.full else split['held_sids']
    seed = args.seed if args.seed is not None else int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model_config = {'bits': config['fingerprint']['bits'], 'dimension': args.dimension,
                    'layers': args.layers, 'num_adducts': len(cache.manifest['adducts']), 'dropout': .15}
    if args.dimension % 6:
        raise ValueError('dimension must be divisible by 6 attention heads.')
    dataset = PeakTrainingData(cache, aux, config, held, args.profile,
                               args.hard_negatives, args.extra_views)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate_peaks,
                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=generator)
    model = PeakSetNet(model_config).to(device)
    frequency = np.zeros(model_config['bits']+MACCS_BITS+PATH_BITS, dtype=np.float64)
    formula_sum = np.zeros(len(ELEMENTS), dtype=np.float64)
    for start in range(0, len(dataset.sids), 4096):
        selected = dataset.sids[start:start+4096]
        frequency += aux.targets(selected).sum(0)
        formula_sum += (np.log1p(aux.counts[selected])/4.0).sum(0)
    frequency = np.clip(frequency/len(dataset.sids), 1e-4, 1-1e-4)
    with torch.no_grad():
        model.fingerprint.bias.copy_(torch.from_numpy(np.log(frequency/(1-frequency))).float().to(device))
        formula_mean = np.maximum(formula_sum/len(dataset.sids), 1e-3)
        model.formula.bias.copy_(torch.from_numpy(np.log(np.expm1(formula_mean))).float().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    total_steps = len(loader)*args.epochs
    warmup = min(500, max(1, total_steps//10))

    def rate(step):
        if step < warmup:
            return (step+1)/warmup
        progress = min(1.0, (step-warmup)/max(total_steps-warmup, 1))
        return .5*(1+math.cos(math.pi*progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    dest = Path(args.output_dir or Path(config['work_dir']) / ('models/peak_full' if args.full else 'models/peak_cv'))
    dest.mkdir(parents=True, exist_ok=True)
    info = {'full': args.full, 'held_keys': [] if args.full else split['held_keys'],
            'split_sha256': None if args.full else file_digest(split_path), 'seed': seed,
            'profile': args.profile, 'training_structures': len(dataset.sids), 'partial_cache': cache.manifest['partial_cache']}
    training_config = {k: getattr(args, k) for k in ['epochs', 'extra_views', 'batch_size', 'profile',
                       'hard_negatives', 'learning_rate', 'validation_molecules']}
    epoch_start, best, history = 0, -1.0, []
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key, value in [('model_config', model_config), ('training_config', training_config), ('training_info', info)]:
            if previous[key] != value:
                raise ValueError(f'Resume differs in {key}; retain original arguments.')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        epoch_start, best, history = previous['epoch'], previous['best_mrr'], previous['history']
        torch.set_rng_state(previous['torch_rng'])
        generator.set_state(previous['loader_rng'])
        rng = previous['numpy_rng']
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        random.setstate(previous['python_rng'])
        if previous['cuda_rng'] is not None and device.type == 'cuda':
            torch.cuda.set_rng_state_all(previous['cuda_rng'])
    elif (dest / 'last.pt').exists():
        raise FileExistsError('Choose a new output directory or use --resume.')
    print('Explicit training:', {'device': str(device), 'parameters': sum(p.numel() for p in model.parameters()), **training_config})
    for epoch in range(epoch_start, args.epochs):
        started, total, seen = time.monotonic(), 0.0, 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for batch in tqdm(loader, desc=f'peak epoch {epoch+1}/{args.epochs}'):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = peak_loss(model, model(batch), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach()) * len(batch['sid'])
            seen += len(batch['sid'])
        validation = None if args.full else validation_rank(model, config, cache, pool, split, info, device, args.validation_molecules)
        improved = validation is not None and validation['mrr_at_25'] > best
        if improved:
            best = validation['mrr_at_25']
        record = {'epoch': epoch+1, 'loss': total/max(seen, 1), 'validation': validation,
                  'seconds': time.monotonic()-started,
                  'cuda_peak_allocated_gb': torch.cuda.max_memory_allocated(device)/2**30 if amp else 0.0}
        history.append(record)
        rng = np.random.get_state()
        payload = {'kind': 'casmi-peak-set', 'version': VERSION, 'model': model.state_dict(),
            'model_config': model_config, 'training_config': training_config, 'training_info': info,
            **{k: cache.manifest[k] for k in ['rdkit', 'adducts', 'preprocess', 'fingerprint']},
            'epoch': epoch+1, 'best_mrr': best, 'history': history,
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if amp else None,
            'loader_rng': generator.get_state(), 'python_rng': random.getstate(),
            'numpy_rng': [rng[0], rng[1].tolist(), int(rng[2]), int(rng[3]), float(rng[4])]}
        torch.save(payload, dest / 'last.pt.tmp')
        (dest / 'last.pt.tmp').replace(dest / 'last.pt')
        if improved:
            torch.save(payload, dest / 'best.pt')
        if args.full:
            torch.save(payload, dest / 'final.pt')
        write_json(dest / 'history.json', history)
        print(json.dumps(record, ensure_ascii=False))
    print('Training output:', dest)


def resolve_models(config, args):
    path = Path(config['work_dir']) / 'diversity/settings.json'
    settings = read_json(path) if path.exists() else {}
    work = Path(config['work_dir'])
    bags = args.checkpoint if args.checkpoint is not None else [str(work / p) for p in settings.get('checkpoints', [])]
    peaks = args.peak_checkpoints if args.peak_checkpoints is not None else [str(work / p) for p in settings.get('peak_checkpoints', [])]
    fusion = args.fusion or (str(work / settings['fusion']) if settings.get('fusion') else None)
    return bags, peaks, fusion


def run_predict(config, args, validation=False):
    from .metric import check_submission, evaluate_submission
    started = time.monotonic()
    cache, pool = open_pool(config)
    bags, peaks, fusion = resolve_models(config, args)
    split, truth, truth_map, blocked_sids, blocked_rows = None, None, None, [], []
    if validation:
        split_path = Path(args.split or Path(config['work_dir']) / 'validation/split.json')
        split = load_split(split_path, cache)
        if args.scenario == 'library' and split['kind'] != 'np-panel':
            raise ValueError('library scenario requires the existing np-panel split.')
        for path in bags + peaks:
            payload = torch.load(path, map_location='cpu', weights_only=True)
            info = payload['training_info']
            if info['full'] or not set(split['held_keys']).issubset(set(info['held_keys'])):
                raise ValueError(f'{path} has seen held identities. Use a CV model trained with this split.')
        blocked_sids = split['held_sids'] if args.scenario == 'no-reference' else []
        blocked_rows = split['blocked_rows']
        frame = pd.read_parquet(split_path.parent / 'queries.parquet')
        truth = pd.read_csv(split_path.parent / 'truth.csv', dtype=str)
        truth_map = dict(zip(truth['molecule_id'], [scorer_key(s) for s in truth['smiles']]))
        output = Path(args.output or split_path.parent / f'diversity_{args.scenario}.csv')
        feature_path = args.save_features or str(output.with_suffix('.features.parquet'))
    else:
        if cache.manifest['partial_cache']:
            raise ValueError('Submission requires the existing full reference cache.')
        frame = pd.read_parquet(args.test_path or Path(config['data_dir']) / 'test.parquet')
        output = Path(args.output or 'submission.csv')
        feature_path = args.save_features
    frame['molecule_id'] = frame['molecule_id'].astype(str)
    expected = frame['molecule_id'].drop_duplicates().tolist()
    ranker = DiverseRanker(config, cache, pool, bags, peaks, args.device, fusion, blocked_sids, blocked_rows)
    grouped = frame.groupby('molecule_id', sort=False)
    submissions, details, all_features, diagnostics = [], [], [], []
    for molecule in tqdm(expected, desc='diversity molecule inference'):
        ranked, diagnostic = ranker.rank(grouped.get_group(molecule))
        ranked.insert(0, 'molecule_id', molecule)
        seen_keys = set()
        selected_rows = []
        for idx, smiles in ranked['smiles'].items():
            key = scorer_key(smiles)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected_rows.append(idx)
            if len(selected_rows) == 25:
                break
        selected = ranked.loc[selected_rows].copy()
        selected.insert(1, 'rank', np.arange(1, len(selected)+1))
        submissions.append({'molecule_id': molecule, 'smiles': ';'.join(selected['smiles'])})
        details.append(selected)
        if truth_map is not None:
            ranked['truth_key'] = truth_map[molecule]
            metric_keys = ranked['smiles'].map(scorer_key)
            ranked['correct'] = metric_keys == truth_map[molecule]
            ranked = ranked.loc[~metric_keys.duplicated()].copy()
            diagnostic['candidate_covered'] = bool(ranked['correct'].any())
        if feature_path:
            all_features.append(ranked)
        diagnostics.append({'molecule_id': molecule, **diagnostic})
    submission = pd.DataFrame(submissions)
    check_submission(submission, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    pd.concat(details, ignore_index=True).to_csv(output.with_suffix('.candidates.csv'), index=False)
    summary = {'version': VERSION, 'seconds': time.monotonic()-started, 'molecules': len(submission),
               'spectra': len(frame), 'checkpoints': bags, 'peak_checkpoints': peaks,
               'fusion': fusion, 'queries': diagnostics}
    write_json(output.with_suffix('.diagnostics.json'), summary)
    features = pd.concat(all_features, ignore_index=True) if all_features else None
    if feature_path:
        path = Path(feature_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(path, index=False)
        write_json(path.with_suffix('.meta.json'), {'version': VERSION, 'features': FEATURES,
            'validation': validation, 'leakage_checked': validation, 'scenario': args.scenario if validation else None,
            'split_kind': split['kind'] if validation else None, 'checkpoints': bags, 'peak_checkpoints': peaks})
    if validation:
        report = evaluate_submission(submission, truth)
        report.update(candidate_coverage=float(np.mean([d['candidate_covered'] for d in diagnostics])),
                      scenario=args.scenario, split_kind=split['kind'])
        report['single_engine_mrr'] = {}
        for engine in FEATURES:
            if np.ptp(features[engine]) > 1e-9:
                report['single_engine_mrr'][engine] = evaluate_feature_groups(features, engine)
        write_json(output.with_suffix('.metrics.json'), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Saved {output}; {len(submission)} molecules; {time.monotonic()-started:.1f}s')
    return submission


def reciprocal(correct, score):
    order = np.argsort(-score, kind='stable')
    positions = np.flatnonzero(correct[order])
    return 1.0/(positions[0]+1) if len(positions) and positions[0] < 25 else 0.0


def evaluate_feature_groups(frame, name):
    scores = []
    for _, group in frame.groupby('molecule_id', sort=False):
        group = group.sort_values(['error_ppm', 'key'], kind='stable')
        scores.append(reciprocal(group['correct'].to_numpy(bool), group[name].to_numpy()))
    return float(np.mean(scores))


def fit_fusion(config, args):
    frames = []
    tune_strong = False
    for i, path in enumerate(args.features):
        meta = read_json(Path(path).with_suffix('.meta.json'))
        if not meta['validation'] or not meta['leakage_checked'] or meta['features'] != FEATURES:
            raise ValueError('Fusion requires features produced by diversity validate, with unseen-identity models.')
        tune_strong |= meta['scenario'] == 'library'
        frame = pd.read_parquet(path)
        frame['group'] = str(i) + ':' + frame['molecule_id']
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    identities = sorted(frame['truth_key'].unique())
    if len(identities) < 5:
        raise ValueError('Fusion needs at least five validation identities.')
    rng = np.random.default_rng(args.seed)
    rng.shuffle(identities)
    boundary = min(len(identities)-1, max(1, int(len(identities)*.8)))
    calibration = set(identities[:boundary])
    train_groups, audit_groups = [], []
    for _, group in frame.groupby('group', sort=False):
        group = group.sort_values(['error_ppm', 'key'], kind='stable')
        arrays = (group[FEATURES].to_numpy(np.float32), float(group['gate'].iloc[0]), group['correct'].to_numpy(bool))
        (train_groups if group['truth_key'].iloc[0] in calibration else audit_groups).append(arrays)

    def objective(groups, weak, strong):
        results = []
        for features, gate, correct in groups:
            active = np.any(features != 0, axis=0)
            weights = ((1-gate)*weak + gate*strong) * active
            results.append(reciprocal(correct, features @ (weights/max(weights.sum(), 1e-8))))
        return float(np.mean(results))

    best_weak, best_strong = WEAK.copy(), STRONG.copy()
    baseline_fit = objective(train_groups, best_weak, best_strong)
    baseline_audit = objective(audit_groups, best_weak, best_strong)
    best = baseline_fit
    for i in tqdm(range(args.trials), desc='explicit fusion calibration'):
        concentration = 6 if i % 3 == 0 else 30
        weak = rng.dirichlet(WEAK*concentration+.15).astype(np.float32)
        strong = rng.dirichlet(STRONG*concentration+.15).astype(np.float32) if tune_strong else STRONG.copy()
        score = objective(train_groups, weak, strong)
        if score > best:
            best, best_weak, best_strong = score, weak, strong
    audit = objective(audit_groups, best_weak, best_strong)
    dest = Path(args.output or Path(config['work_dir']) / 'diversity/fusion.json')
    report = {'version': VERSION, 'features': FEATURES, 'weak': best_weak.tolist(), 'strong': best_strong.tolist(),
        'seed': args.seed, 'trials': args.trials, 'calibration_identities': len(calibration),
        'audit_identities': len(identities)-len(calibration), 'calibration_keys': sorted(calibration),
        'audit_keys': sorted(set(identities)-calibration), 'baseline_calibration_mrr': baseline_fit,
        'baseline_audit_mrr': baseline_audit, 'calibrated_mrr': best, 'audit_mrr': audit,
        'strong_weights_calibrated': bool(tune_strong)}
    # Audit labels never select weights. If audit loses, keep the default fusion for submission.
    write_json(dest, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Saved', dest, '; compare audit_mrr with baseline_audit_mrr before choosing this fusion.')


def fuse_submissions(config, args):
    """Fuse independent engines, including candidates generated outside this catalog."""
    frames = [pd.read_csv(p, dtype=str) for p in args.inputs]
    query = pd.read_parquet(args.test_path or Path(config['data_dir']) / 'test.parquet')
    query['molecule_id'] = query['molecule_id'].astype(str)
    expected = query['molecule_id'].drop_duplicates().tolist()
    for path, frame in zip(args.inputs, frames):
        if frame['molecule_id'].duplicated().any() or set(frame['molecule_id']) != set(expected):
            raise ValueError(f'{path}: predictions must be regenerated for this exact current test file.')
    weights = args.weights or [1.0]*len(frames)
    if len(weights) != len(frames) or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Supply one nonnegative weight per input, with positive total weight.')
    lookups = [f.set_index('molecule_id')['smiles'].to_dict() for f in frames]
    grouped = query.groupby('molecule_id', sort=False)
    output = []
    for molecule in tqdm(expected, desc='independent engine RRF'):
        rows = grouped.get_group(molecule)
        masses = [neutral_mass(r.precursor_mz, r.adduct) for r in rows.itertuples()]
        mass = float(np.median(masses))
        combined = {}
        for engine, (lookup, weight) in enumerate(zip(lookups, weights)):
            seen = set()
            for rank, smiles in enumerate(str(lookup[molecule]).split(';'), 1):
                try:
                    record, _ = describe_structure(smiles, **config['fingerprint'])
                except (ValueError, RuntimeError, KeyError):
                    continue
                if record['key'] in seen:
                    continue
                seen.add(record['key'])
                if abs(record['mass']-mass) > max(mass*50e-6, .02):
                    continue
                item = combined.setdefault(record['key'], {'score': 0.0, 'tie': (engine, rank), 'smiles': record['smiles']})
                item['score'] += weight/(args.rrf_k+rank)
                item['tie'] = min(item['tie'], (engine, rank))
        if not combined:
            raise ValueError(f'{molecule}: no mass-compatible valid candidate in the input engines.')
        ranked = sorted(combined.values(), key=lambda r: (-r['score'], r['tie'], r['smiles']))[:25]
        output.append({'molecule_id': molecule, 'smiles': ';'.join(r['smiles'] for r in ranked)})
    pd.DataFrame(output).to_csv(args.output, index=False)
    print('Saved independent-engine fusion:', args.output)


def export(config, args):
    from .cli import export_assets
    cache, pool = open_pool(config)
    bags, peaks = args.checkpoint or [], args.peak_checkpoints or []
    # Export only model inference state, not optimizer/epoch history.
    for path in bags + peaks:
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if not payload['training_info']['full'] or payload['training_info']['partial_cache']:
            raise ValueError(f'{path}: use the full-data final checkpoint for official export.')
        for key in ['rdkit', 'adducts', 'preprocess', 'fingerprint']:
            if payload[key] != cache.manifest[key]:
                raise ValueError(f'{path}: {key} does not match cache.')
    if peaks:
        AuxiliaryPool(config, cache, pool)
    dest = Path(args.dest)
    export_assets(config, dest)
    (dest / 'models').mkdir(exist_ok=True)
    settings = {'version': VERSION, 'checkpoints': [], 'peak_checkpoints': [], 'fusion': None}
    for family, paths in [('checkpoints', bags), ('peak_checkpoints', peaks)]:
        for i, path in enumerate(paths):
            payload = torch.load(path, map_location='cpu', weights_only=True)
            kept = {k: payload[k] for k in ['model', 'model_config', 'training_info', 'rdkit', 'adducts', 'preprocess', 'fingerprint']}
            if family == 'checkpoints':
                kept['format_version'] = payload['format_version']
            else:
                kept.update(kind=payload['kind'], version=payload['version'])
            name = f'models/{"bag" if family == "checkpoints" else "peak"}_{i}.pt'
            torch.save(kept, dest / name)
            settings[family].append(name)
    directory = dest / 'diversity'
    directory.mkdir(exist_ok=True)
    if peaks:
        for name in ['manifest.json', 'maccs.npy', 'paths.npy', 'counts.npy']:
            shutil.copy2(Path(config['work_dir']) / 'diversity' / name, directory / name)
    if args.fusion:
        shutil.copy2(args.fusion, directory / 'fusion.json')
        settings['fusion'] = 'diversity/fusion.json'
    write_json(directory / 'settings.json', settings)
    metadata = read_json(dest / 'assets.json')
    metadata.update(engine='diversity', model=bool(bags or peaks),
                    bag_models=len(bags), peak_models=len(peaks))
    write_json(dest / 'assets.json', metadata)
    print('Diversity inference assets ready:', dest)


def parser():
    p = argparse.ArgumentParser(description='Compatible diverse CASMI engines. Training is always explicit.')
    p.add_argument('--config', default=str(ROOT / 'configs/local.yaml'))
    p.add_argument('--data-dir')
    p.add_argument('--work-dir')
    sub = p.add_subparsers(dest='command', required=True)
    item = sub.add_parser('prepare', help='Compute MACCS/path/formula descriptors, no model training')
    item.add_argument('--workers', type=int, default=4)
    item = sub.add_parser('import-candidates', help='Append public CSV/Parquet/SDF/ZIP/SMI structures')
    item.add_argument('--path', nargs='+', required=True)
    item.add_argument('--smiles-column')
    item.add_argument('--name', default='external')
    item.add_argument('--min-mass', type=float, default=100)
    item.add_argument('--max-mass', type=float, default=1400)
    item.add_argument('--workers', type=int, default=4)
    item = sub.add_parser('train', help='ONLY this command fits the new peak-set model')
    item.add_argument('--split')
    item.add_argument('--full', action='store_true')
    item.add_argument('--device', default='cuda')
    item.add_argument('--batch-size', type=int, default=48)
    item.add_argument('--epochs', type=int, default=12)
    item.add_argument('--extra-views', type=float, default=.5)
    item.add_argument('--profile', choices=['balanced', 'np'], default='balanced')
    item.add_argument('--seed', type=int)
    item.add_argument('--dimension', type=int, default=192)
    item.add_argument('--layers', type=int, default=4)
    item.add_argument('--workers', type=int, default=2)
    item.add_argument('--hard-negatives', type=int, default=4)
    item.add_argument('--learning-rate', type=float, default=3e-4)
    item.add_argument('--validation-molecules', type=int, default=100)
    item.add_argument('--output-dir')
    item.add_argument('--resume')
    for name in ['predict', 'validate', 'export']:
        item = sub.add_parser(name)
        item.add_argument('--checkpoint', action='append', help='Repeat this flag for multiple existing fingerprint models')
        item.add_argument('--peak-checkpoints', nargs='+', help='One or more new peak-set models')
        item.add_argument('--fusion')
        if name == 'export':
            item.add_argument('--dest', required=True)
        else:
            item.add_argument('--device', default='auto')
            item.add_argument('--output')
            item.add_argument('--save-features')
            if name == 'predict':
                item.add_argument('--test-path')
            else:
                item.add_argument('--split')
                item.add_argument('--scenario', choices=['no-reference', 'library'], default='no-reference')
    item = sub.add_parser('fit-fusion', help='Explicit weight calibration; reports a separate identity holdout')
    item.add_argument('--features', nargs='+', required=True)
    item.add_argument('--trials', type=int, default=500)
    item.add_argument('--seed', type=int, default=20261012)
    item.add_argument('--output')
    item = sub.add_parser('fuse-submissions', help='Fuse independently rerun engines, including de novo proposals')
    item.add_argument('--inputs', nargs='+', required=True)
    item.add_argument('--weights', nargs='+', type=float)
    item.add_argument('--rrf-k', type=float, default=5)
    item.add_argument('--test-path')
    item.add_argument('--output', default='submission.csv')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config, args.data_dir, args.work_dir)
    if args.command == 'prepare':
        return prepare(config, args.workers)
    if args.command == 'import-candidates':
        return import_catalog(config, args)
    if args.command == 'train':
        return train(config, args)
    if args.command in ['predict', 'validate']:
        return run_predict(config, args, args.command == 'validate')
    if args.command == 'fit-fusion':
        return fit_fusion(config, args)
    if args.command == 'fuse-submissions':
        return fuse_submissions(config, args)
    if args.command == 'export':
        return export(config, args)


if __name__ == '__main__':
    main()
