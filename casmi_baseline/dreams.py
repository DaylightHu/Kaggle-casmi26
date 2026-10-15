"""Frozen official DreaMS embeddings added to the existing analog CASMI engine.

Only ``download`` accesses the network. ``prepare`` reads the original training
Parquet in batches and encodes selected reference spectra; it never trains.
``predict`` and ``validate`` use local TorchScript weights and local embeddings.

The peak preprocessing follows pluskal-lab/DreaMS, whose MIT license is included
below and written alongside the downloaded weights. Positive spectra with
precursor m/z <= 1000 are used in this first version. All other spectra keep the
existing analog/diversity evidence.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from . import analog, diversity
from .cache import ReferenceCache
from .chemistry import check_rdkit, neutral_mass
from .common import ROOT, file_digest, load_config, read_json, write_json
from .fingerprint import choose_device
from .spectra import clean_spectrum, mode_code, query_quality


VERSION = 1
HF_REVISION = 'c81a62766b10dd1d39fcda3edec5ef88623e5f6b'
MODEL_URL = ('https://huggingface.co/roman-bushuiev/DreaMS/resolve/'
             + HF_REVISION + '/DreaMS_embedding_model_torchscript.pt')
MODEL_SHA256 = 'df69866990b194b27d5497e440f87f6ceca568ee512e7c22cb1f278721376f0c'
MODEL_BYTES = 468475544
SETTINGS = {'model_name': 'DreaMS', 'embedding_dim': 1024,
            'prec_intens': 1.1, 'n_highest_peaks': 100}
LICENSE = '''MIT License

Copyright (c) 2024 Roman Bushuiev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''


def download(config):
    """Explicit, resumable download; no download fallback exists in inference."""
    dest = Path(config['work_dir']) / 'dreams'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'encoder.pt'
    if path.exists():
        if file_digest(path) != MODEL_SHA256:
            raise ValueError('Existing dreams/encoder.pt is not the pinned official model.')
    else:
        temporary = dest / 'encoder.pt.part'
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset > MODEL_BYTES:
            raise ValueError('Oversized encoder.pt.part; remove this incomplete download and retry.')
        if offset < MODEL_BYTES:
            headers = {'User-Agent': 'casmi26-dreams/1'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(MODEL_URL, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if not append:
                    offset = 0
                elif not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Download resumed at an unexpected byte offset.')
                with temporary.open('ab' if append else 'wb') as output:
                    with tqdm(total=MODEL_BYTES, initial=offset, unit='B', unit_scale=True,
                              desc='official DreaMS encoder') as progress:
                        while True:
                            block = response.read(4 * 1024 * 1024)
                            if not block:
                                break
                            output.write(block)
                            progress.update(len(block))
        if temporary.stat().st_size != MODEL_BYTES or file_digest(temporary) != MODEL_SHA256:
            raise ValueError('DreaMS download is incomplete or has an unexpected checksum.')
        temporary.replace(path)
    write_json(dest / 'pretrained_settings.json', SETTINGS)
    (dest / 'DreaMS_LICENSE.txt').write_text(LICENSE, encoding='utf-8')
    print('Official frozen encoder ready:', path)


def preprocess(mzs, intensities, precursor, max_precursor=1000.0):
    """Raw m/z, top-100 peaks, base-peak normalization, precursor token, padding."""
    try:
        precursor = float(precursor)
        mz = np.asarray(mzs, dtype=np.float64)
        intensity = np.asarray(intensities, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (not np.isfinite(precursor) or not 0 < precursor <= max_precursor
            or mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity)):
        return None
    valid = (np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
             & (mz <= min(max_precursor, precursor + 2.0)))
    mz, intensity = mz[valid], intensity[valid]
    if len(mz) < 3:
        return None
    # Same top-peak selection/order as the official SpectrumPreprocessor.
    selected = np.sort(np.argsort(intensity)[-SETTINGS['n_highest_peaks']:])
    mz, intensity = mz[selected], intensity[selected]
    intensity = intensity / intensity.max()
    output = np.zeros((SETTINGS['n_highest_peaks'] + 1, 2), dtype=np.float32)
    output[0] = precursor, SETTINGS['prec_intens']
    output[1:len(mz) + 1, 0] = mz
    output[1:len(mz) + 1, 1] = intensity
    return output


class Encoder:
    def __init__(self, path, device='cpu', batch_size=8):
        self.device = choose_device(device)
        self.batch_size = int(batch_size)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        # The author's ScriptModule stores the normalization tensor's device in
        # a non-parameter attribute; map_location alone does not move that value.
        for module in self.model.modules():
            if hasattr(module, '_device'):
                module._device = self.device
        self.dtype = next(self.model.parameters()).dtype

    def encode(self, spectra):
        pieces = []
        with torch.inference_mode():
            for start in range(0, len(spectra), self.batch_size):
                values = np.stack(spectra[start:start + self.batch_size])
                tensor = torch.from_numpy(values).to(device=self.device, dtype=self.dtype)
                # Full precision: official attention fills masked scores with
                # -1e9, which is not representable under FP16 autocast.
                result = self.model(tensor).float().cpu().numpy()
                if result.shape != (len(values), SETTINGS['embedding_dim']):
                    raise ValueError(f'Unexpected DreaMS output shape: {result.shape}')
                norm = np.linalg.norm(result, axis=1, keepdims=True)
                if not np.isfinite(result).all() or np.any(norm <= 1e-12):
                    raise ValueError('DreaMS produced a nonfinite or zero embedding.')
                pieces.append(result / norm)
        return np.concatenate(pieces) if pieces else np.empty((0, SETTINGS['embedding_dim']), np.float32)


def select_references(cache, per_structure, max_precursor):
    natural = {i for i, name in enumerate(cache.manifest['libraries']) if name in
               {'riken', 'gnps', 'massbank', 'mona', 'msdial', 'enveda-np-examples', 'spectraverse'}}
    meta, selected = cache.meta, []
    for sid in tqdm(range(len(cache.structures)), desc='choose DreaMS reference views'):
        rows = np.asarray(cache.rows_for_sid(sid), dtype=np.int64)
        rows = rows[(meta['mode'][rows] == 1) & (meta['n'][rows] >= 3)
                    & (meta['precursor'][rows] > 0) & (meta['precursor'][rows] <= max_precursor)]
        if not len(rows):
            continue
        priority = (2.0 * np.isin(meta['lib'][rows], list(natural))
                    + meta['timstof'][rows] + np.minimum(meta['n'][rows], 100) / 125.0)
        ordered = rows[np.lexsort((rows, -priority))]
        chosen, conditions = [], set()
        for row in ordered:
            energy = float(meta['ce'][row])
            condition = (int(meta['adduct'][row]), round(energy / 10) if np.isfinite(energy) else None)
            if condition not in conditions:
                chosen.append(int(row))
                conditions.add(condition)
            if len(chosen) == per_structure:
                break
        for row in ordered:
            if len(chosen) == per_structure:
                break
            if int(row) not in chosen:
                chosen.append(int(row))
        selected.extend(chosen)
    return np.sort(np.asarray(selected, dtype=np.int64))


def signature(cache):
    return {'cache_manifest_sha256': file_digest(cache.path / 'manifest.json'),
            'training_keys_sha256': hashlib.sha256(
                '\n'.join(cache.structures['key'].astype(str)).encode()).hexdigest()}


def prepare(config, args):
    check_rdkit()
    cache = ReferenceCache(Path(config['work_dir']) / 'cache')
    cache.require_config(config)
    dest = Path(config['work_dir']) / 'dreams'
    model_path = dest / 'encoder.pt'
    if file_digest(model_path) != MODEL_SHA256:
        raise ValueError('Use the pinned official DreaMS TorchScript encoder.')
    train_path = Path(args.train_path or Path(config['data_dir']) / 'train.parquet')
    parquet = pq.ParquetFile(train_path)
    if parquet.metadata.num_rows != len(cache.meta):
        raise ValueError('Original training Parquet row count differs from the reference cache.')
    references = select_references(cache, args.per_structure, args.max_precursor)
    if not len(references):
        raise ValueError('No eligible positive-ion reference spectra were selected.')
    identity = {'version': VERSION, 'model_sha256': MODEL_SHA256, **signature(cache),
                'per_structure': args.per_structure, 'max_precursor': args.max_precursor,
                'reference_rows_sha256': hashlib.sha256(references.tobytes()).hexdigest(),
                'preprocessing': SETTINGS, 'polarity': 'positive', 'raw_training_spectra': True}
    manifest_path = dest / 'manifest.json'
    completed, invalid = 0, 0
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous['identity'] != identity:
            raise ValueError('DreaMS prepare parameters/cache changed; use another asset directory.')
        if previous['complete']:
            print('DreaMS reference embeddings already complete:', dest)
            return
        if not args.resume:
            raise FileExistsError('Incomplete DreaMS prepare exists; rerun the same command with --resume.')
        completed, invalid = previous['completed'], previous['invalid']
        mode = 'r+'
    else:
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'references.npy', references, allow_pickle=False)
        mode = 'w+'
    embeddings = np.lib.format.open_memmap(dest / 'embeddings.npy', mode=mode, dtype=np.float16,
                                         shape=(len(references), SETTINGS['embedding_dim']))
    valid_refs = np.lib.format.open_memmap(dest / 'valid.npy', mode=mode, dtype=bool, shape=(len(references),))
    if mode == 'w+':
        valid_refs[:] = False

    def checkpoint(position, complete=False):
        embeddings.flush()
        valid_refs.flush()
        write_json(manifest_path, {'identity': identity, 'references': len(references),
                   'completed': int(position), 'invalid': int(invalid), 'complete': complete,
                   'source_train': str(train_path.resolve()), 'source_model': MODEL_URL,
                   'torch': torch.__version__})

    checkpoint(completed)
    encoder = Encoder(model_path, args.device, args.batch_size)
    cursor = 0
    with tqdm(total=len(references), initial=completed, desc='encode DreaMS reference spectra') as progress:
        for batch in parquet.iter_batches(batch_size=2048, columns=[
                'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']):
            stop = cursor + len(batch)
            left = max(completed, int(np.searchsorted(references, cursor)))
            right = int(np.searchsorted(references, stop))
            if right <= left:
                cursor = stop
                continue
            rows = references[left:right]
            values = batch.take(rows - cursor).to_pydict()
            ordinals, spectra = [], []

            def flush():
                if ordinals:
                    vectors = encoder.encode(spectra)
                    embeddings[ordinals] = vectors.astype(np.float16)
                    valid_refs[ordinals] = True
                    progress.update(len(ordinals))
                    ordinals.clear()
                    spectra.clear()

            for local, ordinal in enumerate(range(left, right)):
                precursor = float(values['precursor_mz'][local])
                cached_precursor = float(cache.meta['precursor'][references[ordinal]])
                if not np.isclose(precursor, cached_precursor, rtol=1e-7, atol=1e-6):
                    raise ValueError('Training Parquet spectra are not in the original cache row order.')
                spectrum = preprocess(values['ms2_mzs'][local], values['ms2_normalized_intensities'][local],
                                      precursor, args.max_precursor)
                if spectrum is None:
                    embeddings[ordinal] = 0
                    valid_refs[ordinal] = False
                    invalid += 1
                    progress.update(1)
                else:
                    ordinals.append(ordinal)
                    spectra.append(spectrum)
                    if len(ordinals) == args.batch_size:
                        flush()
            flush()
            completed = right
            checkpoint(completed)
            cursor = stop
            if completed == len(references):
                break
    if completed != len(references):
        raise ValueError('Training Parquet ended before all reference embeddings were prepared.')
    checkpoint(completed, complete=True)
    print('DreaMS reference cache ready:', dest, len(references), 'views;', invalid, 'unusable views')


class EmbeddingIndex:
    def __init__(self, cache, path, blocked_sids, blocked_rows, scan_batch=16384):
        path = Path(path)
        self.manifest = read_json(path / 'manifest.json')
        identity = self.manifest['identity']
        if not self.manifest['complete'] or identity['version'] != VERSION:
            raise ValueError('DreaMS prepare is incomplete or has an incompatible version.')
        for name, value in signature(cache).items():
            if identity[name] != value:
                raise ValueError(f'DreaMS reference cache differs in {name}.')
        if identity['model_sha256'] != MODEL_SHA256:
            raise ValueError('DreaMS reference embeddings were made by a different encoder.')
        self.refs = np.load(path / 'references.npy', mmap_mode='r', allow_pickle=False)
        self.embeddings = np.load(path / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        valid = np.load(path / 'valid.npy', mmap_mode='r', allow_pickle=False)
        if self.embeddings.shape != (len(self.refs), SETTINGS['embedding_dim']) or len(valid) != len(self.refs):
            raise ValueError('DreaMS reference arrays are not aligned.')
        if hashlib.sha256(self.refs.tobytes()).hexdigest() != identity['reference_rows_sha256']:
            raise ValueError('DreaMS reference-row mapping changed.')
        self.sids = cache.meta['sid'][self.refs].astype(np.int32)
        self.masses = cache.structures['mass'].to_numpy(np.float64)[self.sids]
        self.allowed = np.asarray(valid, dtype=bool).copy()
        if blocked_sids:
            self.allowed &= ~np.isin(self.sids, list(blocked_sids))
        self.allowed &= ~np.asarray(blocked_rows, dtype=bool)[self.refs]
        self.per_structure = identity['per_structure']
        self.max_precursor = identity['max_precursor']
        self.scan_batch = scan_batch
        self.train_size = len(cache.structures)

    def search(self, vectors, records, candidate_ids, mass, neighbor_limit, min_similarity):
        direct = np.zeros(len(candidate_ids), dtype=np.float32)
        if not len(vectors):
            return direct, []
        columns = np.full(self.train_size, -1, dtype=np.int32)
        train = candidate_ids < self.train_size
        columns[candidate_ids[train]] = np.flatnonzero(train)
        kept_scores, kept_rows = np.empty(0, np.float32), np.empty(0, np.int64)
        capacity = neighbor_limit * self.per_structure
        for start in range(0, len(self.refs), self.scan_batch):
            stop = min(start + self.scan_batch, len(self.refs))
            good = np.flatnonzero(self.allowed[start:stop]) + start
            if not len(good):
                continue
            block = self.embeddings[good].astype(np.float32)
            # Torch uses the CLI's thread count even when NumPy is linked to a
            # BLAS implementation with a different default thread count.
            cosine = np.clip((torch.from_numpy(block) @ torch.from_numpy(vectors.T)).numpy(), 0.0, 1.0)
            score = diversity.aggregate_views(cosine.T, records)
            column = columns[self.sids[good]]
            same_mass = (column >= 0) & (score >= min_similarity)
            np.maximum.at(direct, column[same_mass], score[same_mass])
            delta = np.abs(self.masses[good] - mass)
            eligible = (delta >= .02) & (delta <= 250.0) & (score >= min_similarity)
            scores, rows = score[eligible], good[eligible]
            if len(rows) > capacity:
                take = np.lexsort((rows, -scores))[:capacity]
                scores, rows = scores[take], rows[take]
            kept_scores = np.r_[kept_scores, scores]
            kept_rows = np.r_[kept_rows, rows]
            if len(kept_rows) > capacity:
                take = np.lexsort((kept_rows, -kept_scores))[:capacity]
                kept_scores, kept_rows = kept_scores[take], kept_rows[take]
        by_sid = {}
        for row, score in zip(kept_rows, kept_scores):
            sid = int(self.sids[row])
            by_sid[sid] = max(by_sid.get(sid, 0.0), float(score))
        neighbors = sorted(by_sid.items(), key=lambda item: (-item[1], item[0]))[:neighbor_limit]
        return direct, neighbors


class DreamsRanker(analog.AnalogRanker):
    dreams_path = None
    direct_weight = .10
    propagation_weight = .10
    min_similarity = .60
    neighbor_limit = 64
    encoder_batch = 8
    scan_batch = 16384

    def __init__(self, config, cache, pool, checkpoints=(), peak_checkpoints=(), device='auto',
                 fusion=None, blocked_sids=None, blocked_rows=None, peak_predictors=None, use_library=True):
        super().__init__(config, cache, pool, checkpoints, peak_checkpoints, device,
                         fusion, blocked_sids, blocked_rows, peak_predictors, use_library)
        self.raw_frame = None
        self.dreams_index = None
        if self.direct_weight or self.propagation_weight:
            path = Path(self.dreams_path or Path(config['work_dir']) / 'dreams')
            if file_digest(path / 'encoder.pt') != MODEL_SHA256:
                raise ValueError('Local DreaMS encoder differs from the official prepared weights.')
            self.dreams_index = EmbeddingIndex(cache, path, self.blocked_sids,
                                               self.blocked_rows, self.scan_batch)
            self.encoder = Encoder(path / 'encoder.pt', device, self.encoder_batch)

    def rank(self, frame):
        self.raw_frame = frame
        try:
            return super().rank(frame)
        finally:
            self.raw_frame = None

    def query_embeddings(self):
        spectra, records = [], []
        if self.raw_frame is None:
            return np.empty((0, SETTINGS['embedding_dim']), np.float32), records
        for row in self.raw_frame.to_dict('records'):
            if mode_code(row.get('ionization_mode')) != 1:
                continue
            spectrum = preprocess(row.get('ms2_mzs'), row.get('ms2_normalized_intensities'),
                                  row.get('precursor_mz'), self.dreams_index.max_precursor)
            if spectrum is None:
                continue
            try:
                neutral_mass(row['precursor_mz'], row['adduct'])
                mz, weights = clean_spectrum(row['ms2_mzs'], row['ms2_normalized_intensities'],
                                            row['precursor_mz'], self.config['preprocess'])
            except (ValueError, TypeError, KeyError):
                continue
            spectra.append(spectrum)
            records.append({'adduct': self.cache.adduct_map.get(str(row['adduct']), 0), 'mode': 1,
                            'quality': query_quality(mz, weights, row.get('base_peak_intensity'))})
        return self.encoder.encode(spectra), records

    def rank_queries(self, queries):
        ranked, diagnostic = super().rank_queries(queries)
        direct = np.zeros(len(ranked), dtype=np.float32)
        propagated = direct.copy()
        neighbors, records = [], []
        if self.dreams_index is not None:
            vectors, records = self.query_embeddings()
            ids = ranked['candidate_id'].to_numpy(np.int64)
            direct, neighbors = self.dreams_index.search(vectors, records, ids,
                diagnostic['neutral_mass'], self.neighbor_limit, self.min_similarity)
            propagated = analog.analog_evidence(self.pool, ids, neighbors)
        strength = np.clip((direct - self.min_similarity) / (1.0 - self.min_similarity), 0, 1)
        direct_feature = .65 * strength + .35 * diversity.rank_column(strength)
        best_neighbor = max((score for _, score in neighbors), default=0.0)
        confidence = max(0.0, (best_neighbor - self.min_similarity) / (1.0 - self.min_similarity))
        propagation_feature = .65 * propagated + .35 * confidence * diversity.rank_column(propagated)
        ranked['dreams_direct_raw'], ranked['dreams_direct'] = direct, direct_feature
        ranked['dreams_propagation_raw'], ranked['dreams_propagation'] = propagated, propagation_feature
        gate = 1.0 - .75 * float(diagnostic['library_gate'])
        ranked['score'] += gate * (self.direct_weight * direct_feature
                                    + self.propagation_weight * propagation_feature)
        ranked = ranked.sort_values(['score', 'error_ppm', 'key'],
                                    ascending=[False, True, True], kind='stable')
        diagnostic.update(dreams_spectra=len(records), dreams_neighbors=len(neighbors),
                          best_dreams_direct=float(direct.max(initial=0)), best_dreams_neighbor=best_neighbor)
        return ranked, diagnostic


def parser():
    p = argparse.ArgumentParser(description='Frozen official DreaMS evidence for the existing analog engine')
    p.add_argument('--config', default=str(ROOT / 'configs/local.yaml'))
    p.add_argument('--data-dir')
    p.add_argument('--work-dir')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('download', help='Explicitly download only the official TorchScript encoder')
    item = sub.add_parser('prepare', help='Encode original reference spectra; no training')
    item.add_argument('--train-path')
    item.add_argument('--device', default='cuda')
    item.add_argument('--batch-size', type=int, default=32)
    item.add_argument('--per-structure', type=int, choices=[1, 2, 3], default=2)
    item.add_argument('--max-precursor', type=float, default=1000.0)
    item.add_argument('--resume', action='store_true')
    item.add_argument('--threads', type=int, default=2)
    for name in ('predict', 'validate'):
        item = sub.add_parser(name)
        item.add_argument('--checkpoint', action='append')
        item.add_argument('--peak-checkpoints', nargs='+')
        item.add_argument('--fusion')
        item.add_argument('--device', default='auto')
        item.add_argument('--output')
        item.add_argument('--save-features')
        item.add_argument('--index-dir')
        item.add_argument('--analog-weight', type=float, default=.18)
        item.add_argument('--dreams-dir')
        item.add_argument('--direct-weight', type=float, default=.10)
        item.add_argument('--propagation-weight', type=float, default=.10)
        item.add_argument('--min-similarity', type=float, default=.60)
        item.add_argument('--neighbors', type=int, default=64)
        item.add_argument('--batch-size', type=int, default=8)
        item.add_argument('--scan-batch', type=int, default=16384)
        item.add_argument('--threads', type=int, default=2)
        if name == 'predict':
            item.add_argument('--test-path')
        else:
            item.add_argument('--split')
            item.add_argument('--scenario', choices=['no-reference', 'library'], default='no-reference')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config, args.data_dir, args.work_dir)
    if args.command == 'download':
        return download(config)
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError('Batch size and thread count must be positive.')
    torch.set_num_threads(args.threads)
    if args.command == 'prepare':
        if not 0 < args.max_precursor <= 1000:
            raise ValueError('This version uses the official <=1000 m/z support range.')
        return prepare(config, args)
    if (min(args.analog_weight, args.direct_weight, args.propagation_weight) < 0
            or not 0 <= args.min_similarity < 1 or args.neighbors < 1 or args.scan_batch < 1):
        raise ValueError('Invalid DreaMS fusion/search parameter.')
    bound = type('BoundDreamsRanker', (DreamsRanker,), {
        'index_dir': args.index_dir, 'weight': args.analog_weight,
        'dreams_path': args.dreams_dir, 'direct_weight': args.direct_weight,
        'propagation_weight': args.propagation_weight, 'min_similarity': args.min_similarity,
        'neighbor_limit': args.neighbors, 'encoder_batch': args.batch_size, 'scan_batch': args.scan_batch})
    with patch.object(diversity, 'DiverseRanker', bound):
        result = diversity.run_predict(config, args, args.command == 'validate')
    if args.output:
        path = Path(args.output).with_suffix('.diagnostics.json')
        metadata = read_json(path)
        metadata.update(engine='dreams-analog', dreams={
            'version': VERSION, 'model_sha256': MODEL_SHA256, 'source_model': MODEL_URL,
            'direct_weight': args.direct_weight, 'propagation_weight': args.propagation_weight,
            'min_similarity': args.min_similarity, 'neighbors': args.neighbors,
            'scope': 'positive ionization, precursor m/z <= 1000'})
        write_json(path, metadata)
    return result


if __name__ == '__main__':
    main()
