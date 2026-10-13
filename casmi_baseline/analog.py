"""Mass-shifted spectral analog propagation for the existing CASMI diversity ranker.

This is a separate, optional inference channel.  ``prepare`` builds a compact
inverted peak index from the existing reference cache.  ``validate`` and
``predict`` add evidence from different-mass library structures to the
unchanged diversity ranker.  None of these commands trains a model.
"""

import argparse
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tqdm import tqdm

from . import diversity
from .common import ROOT, load_config, read_json, write_json
from .spectra import cosine_scan


INDEX_VERSION = 1
BIN_WIDTH = 0.1
MAX_MZ = 1500.0
TOP_PEAKS = 24
POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1)


def select_references(cache, per_structure):
    meta = cache.meta
    natural = {i for i, name in enumerate(cache.manifest['libraries'])
               if name in {'riken', 'gnps', 'massbank', 'mona', 'msdial',
                           'enveda-np-examples', 'spectraverse'}}
    selected = []
    for sid in tqdm(range(len(cache.structures)), desc='choose analog references'):
        rows = np.asarray(cache.rows_for_sid(sid), dtype=np.int64)
        rows = rows[meta['n'][rows] >= 6]
        if not len(rows):
            continue
        priority = (2.0 * np.isin(meta['lib'][rows], list(natural))
                    + meta['timstof'][rows].astype(float)
                    + np.minimum(meta['n'][rows], 96) / 120.0)
        ordered = rows[np.argsort(-priority, kind='stable')]
        chosen = []
        for row in ordered:
            if not chosen or (len(chosen) < per_structure and
                              meta['mode'][row] not in meta['mode'][chosen]):
                chosen.append(int(row))
            if len(chosen) == per_structure:
                break
        for row in ordered:
            if len(chosen) == per_structure:
                break
            if int(row) not in chosen:
                chosen.append(int(row))
        selected.extend(chosen)
    return np.asarray(selected, dtype=np.int32)


def prepare(config, per_structure=2):
    cache, _ = diversity.open_pool(config)
    out = Path(config['work_dir']) / 'analog'
    out.mkdir(parents=True, exist_ok=True)
    references = select_references(cache, per_structure)
    parts_bins, parts_rows, parts_weights = [], [], []
    max_bin = int(MAX_MZ / BIN_WIDTH) + 2
    for start in tqdm(range(0, len(references), 2048), desc='index analog peaks'):
        bins, rows, weights = [], [], []
        for ordinal in range(start, min(start + 2048, len(references))):
            ref = int(references[ordinal])
            count = int(cache.meta['n'][ref])
            mz = np.asarray(cache.mzs[ref, :count])
            weight = np.asarray(cache.weights[ref, :count], dtype=np.float32)
            chosen = np.argsort(-weight, kind='stable')[:TOP_PEAKS]
            b = np.floor(mz[chosen] / BIN_WIDTH + .5).astype(np.int32)
            valid = (b >= 450) & (b < max_bin) & (weight[chosen] > 0)
            b, w = b[valid], weight[chosen][valid]
            # One reference contributes only once to a posting list.
            unique, inverse = np.unique(b, return_inverse=True)
            combined = np.zeros(len(unique), dtype=np.float32)
            np.add.at(combined, inverse, w)
            bins.extend(unique)
            rows.extend([ordinal] * len(unique))
            weights.extend(combined)
        parts_bins.append(np.asarray(bins, dtype=np.int32))
        parts_rows.append(np.asarray(rows, dtype=np.int32))
        parts_weights.append(np.asarray(weights, dtype=np.float32))
    bins = np.concatenate(parts_bins) if parts_bins else np.empty(0, dtype=np.int32)
    owners = np.concatenate(parts_rows) if parts_rows else np.empty(0, dtype=np.int32)
    weights = np.concatenate(parts_weights) if parts_weights else np.empty(0, dtype=np.float32)
    order = np.argsort(bins, kind='stable')
    offsets = np.empty(max_bin + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(np.bincount(bins, minlength=max_bin))
    np.save(out / 'references.npy', references)
    np.save(out / 'offsets.npy', offsets)
    np.save(out / 'owners.npy', owners[order])
    np.save(out / 'weights.npy', weights[order].astype(np.float16))
    write_json(out / 'manifest.json', {
        'version': INDEX_VERSION, 'cache_rows': len(cache.meta),
        'structures': len(cache.structures), 'rdkit': cache.manifest['rdkit'],
        'references': len(references), 'postings': len(owners),
        'per_structure': per_structure, 'bin_width': BIN_WIDTH})
    print('Analog index:', out, len(references), 'references,', len(owners), 'postings')


class AnalogIndex:
    def __init__(self, cache, path):
        path = Path(path)
        manifest = read_json(path / 'manifest.json')
        if (manifest['version'] != INDEX_VERSION or manifest['cache_rows'] != len(cache.meta)
                or manifest['structures'] != len(cache.structures)
                or manifest['rdkit'] != cache.manifest['rdkit']):
            raise ValueError('Analog index does not match reference cache; rebuild prepare.')
        self.refs = np.load(path / 'references.npy', mmap_mode='r')
        self.offsets = np.load(path / 'offsets.npy', mmap_mode='r')
        self.owners = np.load(path / 'owners.npy', mmap_mode='r')
        self.weights = np.load(path / 'weights.npy', mmap_mode='r')
        self.sid = cache.meta['sid'][self.refs].astype(np.int32)
        self.mode = cache.meta['mode'][self.refs].astype(np.int8)
        self.masses = cache.structures['mass'].to_numpy(np.float64)[self.sid]
        self.cache = cache

    def shortlist(self, query, blocked_sids, blocked_rows, limit=160):
        mz, weight = query['mz'], query['weights']
        if len(mz) < 3:
            return []
        score = np.zeros(len(self.refs), dtype=np.float32)
        top = np.argsort(-weight, kind='stable')[:TOP_PEAKS]
        bins = np.floor(mz[top] / BIN_WIDTH + .5).astype(np.int32)
        for center, w in zip(bins, weight[top]):
            if center < 450 or center >= len(self.offsets) - 2:
                continue
            for b in (center - 1, center, center + 1):
                lo, hi = int(self.offsets[b]), int(self.offsets[b + 1])
                owners = self.owners[lo:hi]
                # Correctly accumulates repeated query bins across separate calls.
                distinctiveness = math.sqrt(math.log1p(len(self.refs) / (1 + hi - lo))
                                            / max(math.log1p(len(self.refs)), 1))
                score[owners] += float(w) * distinctiveness * self.weights[lo:hi].astype(np.float32)
        delta = np.abs(self.masses - query['mass'])
        score[(delta < .02) | (delta > 250)] = 0
        if query['mode']:
            score[(self.mode != 0) & (self.mode != query['mode'])] = 0
        if blocked_sids:
            score[np.isin(self.sid, list(blocked_sids))] = 0
        score[blocked_rows[self.refs]] = 0
        good = np.flatnonzero(score > .10)
        if len(good) > limit:
            good = good[np.argpartition(score[good], -limit)[-limit:]]
        good = good[np.argsort(-score[good], kind='stable')]
        return good

    def neighbors(self, queries, blocked_sids, blocked_rows, limit=64):
        best = {}
        cache = self.cache
        for query in queries:
            good = self.shortlist(query, blocked_sids, blocked_rows)
            for ordinal in good:
                row = int(self.refs[ordinal])
                sid = int(self.sid[ordinal])
                n = int(cache.meta['n'][row])
                rmz = np.asarray(cache.mzs[row, :n], dtype=np.float32)
                rw = np.asarray(cache.weights[row, :n], dtype=np.float32)
                direct = cosine_scan(query['mz'], query['weights'], rmz, rw, .025, 20.0)
                qloss = query['precursor'] - query['mz'][::-1]
                rloss = float(cache.meta['precursor'][row]) - rmz[::-1]
                loss = cosine_scan(qloss, query['weights'][::-1], rloss, rw[::-1], .025, 20.0)
                # The larger of aligned fragments and mass-shifted fragments drives propagation.
                similarity = float(.85 * max(direct, loss) + .15 * min(direct, loss))
                similarity *= float(query['quality']) ** .25
                if similarity >= .24:
                    best[sid] = max(best.get(sid, 0.0), similarity)
        return sorted(best.items(), key=lambda x: -x[1])[:limit]


def packed_fingerprints(pool, ids):
    ids = np.asarray(ids, dtype=np.int64)
    packed = np.empty((len(ids), pool.fps[0].shape[1]), dtype=np.uint8)
    train = ids < pool.train_size
    packed[train] = pool.fps[0][ids[train]]
    if (~train).any():
        packed[~train] = pool.fps[1][ids[~train] - pool.train_size]
    return packed


def analog_evidence(pool, candidate_ids, neighbors):
    if not neighbors:
        return np.zeros(len(candidate_ids), dtype=np.float32)
    candidate = packed_fingerprints(pool, candidate_ids)
    sizes = POPCOUNT[candidate].sum(axis=1).astype(np.float32)
    result = np.zeros(len(candidate), dtype=np.float32)
    for sid, spectrum_score in neighbors:
        ref = np.asarray(pool.fps[0][sid], dtype=np.uint8)
        ref_size = float(POPCOUNT[ref].sum())
        intersection = POPCOUNT[np.bitwise_and(candidate, ref)].sum(axis=1)
        tanimoto = intersection / np.maximum(sizes + ref_size - intersection, 1)
        np.maximum(result, float(spectrum_score) ** 2 * tanimoto, out=result)
    return result


class AnalogRanker(diversity.DiverseRanker):
    index_dir = None
    weight = .18

    def __init__(self, config, cache, pool, *args, **kwargs):
        super().__init__(config, cache, pool, *args, **kwargs)
        self.analog = AnalogIndex(cache, self.index_dir or Path(config['work_dir']) / 'analog')

    def rank_queries(self, queries):
        ranked, diagnostic = super().rank_queries(queries)
        neighbors = self.analog.neighbors(queries, self.blocked_sids, self.blocked_rows)
        raw = analog_evidence(self.pool, ranked['candidate_id'].to_numpy(), neighbors)
        analog = diversity.rank_column(raw)
        ranked['analog_raw'] = raw
        ranked['analog'] = analog
        # Strong same-mass library hits already have direct evidence.
        gate = float(diagnostic['library_gate'])
        ranked['score'] += self.weight * (1.0 - .75 * gate) * analog
        ranked = ranked.sort_values(['score', 'error_ppm', 'key'],
                                    ascending=[False, True, True], kind='stable')
        diagnostic['analog_neighbors'] = len(neighbors)
        diagnostic['best_analog'] = float(raw.max(initial=0))
        return ranked, diagnostic


def parser():
    p = argparse.ArgumentParser(description='Optional different-mass analog evidence for CASMI diversity')
    p.add_argument('--config', default=str(ROOT / 'configs/local.yaml'))
    p.add_argument('--data-dir')
    p.add_argument('--work-dir')
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--per-structure', type=int, choices=[1, 2, 3], default=2)
    for name in ('validate', 'predict'):
        item = sub.add_parser(name)
        item.add_argument('--checkpoint', action='append')
        item.add_argument('--peak-checkpoints', nargs='+')
        item.add_argument('--fusion')
        item.add_argument('--device', default='auto')
        item.add_argument('--output')
        item.add_argument('--save-features')
        item.add_argument('--index-dir')
        item.add_argument('--analog-weight', type=float, default=.18)
        if name == 'validate':
            item.add_argument('--split')
            item.add_argument('--scenario', choices=['no-reference', 'library'], default='no-reference')
        else:
            item.add_argument('--test-path')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config, args.data_dir, args.work_dir)
    if args.command == 'prepare':
        return prepare(config, args.per_structure)
    if args.analog_weight < 0:
        raise ValueError('--analog-weight must be nonnegative')
    bound = type('BoundAnalogRanker', (AnalogRanker,),
                 {'index_dir': args.index_dir, 'weight': args.analog_weight})
    with patch.object(diversity, 'DiverseRanker', bound):
        return diversity.run_predict(config, args, args.command == 'validate')


if __name__ == '__main__':
    main()
