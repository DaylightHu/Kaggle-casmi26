"""Supervised fingerprint head on frozen DreaMS embeddings.

``train`` is the only command that optimizes parameters.  The official DreaMS
encoder stays frozen.  ``predict`` and ``validate`` add an independent
spectrum-to-structure channel to the existing DreaMS/analog/diversity ranker.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import diversity, dreams
from .common import ROOT, file_digest, load_config, read_json, stable_fraction, write_json
from .fingerprint import choose_device, seed_worker
from .validation import load_split


VERSION = 1
CHANNELS = ('morgan', 'maccs', 'path', 'embedding', 'formula')
CHANNEL_WEIGHTS = np.asarray([.35, .15, .15, .25, .10], dtype=np.float32)


class DreamsFingerprintHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config['hidden']
        bits = config['bits'] + diversity.MACCS_BITS + diversity.PATH_BITS
        self.trunk = nn.Sequential(
            nn.LayerNorm(dreams.SETTINGS['embedding_dim']),
            nn.Linear(dreams.SETTINGS['embedding_dim'], hidden), nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.fingerprint = nn.Linear(hidden, bits)
        self.formula = nn.Linear(hidden, len(diversity.ELEMENTS))
        self.spectrum_projection = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 128))
        self.structure_projection = nn.Sequential(
            nn.Linear(bits + len(diversity.ELEMENTS), 384), nn.GELU(), nn.Linear(384, 128))

    def structure_embedding(self, features):
        return nn.functional.normalize(self.structure_projection(features), dim=-1)

    def forward(self, embedding):
        hidden = self.trunk(embedding)
        return {
            'logits': self.fingerprint(hidden),
            'counts': nn.functional.softplus(self.formula(hidden)),
            'embedding': nn.functional.normalize(self.spectrum_projection(hidden), dim=-1),
        }


class EmbeddingStore:
    def __init__(self, cache, path):
        self.path = Path(path)
        blocked = np.zeros(len(cache.meta), dtype=bool)
        index = dreams.EmbeddingIndex(cache, self.path, set(), blocked)
        self.refs = np.asarray(index.refs)
        self.sids = np.asarray(index.sids)
        self.embeddings = index.embeddings
        self.valid = np.asarray(index.allowed, dtype=bool)
        self.identity = index.manifest['identity']


class HeadTrainingData(Dataset):
    def __init__(self, store, aux, excluded_sids, hard_negatives):
        self.store, self.aux = store, aux
        excluded_sids = np.asarray(excluded_sids, dtype=np.int64)
        keep = store.valid.copy()
        if len(excluded_sids):
            keep &= ~np.isin(store.sids, excluded_sids)
        self.ordinals = np.flatnonzero(keep)
        if not len(self.ordinals):
            raise ValueError('No DreaMS embeddings remain for head training.')
        self.hard_negatives = int(hard_negatives)
        self.mass_order = np.argsort(aux.pool.masses, kind='stable')
        self.sorted_mass = aux.pool.masses[self.mass_order]

    def __len__(self):
        return len(self.ordinals)

    def negatives(self, sid):
        mass = float(self.aux.pool.masses[sid])
        delta = max(mass * 50e-6, .02)
        left = np.searchsorted(self.sorted_mass, mass - delta, side='left')
        right = np.searchsorted(self.sorted_mass, mass + delta, side='right')
        candidates = self.mass_order[left:right]
        candidates = candidates[candidates != sid]
        if not len(candidates):
            pos = int(np.searchsorted(self.sorted_mass, mass))
            candidates = self.mass_order[max(0, pos - 64):min(len(self.mass_order), pos + 64)]
            candidates = candidates[candidates != sid]
        if not len(candidates):
            raise ValueError('At least two candidate structures are required for contrastive training.')
        return np.random.choice(candidates, self.hard_negatives,
                                replace=len(candidates) < self.hard_negatives)

    def __getitem__(self, index):
        ordinal = int(self.ordinals[index])
        sid = int(self.store.sids[ordinal])
        molecule = self.aux.features([sid])[0]
        return {
            'embedding': np.asarray(self.store.embeddings[ordinal], dtype=np.float32),
            'sid': sid,
            'target': molecule[:-len(diversity.ELEMENTS)],
            'counts': molecule[-len(diversity.ELEMENTS):],
            'molecule': molecule,
            'negatives': self.aux.features(self.negatives(sid)),
        }


def move(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def head_loss(model, output, batch):
    sections = [model.config['bits'], diversity.MACCS_BITS, diversity.PATH_BITS]
    predictions = output['logits'].split(sections, dim=1)
    targets = batch['target'].split(sections, dim=1)
    loss = output['logits'].sum() * 0
    for weight, prediction, target in zip([.60, .20, .20], predictions, targets):
        probability = prediction.float().sigmoid()
        dice = 1 - ((2 * (probability * target).sum(1) + 1)
                    / (probability.sum(1) + target.sum(1) + 1)).mean()
        loss = loss + weight * (nn.functional.binary_cross_entropy_with_logits(prediction, target)
                                + .20 * dice)
    loss = loss + .20 * nn.functional.smooth_l1_loss(output['counts'].float(), batch['counts'])
    positive = model.structure_embedding(batch['molecule'])
    negative = model.structure_embedding(batch['negatives'])
    query = output['embedding']
    hard = torch.cat([(query * positive).sum(1, keepdim=True),
                      torch.einsum('bd,bkd->bk', query, negative)], dim=1) / .10
    loss = loss + .10 * nn.functional.cross_entropy(
        hard.float(), torch.zeros(len(query), dtype=torch.long, device=query.device))
    # Two selected spectra of one structure are positives, not false negatives.
    logits = query @ positive.T / .10
    positives = batch['sid'][:, None] == batch['sid'][None, :]
    contrastive = (torch.logsumexp(logits.float(), dim=1)
                   - torch.logsumexp(logits.float().masked_fill(~positives, -torch.inf), dim=1))
    return loss + .05 * contrastive.mean()


class HeadPredictor:
    def __init__(self, path, config, cache, store, pool, device):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('kind') != 'casmi-dreams-fingerprint-head' or payload.get('version') != VERSION:
            raise ValueError('Not a compatible DreaMS fingerprint-head checkpoint.')
        for name in ['rdkit', 'preprocess', 'fingerprint', 'adducts']:
            if payload[name] != cache.manifest[name]:
                raise ValueError(f'DreaMS head differs from cache in {name}.')
        if payload['dreams_identity'] != store.identity:
            raise ValueError('DreaMS head was trained from another embedding cache.')
        self.device = choose_device(device)
        self.model = DreamsFingerprintHead(payload['model_config']).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval()
        self.training_info = payload['training_info']
        self.aux = diversity.AuxiliaryPool(config, cache, pool)

    @torch.inference_mode()
    def predict(self, vectors, records):
        if not len(vectors):
            return None
        result = self.model(torch.from_numpy(vectors).to(self.device))
        probability = diversity.aggregate_views(
            result['logits'].sigmoid().float().cpu().numpy(), records)
        counts = diversity.aggregate_views(result['counts'].float().cpu().numpy(), records)
        embedding = diversity.aggregate_views(result['embedding'].float().cpu().numpy(), records)
        embedding /= max(np.linalg.norm(embedding), 1e-8)
        return probability, counts, embedding

    @torch.inference_mode()
    def similarities(self, features, embedding):
        values = torch.from_numpy(features).to(self.device)
        return (self.model.structure_embedding(values).float().cpu().numpy() @ embedding)


def feature_scores(predictor, candidate_ids, prediction):
    count = len(candidate_ids)
    raw = np.zeros((count, len(CHANNELS)), dtype=np.float32)
    if prediction is None:
        return raw, raw.copy()
    probability, predicted_counts, embedding = prediction
    boundaries = [0, predictor.aux.bits,
                  predictor.aux.bits + diversity.MACCS_BITS,
                  predictor.aux.bits + diversity.MACCS_BITS + diversity.PATH_BITS]
    for start in range(0, count, 1024):
        ids = candidate_ids[start:start + 1024]
        features = predictor.aux.features(ids)
        for channel in range(3):
            left, right = boundaries[channel:channel + 2]
            raw[start:start + len(ids), channel] = diversity.tanimoto(
                features[:, left:right], probability[left:right])
        raw[start:start + len(ids), 3] = predictor.similarities(features, embedding)
        error = np.abs(features[:, -len(diversity.ELEMENTS):] - predicted_counts)
        weights = np.asarray([.25, .25] + [1.] * (len(diversity.ELEMENTS) - 2), dtype=np.float32)
        raw[start:start + len(ids), 4] = -(error * weights).sum(1)
    normalized = np.stack([diversity.rank_column(raw[:, i]) for i in range(raw.shape[1])], axis=1)
    return raw, normalized


def combine_channels(raw, normalized):
    active = np.ptp(raw, axis=0) > 1e-9
    weights = CHANNEL_WEIGHTS * active
    if weights.sum() <= 0:
        return np.zeros(len(raw), dtype=np.float32)
    return normalized @ (weights / weights.sum())


def validation_rank(model, config, cache, pool, store, split, device, limit):
    aux = diversity.AuxiliaryPool(config, cache, pool)
    selected = [sid for sid in split['selected_sids'] if np.any(store.valid & (store.sids == sid))]
    selected = sorted(selected,
                      key=lambda sid: stable_fraction(cache.structures.iloc[sid]['key'], 307))[:limit]
    scores, covered = [], []
    model.eval()
    with torch.inference_mode():
        for sid in tqdm(selected, desc='DreaMS-head validation', leave=False):
            ordinals = np.flatnonzero(store.valid & (store.sids == sid))
            vectors = np.asarray(store.embeddings[ordinals], dtype=np.float32)
            output = model(torch.from_numpy(vectors).to(device))
            probability = output['logits'].sigmoid().float().mean(0).cpu().numpy()
            counts = output['counts'].float().mean(0).cpu().numpy()
            embedding = output['embedding'].float().mean(0).cpu().numpy()
            embedding /= max(np.linalg.norm(embedding), 1e-8)
            mass = float(pool.masses[sid])
            retrieval = config['retrieval']
            ids = pool.window(mass, retrieval['mass_ppm'], retrieval['mass_da'])
            if not len(ids):
                ids = pool.window(mass, retrieval['fallback_ppm'], retrieval['fallback_da'])
            raw = np.zeros((len(ids), len(CHANNELS)), dtype=np.float32)
            boundaries = [0, aux.bits, aux.bits + diversity.MACCS_BITS,
                          aux.bits + diversity.MACCS_BITS + diversity.PATH_BITS]
            for start in range(0, len(ids), 1024):
                chosen = ids[start:start + 1024]
                features = aux.features(chosen)
                for channel in range(3):
                    left, right = boundaries[channel:channel + 2]
                    raw[start:start + len(chosen), channel] = diversity.tanimoto(
                        features[:, left:right], probability[left:right])
                vectors_s = model.structure_embedding(torch.from_numpy(features).to(device))
                raw[start:start + len(chosen), 3] = vectors_s.float().cpu().numpy() @ embedding
                error = np.abs(features[:, -len(diversity.ELEMENTS):] - counts)
                element_weights = np.asarray([.25, .25] + [1.] * (len(diversity.ELEMENTS) - 2))
                raw[start:start + len(chosen), 4] = -(error * element_weights).sum(1)
            normalized = np.stack([diversity.rank_column(raw[:, i]) for i in range(raw.shape[1])], axis=1)
            ranking = combine_channels(raw, normalized)
            order = np.argsort(-ranking, kind='stable')
            correct = pool.table.iloc[ids[order]]['key'].to_numpy() == cache.structures.iloc[sid]['key']
            positions = np.flatnonzero(correct)
            covered.append(bool(len(positions)))
            scores.append(1. / (positions[0] + 1) if len(positions) and positions[0] < 25 else 0.)
    return {'mrr_at_25': float(np.mean(scores)) if scores else 0.,
            'candidate_coverage': float(np.mean(covered)) if covered else 0.,
            'molecules': len(scores)}


def inference_payload(model, model_config, training_config, info, cache, store, epoch, best, history):
    return {'kind': 'casmi-dreams-fingerprint-head', 'version': VERSION,
            'model': model.state_dict(), 'model_config': model_config,
            'training_config': training_config, 'training_info': info,
            'dreams_identity': store.identity, 'epoch': epoch, 'best_mrr': best, 'history': history,
            **{name: cache.manifest[name] for name in ['rdkit', 'adducts', 'preprocess', 'fingerprint']}}


def train(config, args):
    cache, pool = diversity.open_pool(config)
    aux = diversity.AuxiliaryPool(config, cache, pool)
    store = EmbeddingStore(cache, args.dreams_dir or Path(config['work_dir']) / 'dreams')
    split_path = Path(args.split or Path(config['work_dir']) / 'validation/split.json')
    split = None if args.full else load_split(split_path, cache)
    held = [] if args.full else split['held_sids']
    seed = int(args.seed if args.seed is not None else config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(args.device)
    torch.set_num_threads(min(args.threads, torch.get_num_threads()))
    model_config = {'bits': config['fingerprint']['bits'], 'hidden': args.hidden,
                    'dropout': args.dropout}
    model = DreamsFingerprintHead(model_config).to(device)
    dataset = HeadTrainingData(store, aux, held, args.hard_negatives)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda',
                        worker_init_fn=seed_worker, generator=generator,
                        persistent_workers=args.workers > 0)
    training_sids = np.unique(store.sids[dataset.ordinals])
    frequency = np.zeros(model_config['bits'] + diversity.MACCS_BITS + diversity.PATH_BITS,
                         dtype=np.float64)
    formula_sum = np.zeros(len(diversity.ELEMENTS), dtype=np.float64)
    for start in range(0, len(training_sids), 4096):
        ids = training_sids[start:start + 4096]
        frequency += aux.targets(ids).sum(0)
        formula_sum += (np.log1p(aux.counts[ids]) / 4.).sum(0)
    frequency = np.clip(frequency / len(training_sids), 1e-4, 1 - 1e-4)
    with torch.no_grad():
        model.fingerprint.bias.copy_(torch.from_numpy(np.log(frequency / (1 - frequency))).float().to(device))
        mean = np.maximum(formula_sum / len(training_sids), 1e-3)
        model.formula.bias.copy_(torch.from_numpy(np.log(np.expm1(mean))).float().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    warmup = min(500, max(1, total_steps // 10))

    def rate(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1., (step - warmup) / max(total_steps - warmup, 1))
        return .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    dest = Path(args.output_dir or Path(config['work_dir']) /
                ('models/dreams_head_full' if args.full else 'models/dreams_head_cv'))
    dest.mkdir(parents=True, exist_ok=True)
    info = {'full': args.full, 'held_keys': [] if args.full else split['held_keys'],
            'split_sha256': None if args.full else file_digest(split_path), 'seed': seed,
            'training_structures': len(training_sids), 'training_embeddings': len(dataset),
            'partial_cache': cache.manifest['partial_cache'], 'encoder_frozen': True}
    training_config = {name: getattr(args, name) for name in [
        'epochs', 'batch_size', 'hidden', 'dropout', 'hard_negatives',
        'learning_rate', 'weight_decay', 'validation_molecules']}
    epoch_start, best, history = 0, -1., []
    if args.resume:
        previous = torch.load(args.resume, map_location='cpu', weights_only=True)
        for name, value in [('model_config', model_config), ('training_config', training_config),
                            ('training_info', info), ('dreams_identity', store.identity)]:
            if previous[name] != value:
                raise ValueError(f'Resume differs in {name}; retain the original arguments.')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        scaler.load_state_dict(previous['scaler'])
        epoch_start, best, history = previous['epoch'], previous['best_mrr'], previous['history']
        torch.set_rng_state(previous['torch_rng'])
        generator.set_state(previous['loader_rng'])
        state = previous['numpy_rng']
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32),
                             state[2], state[3], state[4]))
        random.setstate(previous['python_rng'])
        if previous['cuda_rng'] is not None and amp:
            torch.cuda.set_rng_state_all(previous['cuda_rng'])
    elif (dest / 'last.pt').exists():
        raise FileExistsError('Choose a new output directory or use --resume.')
    print('Explicit DreaMS-head training:', {'device': str(device),
          'parameters': sum(p.numel() for p in model.parameters()), **training_config})
    for epoch in range(epoch_start, args.epochs):
        started, total, seen = time.monotonic(), 0., 0
        if amp:
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        for batch in tqdm(loader, desc=f'DreaMS head epoch {epoch + 1}/{args.epochs}'):
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = head_loss(model, model(batch['embedding']), batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach()) * len(batch['sid'])
            seen += len(batch['sid'])
        validation = None if args.full else validation_rank(
            model, config, cache, pool, store, split, device, args.validation_molecules)
        improved = validation is not None and validation['mrr_at_25'] > best
        if improved:
            best = validation['mrr_at_25']
        record = {'epoch': epoch + 1, 'loss': total / max(seen, 1), 'validation': validation,
                  'seconds': time.monotonic() - started,
                  'cuda_peak_allocated_gb': torch.cuda.max_memory_allocated(device) / 2**30 if amp else 0.}
        history.append(record)
        inference = inference_payload(model, model_config, training_config, info, cache,
                                      store, epoch + 1, best, history)
        state = np.random.get_state()
        resume = {**inference, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                  'scaler': scaler.state_dict(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state_all() if amp else None,
                  'loader_rng': generator.get_state(), 'python_rng': random.getstate(),
                  'numpy_rng': [state[0], state[1].tolist(), int(state[2]),
                                int(state[3]), float(state[4])]}
        torch.save(resume, dest / 'last.pt.tmp')
        (dest / 'last.pt.tmp').replace(dest / 'last.pt')
        if improved:
            torch.save(inference, dest / 'best.pt')
        if args.full:
            torch.save(inference, dest / 'final.pt')
        write_json(dest / 'history.json', history)
        print(json.dumps(record, ensure_ascii=False))
    print('Training output:', dest)


class DreamsHeadRanker(dreams.DreamsRanker):
    head_path = None
    head_weight = .16

    def __init__(self, config, cache, pool, *args, **kwargs):
        super().__init__(config, cache, pool, *args, **kwargs)
        path = Path(self.dreams_path or Path(config['work_dir']) / 'dreams')
        store = EmbeddingStore(cache, path)
        checkpoint = self.head_path or Path(config['work_dir']) / 'models/dreams_head.pt'
        self.head = HeadPredictor(checkpoint, config, cache, store, pool, str(self.encoder.device))
        self._head_query = None

    def rank(self, frame):
        self._head_query = None
        try:
            return super().rank(frame)
        finally:
            self._head_query = None

    def query_embeddings(self):
        if self._head_query is None:
            self._head_query = super().query_embeddings()
        return self._head_query

    def rank_queries(self, queries):
        ranked, diagnostic = super().rank_queries(queries)
        vectors, records = self.query_embeddings()
        prediction = self.head.predict(vectors, records)
        ids = ranked['candidate_id'].to_numpy(np.int64)
        raw, normalized = feature_scores(self.head, ids, prediction)
        feature = combine_channels(raw, normalized)
        for i, name in enumerate(CHANNELS):
            ranked['dreams_head_' + name] = normalized[:, i]
            ranked['dreams_head_raw_' + name] = raw[:, i]
        ranked['dreams_head'] = feature
        gate = 1. - .75 * float(diagnostic['library_gate'])
        ranked['score'] += gate * self.head_weight * feature
        ranked = ranked.sort_values(['score', 'error_ppm', 'key'],
                                    ascending=[False, True, True], kind='stable')
        diagnostic.update(dreams_head_used=prediction is not None,
                          best_dreams_head=float(feature.max(initial=0)))
        return ranked, diagnostic


def parser():
    p = argparse.ArgumentParser(description='Supervised fingerprint head on frozen DreaMS embeddings')
    p.add_argument('--config', default=str(ROOT / 'configs/local.yaml'))
    p.add_argument('--data-dir')
    p.add_argument('--work-dir')
    sub = p.add_subparsers(dest='command', required=True)
    item = sub.add_parser('train')
    item.add_argument('--split')
    item.add_argument('--full', action='store_true')
    item.add_argument('--dreams-dir')
    item.add_argument('--device', default='cuda')
    item.add_argument('--output-dir')
    item.add_argument('--epochs', type=int, default=20)
    item.add_argument('--batch-size', type=int, default=512)
    item.add_argument('--hidden', type=int, default=768)
    item.add_argument('--dropout', type=float, default=.15)
    item.add_argument('--hard-negatives', type=int, default=8)
    item.add_argument('--learning-rate', type=float, default=3e-4)
    item.add_argument('--weight-decay', type=float, default=1e-4)
    item.add_argument('--validation-molecules', type=int, default=400)
    item.add_argument('--workers', type=int, default=2)
    item.add_argument('--threads', type=int, default=4)
    item.add_argument('--seed', type=int)
    item.add_argument('--resume')
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
        item.add_argument('--head-checkpoint')
        item.add_argument('--head-weight', type=float, default=.16)
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
    if args.command == 'train':
        if (args.batch_size < 1 or args.epochs < 1 or args.hidden < 1 or args.hard_negatives < 1
                or args.workers < 0 or args.threads < 1 or not 0 <= args.dropout < 1
                or args.learning_rate <= 0 or args.weight_decay < 0):
            raise ValueError('Invalid DreaMS-head training parameter.')
        return train(config, args)
    if (min(args.analog_weight, args.direct_weight, args.propagation_weight, args.head_weight) < 0
            or not 0 <= args.min_similarity < 1 or args.neighbors < 1
            or args.batch_size < 1 or args.scan_batch < 1 or args.threads < 1):
        raise ValueError('Invalid DreaMS-head inference parameter.')
    torch.set_num_threads(args.threads)
    checkpoint = Path(args.head_checkpoint or Path(config['work_dir']) / 'models/dreams_head.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    info = payload.get('training_info', {})
    if args.command == 'predict' and not info.get('full'):
        raise ValueError('Submission requires a full DreaMS-head checkpoint.')
    if args.command == 'validate':
        cache, _ = diversity.open_pool(config)
        split_path = Path(args.split or Path(config['work_dir']) / 'validation/split.json')
        split = load_split(split_path, cache)
        if info.get('full') or not set(split['held_keys']).issubset(set(info.get('held_keys', []))):
            raise ValueError('DreaMS head has seen held identities; use its matching CV checkpoint.')
    bound = type('BoundDreamsHeadRanker', (DreamsHeadRanker,), {
        'index_dir': args.index_dir, 'weight': args.analog_weight,
        'dreams_path': args.dreams_dir, 'direct_weight': args.direct_weight,
        'propagation_weight': args.propagation_weight, 'min_similarity': args.min_similarity,
        'neighbor_limit': args.neighbors, 'encoder_batch': args.batch_size,
        'scan_batch': args.scan_batch, 'head_path': checkpoint, 'head_weight': args.head_weight})
    with patch.object(diversity, 'DiverseRanker', bound):
        result = diversity.run_predict(config, args, args.command == 'validate')
    if args.output:
        path = Path(args.output).with_suffix('.diagnostics.json')
        metadata = read_json(path)
        metadata.update(engine='dreams-fingerprint-head', dreams_head={
            'version': VERSION, 'checkpoint': str(checkpoint), 'weight': args.head_weight,
            'channels': list(CHANNELS), 'channel_weights': CHANNEL_WEIGHTS.tolist(),
            'encoder_frozen': True})
        write_json(path, metadata)
    return result


if __name__ == '__main__':
    main()
