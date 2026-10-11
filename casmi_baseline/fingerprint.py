"""Optional supervised encoder. Only train_fingerprint() performs optimizer steps."""
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .cache import ReferenceCache
from .chemistry import check_rdkit
from .common import file_digest, write_json
from .validation import load_split


def choose_device(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check the installed torch wheel and nvidia-smi.")
    return torch.device(device)


def tokenize(query, model_config):
    width, maximum = model_config["bin_width"], model_config["max_mz"]
    bins = int(math.ceil(maximum / width)) + 1
    mz = np.asarray(query["mz"], dtype=np.float32)
    weight = np.asarray(query["weights"], dtype=np.float32)
    losses = query["precursor"] - mz
    ids, values = [], []
    for channel, masses in enumerate((mz, losses)):
        good = np.isfinite(masses) & (masses >= 0) & (masses <= maximum) & (weight > 0)
        token = np.clip(np.rint(masses[good]/width).astype(np.int64), 0, bins-1) + 1 + channel*bins
        amplitude = weight[good]
        if len(token):
            unique, inverse = np.unique(token, return_inverse=True)
            merged = np.bincount(inverse, weights=amplitude).astype(np.float32)
            merged /= max(merged.sum(), 1e-8)
            ids.append(unique)
            values.append(merged)
    if ids:
        ids, values = np.concatenate(ids), np.concatenate(values)
    else:
        ids, values = np.array([0], dtype=np.int64), np.array([0], dtype=np.float32)
    ce = query["ce"]
    metadata = np.array([query["precursor"]/maximum, query["mass"]/maximum,
                         np.clip(ce/100, 0, 5) if np.isfinite(ce) else 0,
                         float(not np.isfinite(ce)), float(query["mode"] == 1),
                         float(query["mode"] == -1), float(query["timstof"]),
                         min(len(mz)/128, 2)], dtype=np.float32)
    return ids, values, int(query["adduct"]), metadata


def collate(items):
    lengths = [len(item[0]) for item in items]
    batch = {"tokens": torch.from_numpy(np.concatenate([v[0] for v in items])),
             "weights": torch.from_numpy(np.concatenate([v[1] for v in items])),
             "offsets": torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.long),
             "adduct": torch.tensor([v[2] for v in items], dtype=torch.long),
             "metadata": torch.from_numpy(np.stack([v[3] for v in items]))}
    if len(items[0]) > 4:
        batch["target"] = torch.from_numpy(np.stack([v[4] for v in items]))
    return batch


class SpectrumFingerprintNet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        bins = int(math.ceil(model_config["max_mz"]/model_config["bin_width"])) + 1
        dimension = model_config["embedding_dim"]
        hidden = model_config["hidden_dim"]
        self.peaks = nn.EmbeddingBag(2*bins+1, dimension, mode="sum", include_last_offset=True, padding_idx=0)
        self.adduct = nn.Embedding(model_config["num_adducts"], 24)
        self.head = nn.Sequential(nn.Linear(dimension+24+8, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                  nn.Dropout(model_config["dropout"]), nn.Linear(hidden, hidden), nn.GELU(),
                                  nn.Dropout(model_config["dropout"]), nn.Linear(hidden, model_config["bits"]))

    def forward(self, batch):
        spectral = self.peaks(batch["tokens"], batch["offsets"], per_sample_weights=batch["weights"])
        return self.head(torch.cat([spectral, self.adduct(batch["adduct"]), batch["metadata"]], dim=1))


def query_from_cache(cache, row):
    meta = cache.meta[row]
    n = int(meta["n"])
    from .chemistry import neutral_mass
    try:
        mass = neutral_mass(meta["precursor"], cache.manifest["adducts"][int(meta["adduct"])] )
    except (ValueError, KeyError):
        mass = float(meta["precursor"])
    return {"mz": np.asarray(cache.mzs[row, :n]), "weights": np.asarray(cache.weights[row, :n], dtype=np.float32),
            "precursor": float(meta["precursor"]), "mass": mass,
            "adduct": int(meta["adduct"]), "mode": int(meta["mode"]), "ce": float(meta["ce"]),
            "timstof": bool(meta["timstof"]), "quality": 1.0}


class TrainingSpectra(Dataset):
    def __init__(self, cache, sids, model_config, config):
        self.cache, self.sids, self.model_config, self.config = cache, np.asarray(sids), model_config, config
        valid = np.isin(cache.meta["sid"], self.sids) & (cache.meta["n"] >= 5)
        error = cache.meta["error_ppm"]
        valid &= ~np.isfinite(error) | (np.abs(error) <= cache.manifest["preprocess"]["max_precursor_error_ppm"])
        rows = np.flatnonzero(valid)
        self.rows = rows[np.argsort(cache.meta["sid"][rows], kind="stable")]
        counts = np.bincount(cache.meta["sid"][self.rows], minlength=len(cache.structures))
        self.offsets = np.r_[0, np.cumsum(counts)]
        self.sids = self.sids[counts[self.sids] > 0]
        if not len(self.sids):
            raise ValueError("No training structures with at least five usable peaks")

    def __len__(self):
        return len(self.sids) * self.config["views_per_molecule"]

    def __getitem__(self, index):
        sid = int(self.sids[index % len(self.sids)])
        choices = self.rows[self.offsets[sid]:self.offsets[sid+1]]
        row = int(choices[np.random.randint(len(choices))])
        query = query_from_cache(self.cache, row)
        if self.config["peak_dropout"]:
            keep = np.random.random(len(query["mz"])) >= self.config["peak_dropout"]
            if not keep.any():
                keep[np.argmax(query["weights"])] = True
            query["mz"], query["weights"] = query["mz"][keep], query["weights"][keep]
        target = np.unpackbits(self.cache.fps[sid], bitorder="little").astype(np.float32)
        return (*tokenize(query, self.model_config), target)


def seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def transfer(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def load_checkpoint(path):
    # Checkpoints contain tensors and primitive metadata, so arbitrary Python objects are unnecessary.
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported fingerprint checkpoint")
    return payload


class FingerprintPredictor:
    def __init__(self, path, cache, device="auto"):
        self.device = choose_device(device)
        payload = load_checkpoint(path)
        if payload["adducts"] != cache.manifest["adducts"] or payload["fingerprint"] != cache.manifest["fingerprint"]:
            raise ValueError("Checkpoint and cache use different fingerprints or adduct mappings")
        if payload["preprocess"] != cache.manifest["preprocess"] or payload["rdkit"] != cache.manifest["rdkit"]:
            raise ValueError("Checkpoint chemistry/preprocessing differs from cache")
        self.model_config = payload["model_config"]
        self.training_info = payload["training_info"]
        self.model = SpectrumFingerprintNet(self.model_config).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, queries):
        batch = transfer(collate([tokenize(q, self.model_config) for q in queries]), self.device)
        probabilities = self.model(batch).sigmoid().float().cpu().numpy()
        groups = {}
        for i, q in enumerate(queries):
            groups.setdefault((q["adduct"], q["mode"]), []).append(i)
        averaged = []
        for rows in groups.values():
            averaged.append(np.average(probabilities[rows], axis=0, weights=[queries[i]["quality"] for i in rows]))
        return np.mean(averaged, axis=0)


def validation_bce(model, cache, rows, model_config, device, batch_size):
    if not rows:
        return None
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start:start+batch_size]
            items = []
            for row in batch_rows:
                q = query_from_cache(cache, row)
                # Match inference: neutral mass comes from measured precursor and adduct, not the answer label.
                from .chemistry import neutral_mass
                try:
                    q["mass"] = neutral_mass(q["precursor"], cache.manifest["adducts"][q["adduct"]])
                except (ValueError, KeyError):
                    q["mass"] = q["precursor"]
                target = np.unpackbits(cache.fps[int(cache.meta["sid"][row])], bitorder="little").astype(np.float32)
                items.append((*tokenize(q, model_config), target))
            batch = transfer(collate(items), device)
            loss = nn.functional.binary_cross_entropy_with_logits(model(batch), batch["target"], reduction="sum")
            total += float(loss.item())
            count += batch["target"].numel()
    return total / count


def train_fingerprint(config, split_path=None, full=False, device="auto", batch_size=None,
                      epochs=None, steps_per_epoch=None, output_dir=None, resume=None, allow_mismatch=False):
    # This function is intentionally reachable only via an explicit train-fingerprint command.
    version = check_rdkit(allow_mismatch)
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    cache.require_config(config)
    if version != cache.manifest["rdkit"]:
        raise ValueError("RDKit differs from cache")
    cfg = config["training"].copy()
    for name, value in [("batch_size", batch_size), ("epochs", epochs), ("steps_per_epoch", steps_per_epoch)]:
        if value is not None:
            cfg[name] = value
    if min(cfg["batch_size"], cfg["epochs"], cfg["steps_per_epoch"], cfg["views_per_molecule"]) < 1:
        raise ValueError("Training sizes must be positive")
    if not full and split_path is None:
        split_path = Path(config["work_dir"]) / "validation/split.json"
    split = None if full else load_split(split_path, cache)
    withheld = [] if full else split["held_sids"]
    sids = np.setdiff1d(np.arange(len(cache.structures)), withheld)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(device)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    model_config = {name: cfg[name] for name in ("embedding_dim", "hidden_dim", "bin_width", "max_mz", "dropout")}
    model_config.update(bits=config["fingerprint"]["bits"], num_adducts=len(cache.manifest["adducts"]))
    dataset = TrainingSpectra(cache, sids, model_config, cfg)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"],
                        collate_fn=collate, pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    steps = min(cfg["steps_per_epoch"], len(loader))
    model = SpectrumFingerprintNet(model_config).to(device)
    # Count only training identities; validation fingerprint frequencies are never used to initialise the model.
    frequencies = np.zeros(config["fingerprint"]["bits"], dtype=np.float64)
    for start in range(0, len(dataset.sids), 4096):
        frequencies += np.unpackbits(cache.fps[dataset.sids[start:start+4096]], axis=1, bitorder="little").sum(axis=0)
    frequencies = np.clip(frequencies/len(dataset.sids), 1e-4, 1-1e-4)
    with torch.no_grad():
        model.head[-1].bias.copy_(torch.from_numpy(np.log(frequencies/(1-frequencies))).float().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps*cfg["epochs"])
    amp = bool(cfg["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = Path(output_dir or Path(config["work_dir"]) / ("models/full" if full else "models/cv"))
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "last.pt").exists():
        raise FileExistsError("Existing checkpoint; use another --output-dir or explicitly --resume")
    split_hash = None if full else file_digest(split_path)
    training_info = {"full": bool(full), "split_sha256": split_hash, "held_keys": [] if full else split["held_keys"],
                     "training_structures": int(len(dataset.sids)), "seed": seed,
                     "partial_cache": cache.manifest["partial_cache"]}
    start_epoch, best, history = 0, float("inf"), []
    if resume:
        previous = load_checkpoint(resume)
        for key, expected in [("model_config", model_config), ("training_info", training_info), ("training_config", cfg)]:
            if previous[key] != expected:
                raise ValueError(f"Resume checkpoint differs in {key}; use the original training configuration")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        scaler.load_state_dict(previous["scaler"])
        start_epoch = previous["epoch"]
        best = previous["best_validation_bce"] if previous["best_validation_bce"] is not None else float("inf")
        history = previous["history"]
        torch.set_rng_state(previous["torch_rng"])
        if previous.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        loader.generator.set_state(previous["loader_rng"])
        rng = previous["numpy_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), rng[2], rng[3], rng[4]))
        random.setstate(previous["python_rng"])
    val_rows = [] if full else split["query_rows"][:2000]
    print(f"Explicit training: device={device}, AMP={amp}, batch={cfg['batch_size']}, structures={len(dataset.sids)}, steps/epoch={steps}")
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        total, seen = 0.0, 0
        for step, batch in enumerate(tqdm(loader, total=steps, desc=f"epoch {epoch+1}/{cfg['epochs']}")):
            if step >= steps:
                break
            batch = transfer(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(batch)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])
                p = logits.float().sigmoid()
                target = batch["target"]
                intersection = (p*target).sum(dim=1)
                dice_loss = 1 - ((2*intersection+1)/(p.sum(dim=1)+target.sum(dim=1)+1)).mean()
                loss = bce + 0.20*dice_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += float(loss.detach()) * len(target)
            seen += len(target)
        val = validation_bce(model, cache, val_rows, model_config, device, cfg["batch_size"])
        record = {"epoch": epoch+1, "train_loss": total/max(seen, 1), "validation_bce": val,
                  "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0.0}
        history.append(record)
        improved = val is not None and val < best
        if improved:
            best = val
        nrng = np.random.get_state()
        payload = {"format_version": 1, "model": model.state_dict(), "model_config": model_config,
                   "training_info": training_info, "training_config": cfg, "adducts": cache.manifest["adducts"],
                   "fingerprint": cache.manifest["fingerprint"], "preprocess": cache.manifest["preprocess"],
                   "rdkit": version, "epoch": epoch+1, "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
                   "best_validation_bce": None if not np.isfinite(best) else best,
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                   "loader_rng": loader.generator.get_state(), "python_rng": random.getstate(),
                   "numpy_rng": [nrng[0], nrng[1].tolist(), int(nrng[2]), int(nrng[3]), float(nrng[4])]}
        torch.save(payload, output / "last.pt.tmp")
        (output / "last.pt.tmp").replace(output / "last.pt")
        if improved:
            torch.save(payload, output / "best.pt")
        if full:
            torch.save(payload, output / "final.pt")
        write_json(output / "history.json", history)
        print(record)
    print("Training output:", output)
