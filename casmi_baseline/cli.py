import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from .common import ROOT, load_config, read_json, write_json


def parser():
    p = argparse.ArgumentParser(description="CASMI 2026 baseline: explicit commands, no automatic training")
    p.add_argument("--config", default=str(ROOT / "configs/local.yaml"))
    p.add_argument("--data-dir", help="Directory containing train.parquet/test.parquet/sample_submission.csv")
    p.add_argument("--work-dir", help="Disk cache/model directory")
    p.add_argument("--allow-rdkit-mismatch", action="store_true", help="Development only; scores are not metric-equivalent")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check the environment; never trains")
    sub.add_parser("inspect", help="Inspect mounted competition file schemas")
    prep = sub.add_parser("prepare", help="Build a streaming reference cache; not model training")
    prep.add_argument("--train-path")
    prep.add_argument("--max-rows", type=int, help="Development only; resulting cache cannot be submitted")
    inspect = sub.add_parser("inspect-candidates")
    inspect.add_argument("--path", required=True)
    extra = sub.add_parser("import-candidates")
    extra.add_argument("--path", required=True)
    extra.add_argument("--smiles-column", default="smiles")
    extra.add_argument("--name", default="external")
    split = sub.add_parser("make-split")
    split.add_argument("--kind", choices=["identity", "np-panel"], default="identity")
    split.add_argument("--fraction", type=float, default=0.1)
    split.add_argument("--molecules", type=int, default=400)
    split.add_argument("--spectra-per-molecule", type=int, default=8)
    split.add_argument("--train-path")
    for name in ("predict", "validate"):
        item = sub.add_parser(name)
        item.add_argument("--checkpoint", help="Optional fingerprint checkpoint; absent means no neural model")
        item.add_argument("--device", default="auto")
        item.add_argument("--output", default="submission.csv" if name == "predict" else None)
        if name == "predict":
            item.add_argument("--test-path")
            item.add_argument("--sample-path")
        else:
            item.add_argument("--split")
            item.add_argument("--scenario", choices=["no-reference", "library"], default="no-reference")
    train = sub.add_parser("train-fingerprint", help="ONLY this command fits model weights")
    train.add_argument("--split")
    train.add_argument("--full", action="store_true", help="Fit all identities; not eligible for validation scoring")
    train.add_argument("--device", default="auto")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--steps-per-epoch", type=int)
    train.add_argument("--output-dir")
    train.add_argument("--resume")
    check = sub.add_parser("check-submission")
    check.add_argument("--path", required=True)
    check.add_argument("--test-path")
    export = sub.add_parser("export-assets", help="Copy code/cache/optional weights for offline Kaggle inference")
    export.add_argument("--dest", required=True)
    export.add_argument("--checkpoint")
    export.add_argument("--wheelhouse")
    return p


def doctor(config, allow_mismatch):
    print("Python:", sys.version.split()[0], "Data:", config["data_dir"], "Work:", config["work_dir"])
    failed = []
    for name in ("numpy", "pandas", "pyarrow", "rdkit", "numba", "yaml", "tqdm"):
        try:
            module = importlib.import_module(name)
            print(name, getattr(module, "__version__", "installed"))
        except ImportError as e:
            print(name, "missing:", e)
            if name != "numba":
                failed.append(name)
    if failed:
        raise RuntimeError("Missing baseline dependencies: " + ", ".join(failed))
    from .chemistry import check_rdkit
    check_rdkit(allow_mismatch)
    try:
        import torch
        print("torch", torch.__version__, "CUDA:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0), "supported CUDA architectures:", torch.cuda.get_arch_list())
    except ImportError:
        print("torch absent: CPU retrieval is available; install torch for optional fingerprint training")


def inspect(config):
    import pyarrow.parquet as pq
    for filename in ("train.parquet", "test.parquet"):
        path = Path(config["data_dir"]) / filename
        pf = pq.ParquetFile(path)
        print(filename, "rows=", pf.metadata.num_rows, "size_GiB=", round(path.stat().st_size/2**30, 3))
        print(pf.schema_arrow)


def export_assets(config, dest, checkpoint=None, wheelhouse=None):
    from .cache import ReferenceCache
    from .chemistry import METRIC_RDKIT
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    cache.require_config(config)
    if cache.manifest["rdkit"] != METRIC_RDKIT:
        raise ValueError("Official offline assets require an RDKit 2026.03.3 cache")
    if cache.manifest["partial_cache"]:
        raise ValueError("Cannot export a partial development cache")
    dest = Path(dest).resolve()
    source_work = Path(config["work_dir"]).resolve()
    if dest == source_work or source_work in dest.parents:
        raise ValueError("Export directory must be outside work_dir")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Export directory must be empty")
    if checkpoint:
        from .fingerprint import load_checkpoint
        payload = load_checkpoint(checkpoint)
        if payload["training_info"]["partial_cache"]:
            raise ValueError("Cannot export a checkpoint trained with a partial cache")
        if any(payload[key] != cache.manifest[key] for key in ("adducts", "preprocess", "fingerprint", "rdkit")):
            raise ValueError("Checkpoint does not match the cache")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "casmi_baseline", dest / "casmi_baseline", ignore=shutil.ignore_patterns("__pycache__", "*.nbc", "*.nbi"))
    shutil.copytree(source_work / "cache", dest / "cache")
    if (source_work / "candidates/manifest.json").exists():
        shutil.copytree(source_work / "candidates", dest / "candidates")
    for filename in ("requirements.txt", "README.md", "SOURCES.md"):
        shutil.copy2(ROOT / filename, dest / filename)
    shutil.copy2(ROOT / "kaggle/infer.py", dest / "infer.py")
    exported_config = config.copy()
    exported_config["work_dir"] = "."
    import yaml
    (dest / "config.yaml").write_text(yaml.safe_dump(exported_config, allow_unicode=True), encoding="utf-8")
    if checkpoint:
        (dest / "models").mkdir()
        shutil.copy2(checkpoint, dest / "models/fingerprint.pt")
    if wheelhouse:
        shutil.copytree(wheelhouse, dest / "wheels")
    write_json(dest / "assets.json", {"format_version": 1, "model": bool(checkpoint),
               "rdkit": cache.manifest["rdkit"], "cache_rows": cache.manifest["rows"],
               "cache_structures": cache.manifest["structures"]})
    print("Offline assets ready:", dest)


def validate(config, args):
    import pandas as pd
    from .cache import ReferenceCache
    from .metric import evaluate_submission
    from .retrieval import predict
    from .validation import load_split
    cache = ReferenceCache(Path(config["work_dir"]) / "cache")
    split_path = args.split or Path(config["work_dir"]) / "validation/split.json"
    split = load_split(split_path, cache)
    if args.scenario == "library" and split["kind"] != "np-panel":
        raise ValueError("library validation requires --kind np-panel; identity splits use no-reference")
    if args.checkpoint:
        from .fingerprint import load_checkpoint
        info = load_checkpoint(args.checkpoint)["training_info"]
        if info["full"] or not set(split["held_keys"]).issubset(set(info["held_keys"])):
            raise ValueError("Checkpoint has seen validation identities. Use a CV model trained with this split.")
    blocked_sids = split["held_sids"] if args.scenario == "no-reference" else []
    dest = Path(args.output or Path(config["work_dir"]) / f"validation/{args.scenario}_submission.csv")
    # These files are resolved relative to split.json so validation folders remain relocatable.
    validation_dir = Path(split_path).resolve().parent
    truth = pd.read_csv(validation_dir / "truth.csv", dtype=str)
    sub, diagnostics = predict(config, test_path=validation_dir / "queries.parquet", output=dest,
                   checkpoint=args.checkpoint, device=args.device, allow_mismatch=args.allow_rdkit_mismatch,
                   blocked_sids=blocked_sids, blocked_rows=split["blocked_rows"], require_full_cache=False)
    report = evaluate_submission(sub, truth, args.allow_rdkit_mismatch)
    truth_keys = {"val_"+cache.structures.iloc[sid]["key"]: cache.structures.iloc[sid]["key"] for sid in split["selected_sids"]}
    covered = [truth_keys[q["molecule_id"]] in q["candidate_keys"] for q in diagnostics["queries"]]
    report.update(candidate_coverage=sum(covered)/len(covered), scenario=args.scenario,
                  split_kind=split["kind"], checkpoint=bool(args.checkpoint), rdkit=diagnostics["rdkit"])
    write_json(dest.with_suffix(".metrics.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    config = load_config(args.config, args.data_dir, args.work_dir)
    if not 1 <= config["retrieval"]["top_k"] <= 25:
        raise ValueError("top_k must be between 1 and 25")
    if args.command == "doctor":
        return doctor(config, args.allow_rdkit_mismatch)
    if args.command == "inspect":
        return inspect(config)
    if args.command == "prepare":
        from .cache import prepare_cache
        return prepare_cache(config, args.train_path, args.allow_rdkit_mismatch, args.max_rows)
    if args.command in ("inspect-candidates", "import-candidates"):
        from .candidates import inspect_candidates, import_candidates
        if args.command == "inspect-candidates":
            return inspect_candidates(args.path)
        return import_candidates(config, args.path, args.smiles_column, args.name, args.allow_rdkit_mismatch)
    if args.command == "make-split":
        from .validation import make_split
        return make_split(config, args.kind, args.fraction, args.molecules, args.spectra_per_molecule, args.train_path)
    if args.command == "train-fingerprint":
        from .fingerprint import train_fingerprint
        return train_fingerprint(config, args.split, args.full, args.device, args.batch_size, args.epochs,
                                 args.steps_per_epoch, args.output_dir, args.resume, args.allow_rdkit_mismatch)
    if args.command == "predict":
        from .retrieval import predict
        sample = args.sample_path
        if sample is None and args.test_path is None:
            sample = Path(config["data_dir"]) / "sample_submission.csv"
        return predict(config, args.test_path, sample, args.output, args.checkpoint, args.device, args.allow_rdkit_mismatch)
    if args.command == "validate":
        return validate(config, args)
    if args.command == "check-submission":
        import pandas as pd
        from .metric import check_submission
        ids = pd.read_parquet(args.test_path or Path(config["data_dir"]) / "test.parquet", columns=["molecule_id"])["molecule_id"].astype(str).unique()
        frame = check_submission(args.path, ids, args.allow_rdkit_mismatch)
        print(f"Submission valid: {len(frame)} molecules")
        return
    if args.command == "export-assets":
        return export_assets(config, args.dest, args.checkpoint, args.wheelhouse)


if __name__ == "__main__":
    main()
