"""Offline Kaggle inference. Never invokes prepare or train-fingerprint."""
import argparse
import os
import sys
from pathlib import Path


def locate_competition(explicit=None):
    if explicit:
        roots = [Path(explicit)]
    else:
        roots = sorted({p.parent for p in Path("/kaggle/input").rglob("test.parquet")
                        if (p.parent / "sample_submission.csv").exists() and (p.parent / "train.parquet").exists()})
    if len(roots) != 1:
        raise RuntimeError(f"Found {len(roots)} competition directories; pass --data-dir explicitly")
    return roots[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True, help="Root of the exported inference asset dataset")
    parser.add_argument("--data-dir")
    parser.add_argument("--output", default="/kaggle/working/submission.csv")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    assets = Path(args.assets).resolve()
    # Compiled numba caches must be written to working, never Kaggle's read-only input mount.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/kaggle/working/numba-cache")
    sys.path.insert(0, str(assets))
    from casmi_baseline.common import load_config
    from casmi_baseline.retrieval import predict
    data = locate_competition(args.data_dir)
    config = load_config(assets / "config.yaml", data_dir=data, work_dir=assets)
    checkpoint = assets / "models/fingerprint.pt"
    predict(config, test_path=data / "test.parquet", sample_path=data / "sample_submission.csv",
            output=args.output, checkpoint=checkpoint if checkpoint.exists() else None, device=args.device)


if __name__ == "__main__":
    main()
