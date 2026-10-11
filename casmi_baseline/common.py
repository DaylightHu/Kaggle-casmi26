import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path=None, data_dir=None, work_dir=None):
    path = Path(path or ROOT / "configs/local.yaml").resolve()
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data_dir"] = str(Path(data_dir or config["data_dir"]).expanduser().resolve())
    # Relative work directories are relative to the repository, not the caller's cwd.
    task_work = Path(work_dir or config["work_dir"]).expanduser()
    config["work_dir"] = str((ROOT / task_work).resolve())
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_fraction(key, seed):
    raw = hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8]
    return int.from_bytes(raw, "little") / 2**64


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
