"""Canonical digests, file hashes, atomic JSON, JSONL reading and directory locks."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path


def digest(value):
    """SHA256 of the canonical (sorted, compact, UTF-8) JSON encoding of value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    """Every row of a complete JSONL file (gzip if the name ends in .gz)."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_partial(path):
    """Rows of a JSONL file that workers append to: a missing file is empty and undecodable lines
    (a worker killed mid-write leaves one partial last line) are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


@contextmanager
def locked(directory, name="lock"):
    """Exclusive flock on directory/name for the duration of the block."""
    with (Path(directory) / name).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
