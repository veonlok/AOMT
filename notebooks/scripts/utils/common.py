#!/usr/bin/env python3
"""Shared helpers, path config, and the split-file loader used by the
dataset-audit script (``audit_dataset.py``).

This module is intentionally dependency-light (standard library only) so it can
sit at the bottom of the import graph without creating cycles."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # scripts/utils/common.py -> repo root
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"


def find_repo_root(start=None) -> Path:
    """Walk up from ``start`` (default: cwd) to the repo root — the first ancestor
    holding both ``data/`` and ``scripts/``. For notebooks whose working directory
    isn't the repo root; scripts import ``ROOT`` (above) directly instead."""
    p = Path(start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "data").is_dir() and (cand / "scripts").is_dir():
            return cand
    raise RuntimeError("Could not locate repo root (needs data/ and scripts/).")

SPLIT_FILES = [
    ("train", DATA_DIR / "train.jsonl"),
    ("validation", DATA_DIR / "validation.jsonl"),
    ("test", DATA_DIR / "test.jsonl"),
]


# ---------------------------------------------------------------- filesystem io
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path):
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------- collections
def group_by(items, key_fn):
    out = defaultdict(list)
    for item in items:
        out[key_fn(item)].append(item)
    return out

def counter_object(items):
    counts = Counter(items)
    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], str(pair[0])))
    return {k: v for k, v in ordered}

def render_markdown_table(rows, headers):
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = [
        "| " + " | ".join(str(row.get(header, "")) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


# ----------------------------------------------------------------- data loading
def load_split_rows():
    """Read the local ScienceWorld snapshot split files, tagging each row with
    its source split/file. Returns ``(rows, local_summary)`` where
    ``local_summary`` carries row counts, env counts, and per-file hashes."""
    rows = []
    local_summary = {
        "total_rows": 0,
        "split_counts": {},
        "env_counts": {},
        "file_hashes": {},
    }

    for split_name, file_path in SPLIT_FILES:
        split_rows = read_jsonl(file_path)
        for row in split_rows:
            row["_sourceSplit"] = split_name
            row["_sourceFile"] = str(file_path.relative_to(ROOT)).replace("\\", "/")
        rows.extend(split_rows)
        local_summary["split_counts"][split_name] = len(split_rows)
        local_summary["file_hashes"][split_name] = sha256_file(file_path)

    local_summary["total_rows"] = len(rows)
    local_summary["env_counts"] = counter_object([row.get("env", "unknown") for row in rows])
    return rows, local_summary
