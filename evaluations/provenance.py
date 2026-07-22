"""Reproducibility metadata for datasets, fixtures, models, and evaluation runs."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schema import file_sha256, write_json


def _git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_provenance(
    *,
    fixture_path: Optional[str] = None,
    checkpoint: Optional[str] = None,
    dataset_manifest: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = _git("status", "--porcelain")
    result: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "numpy",
                "torch",
                "transformers",
                "peft",
                "scikit-learn",
                "scipy",
            )
        },
        "checkpoint": checkpoint,
    }
    if fixture_path:
        result["fixture"] = {
            "path": fixture_path,
            "sha256": file_sha256(fixture_path),
        }
        fixture_manifest = f"{fixture_path}.manifest.json"
        if os.path.exists(fixture_manifest):
            result["fixture"]["manifest_path"] = fixture_manifest
            result["fixture"]["manifest_sha256"] = file_sha256(fixture_manifest)
    if dataset_manifest:
        result["dataset_manifest"] = {
            "path": dataset_manifest,
            "sha256": file_sha256(dataset_manifest),
        }
    if extra:
        result["extra"] = extra
    return result


def create_dataset_manifest(
    data_root: str,
    output_path: str,
    *,
    repo_id: str,
    revision: str,
    environments: Iterable[str] = ("scienceworld", "alfworld", "webshop"),
    splits: Iterable[str] = ("train", "validation", "test"),
) -> Dict[str, Any]:
    root = Path(data_root)
    files: Dict[str, Any] = {}
    for environment in environments:
        for split in splits:
            path = root / environment / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            rows = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number}: expected object")
                    rows += 1
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "rows": rows,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    manifest = {
        "schema_version": "dflex-dataset-provenance-v1",
        "repo_id": repo_id,
        "revision": revision,
        "data_root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": dict(sorted(files.items())),
    }
    write_json(output_path, manifest)
    return manifest
