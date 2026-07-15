#!/usr/bin/env python3
"""Fetch the gated Hugging Face trajectory dataset into ``data/<env>/``.

The *extract* step for ``notebooks/trajectory_taxonomy.ipynb``: pulls the 9
per-environment / per-split ``jsonl`` files from the gated repo. Kept out of the
notebook so the ingestion logic is importable and testable, and so the notebook
cell is just a call plus its print summary.

The repo is gated — authenticate once (``HF_TOKEN`` env var, or ``hf auth login``)
with an account that has accepted the dataset's access terms.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ID   = "Joshyxwa/cp2107-textworld-trajectories"
REPO_ENVS = ["scienceworld", "alfworld", "webshop"]
SPLITS    = ["train", "validation", "test"]


def resolve_token():
    """Return ``(token, source)``. Currently: ``HF_TOKEN`` env var, else ``(None, "none")``.

    The token must be visible to the *exact* process running the caller (a value set
    in a different shell than the one that launched the kernel won't be seen)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"], "env HF_TOKEN"
    return None, "none"


def download_hf_dataset(data_dir, root=None, *, force: bool = False):
    """Download every ``<env>/<split>.jsonl`` into ``<data_dir>/<env>/``.

    Already-present files are skipped unless ``force``. Returns ``(got, failed, source)``
    where paths in ``got`` are reported relative to ``root`` (defaults to
    ``data_dir.parent``) for tidy printing, and ``failed`` is a list of
    ``(name, error_type)`` for files that could not be fetched (e.g. gated + no token)."""
    from huggingface_hub import hf_hub_download   # optional dep — only needed to fetch

    data_dir = Path(data_dir)
    root = Path(root) if root is not None else data_dir.parent
    token, source = resolve_token()
    got, failed = [], []
    for env in REPO_ENVS:
        for split in SPLITS:
            dest = data_dir / env / f"{split}.jsonl"
            if dest.exists() and not force:
                got.append(str(dest.relative_to(root)).replace("\\", "/"))
                continue
            try:
                cached = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                         filename=f"{env}/{split}.jsonl", token=token)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached, dest)
                got.append(str(dest.relative_to(root)).replace("\\", "/"))
            except Exception as e:   # noqa: BLE001 - report per-file and keep going
                failed.append((f"{env}/{split}.jsonl", type(e).__name__))
    return got, failed, source
