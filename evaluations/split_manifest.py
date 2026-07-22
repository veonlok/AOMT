"""Create explicit ID/OOD manifests from novelty relative to the training set."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Sequence

from .metrics import normalize_text
from .schema import file_sha256, read_jsonl, stable_hash, write_json


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
LETTERED_ENTITY_RE = re.compile(r"\b(unknown substance|box|object|item)\s+[a-z0-9]+\b")
SPACE_RE = re.compile(r"\s+")


def goal_template(goal: str) -> str:
    value = normalize_text(goal)
    value = LETTERED_ENTITY_RE.sub(lambda match: f"{match.group(1)} <id>", value)
    value = NUMBER_RE.sub("<num>", value)
    return SPACE_RE.sub(" ", value).strip()


def task_group(row: Dict[str, Any], strategy: str) -> str:
    labels = (
        row.get("state_labels") if isinstance(row.get("state_labels"), dict) else {}
    )
    if strategy == "task_family":
        value = labels.get("task_type") or (row.get("metadata") or {}).get("task_type")
        if value:
            return f"task_family:{value}"
        return f"goal_template:{goal_template(str(row.get('goal', '')))}"
    if strategy == "goal_template":
        return f"goal_template:{goal_template(str(row.get('goal', '')))}"
    if strategy == "scene":
        scene = labels.get("scene")
        if scene is None:
            raise ValueError("scene strategy requires state_labels.scene")
        return f"scene:{stable_hash(scene)[:16]}"
    raise ValueError(f"Unsupported split strategy {strategy!r}")


def _length_threshold(rows: Sequence[Dict[str, Any]], quantile: float) -> int:
    lengths = sorted(len(row.get("blocks") or []) for row in rows)
    if not lengths:
        raise ValueError("Training rows are empty")
    index = min(len(lengths) - 1, max(0, math.ceil(quantile * len(lengths)) - 1))
    return lengths[index]


def build_novelty_manifest(
    train_rows: Sequence[Dict[str, Any]],
    evaluation_rows: Iterable[Dict[str, Any]],
    *,
    environment: str,
    strategy: str = "task_family",
    length_quantile: float = 0.95,
) -> Dict[str, Any]:
    if strategy == "length":
        threshold = _length_threshold(train_rows, length_quantile)
        train_groups = {f"blocks_le_{threshold}"}
    else:
        threshold = None
        train_groups = {task_group(row, strategy) for row in train_rows}
    assignments = []
    for row in evaluation_rows:
        trajectory_id = str(row.get("trajectory_id", "")).strip()
        if not trajectory_id:
            raise ValueError("Every row needs trajectory_id")
        if strategy == "length":
            block_count = len(row.get("blocks") or [])
            is_ood = block_count > int(threshold)
            group = f"blocks_{'gt' if is_ood else 'le'}_{threshold}"
            reason = f"trajectory_length_{'above' if is_ood else 'within'}_train_p{length_quantile:g}"
        else:
            group = task_group(row, strategy)
            is_ood = group not in train_groups
            reason = f"{'unseen' if is_ood else 'seen'}_{strategy}"
        assignments.append(
            {
                "trajectory_id": trajectory_id,
                "source_split": str(row.get("split", "unknown")),
                "eval_split": "ood" if is_ood else "id",
                "novelty_group": group,
                "novelty_reason": reason,
                "task_id": group,
            }
        )
    counts = Counter(item["eval_split"] for item in assignments)
    return {
        "schema_version": "dflex-id-ood-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "strategy": strategy,
        "length_quantile": length_quantile if strategy == "length" else None,
        "length_threshold": threshold,
        "train_group_count": len(train_groups),
        "counts": dict(sorted(counts.items())),
        "assignments_sha256": stable_hash(assignments),
        "assignments": assignments,
    }


def create_novelty_manifest(
    train_path: str,
    evaluation_paths: Sequence[str],
    output_path: str,
    *,
    environment: str,
    strategy: str = "task_family",
    length_quantile: float = 0.95,
) -> Dict[str, Any]:
    train_rows = list(read_jsonl(train_path))
    evaluation_rows = [row for path in evaluation_paths for row in read_jsonl(path)]
    manifest = build_novelty_manifest(
        train_rows,
        evaluation_rows,
        environment=environment,
        strategy=strategy,
        length_quantile=length_quantile,
    )
    manifest["sources"] = {
        "train": {"path": train_path, "sha256": file_sha256(train_path)},
        "evaluation": [
            {"path": path, "sha256": file_sha256(path)} for path in evaluation_paths
        ],
    }
    write_json(output_path, manifest)
    return manifest


def load_assignments(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != "dflex-id-ood-manifest-v1":
        raise ValueError(
            f"Unsupported split manifest: {manifest.get('schema_version')!r}"
        )
    result = {}
    for item in manifest.get("assignments", []):
        trajectory_id = str(item["trajectory_id"])
        if trajectory_id in result:
            raise ValueError(f"Duplicate split assignment for {trajectory_id}")
        result[trajectory_id] = item
    return result
