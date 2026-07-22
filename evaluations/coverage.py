"""Audit whether each dataset can support each claimed evaluation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Sequence

from .schema import RESULT_SCHEMA_VERSION, file_sha256, read_jsonl, write_json


def _state_labels(row: Dict[str, Any], block: Dict[str, Any]) -> Dict[str, Any]:
    direct = block.get("state_labels")
    if isinstance(direct, dict) and direct:
        return direct
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    for key in ("state", "state_after", "structured_state"):
        value = metadata.get(key)
        if isinstance(value, dict) and value:
            return value
    labels = row.get("state_labels")
    if not isinstance(labels, dict) or not labels:
        return {}
    by_step = (
        labels.get("by_step") if isinstance(labels.get("by_step"), dict) else labels
    )
    step = block.get("step", -1)
    for key in (str(step), step):
        value = by_step.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def audit_rows(rows: Sequence[Dict[str, Any]], environment: str) -> Dict[str, Any]:
    trajectories_with_transitions = set()
    metadata_flags = Counter()
    trajectory_labels = 0
    transitions = 0
    step_state = 0
    validated = 0
    simulator_revision = 0
    label_keys = set()
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        for key, value in metadata.items():
            if isinstance(value, bool) and value:
                metadata_flags[key] += 1
        if isinstance(row.get("state_labels"), dict) and row["state_labels"]:
            trajectory_labels += 1

        pending_action = None
        for block in row.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "Action":
                pending_action = block
                continue
            if block.get("type") != "Observation" or pending_action is None:
                continue
            transitions += 1
            trajectories_with_transitions.add(str(row.get("trajectory_id", "")))
            labels = _state_labels(row, block)
            if labels:
                step_state += 1
                label_keys.update(labels)
            action_metadata = (
                pending_action.get("metadata")
                if isinstance(pending_action.get("metadata"), dict)
                else {}
            )
            observation_metadata = (
                block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
            )
            if action_metadata.get("valid") is True and action_metadata.get(
                "validity_source"
            ):
                validated += 1
            if observation_metadata.get("simulator_revision"):
                simulator_revision += 1
            pending_action = None

    return {
        "environment": environment,
        "trajectories": len(rows),
        "trajectories_with_action_observation_transition": len(
            trajectories_with_transitions
        ),
        "trajectory_transition_coverage": (
            len(trajectories_with_transitions) / len(rows) if rows else 0.0
        ),
        "transitions": transitions,
        "trajectory_label_coverage": trajectory_labels / len(rows) if rows else 0.0,
        "step_aligned_state_coverage": step_state / transitions if transitions else 0.0,
        "transition_validity_coverage": validated / transitions if transitions else 0.0,
        "simulator_revision_coverage": (
            simulator_revision / transitions if transitions else 0.0
        ),
        "step_state_label_keys": sorted(label_keys),
        "metadata_flags": dict(sorted(metadata_flags.items())),
        "capabilities": {
            "next_observation": transitions > 0,
            "structured_next_state": step_state > 0,
            "transition_validity_labels": validated > 0,
            "closed_loop_offline_targets": transitions >= 3,
            "simulator_task_success": False,
            "simulator_counterfactuals": False,
        },
    }


def audit_datasets(
    inputs: Iterable[tuple[str, str]], output_path: str
) -> Dict[str, Any]:
    environments = {}
    sources = {}
    for environment, path in inputs:
        rows = list(read_jsonl(path))
        environments[environment] = audit_rows(rows, environment)
        sources[environment] = {"path": path, "sha256": file_sha256(path)}
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation": "dataset_evaluation_readiness",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "environments": environments,
        "guard": (
            "A capability is false unless the downloaded data directly supports it; simulator "
            "claims require separately installed and validated adapters."
        ),
    }
    write_json(output_path, report)
    return report
