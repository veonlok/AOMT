"""Build deterministic, leakage-controlled transition evaluation fixtures."""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import (
    FIXTURE_SCHEMA_VERSION,
    TransitionExample,
    file_sha256,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)


def _step(block: Dict[str, Any]) -> int:
    try:
        return int(block.get("step", -1))
    except (TypeError, ValueError):
        return -1


def _leaky_think_steps(row: Dict[str, Any]) -> set[int]:
    result: set[int] = set()
    for flag in row.get("leakage_flags") or []:
        if isinstance(flag, dict) and "step" in flag:
            try:
                result.add(int(flag["step"]))
            except (TypeError, ValueError):
                pass
    return result


def _clean_blocks(
    row: Dict[str, Any], drop_leaky_think: bool
) -> List[Tuple[int, Dict[str, Any]]]:
    leaky_steps = _leaky_think_steps(row) if drop_leaky_think else set()
    clean: List[Tuple[int, Dict[str, Any]]] = []
    for original_index, raw in enumerate(row.get("blocks") or []):
        if not isinstance(raw, dict):
            continue
        block_type = raw.get("type")
        if block_type not in {"Goal", "Think", "Action", "Observation"}:
            continue
        if block_type == "Think" and _step(raw) in leaky_steps:
            continue
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        clean.append((original_index, dict(raw)))
    return clean


def _metadata_value(row: Dict[str, Any], *keys: str) -> Optional[str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        value = row.get(key)
        if value is None:
            value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _eval_split(row: Dict[str, Any]) -> Optional[str]:
    value = _metadata_value(row, "eval_split", "distribution", "ood_split")
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "in_distribution": "id",
        "indistribution": "id",
        "in": "id",
        "out_of_distribution": "ood",
        "outofdistribution": "ood",
        "out": "ood",
    }
    return aliases.get(normalized, normalized)


def _state_at_step(
    row: Dict[str, Any], step: int, block: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    block_labels = block.get("state_labels")
    if isinstance(block_labels, dict) and block_labels:
        return block_labels

    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    for key in ("state", "state_after", "structured_state"):
        value = metadata.get(key)
        if isinstance(value, dict) and value:
            return value

    labels = row.get("state_labels")
    if not isinstance(labels, dict) or not labels:
        return None
    by_step = (
        labels.get("by_step") if isinstance(labels.get("by_step"), dict) else labels
    )
    for key in (str(step), step):
        value = by_step.get(key)
        if isinstance(value, dict) and value:
            return value
    return None


def _transition_candidates(
    row: Dict[str, Any], drop_leaky_think: bool
) -> List[
    Tuple[
        List[Dict[str, Any]],
        Dict[str, Any],
        Dict[str, Any],
        int,
        Optional[Dict[str, Any]],
    ]
]:
    blocks = _clean_blocks(row, drop_leaky_think)
    candidates = []
    previous_observation: Optional[Dict[str, Any]] = None
    for position, (original_index, block) in enumerate(blocks):
        if block.get("type") != "Observation":
            continue
        action: Optional[Dict[str, Any]] = None
        for _, prior in reversed(blocks[:position]):
            if prior.get("type") == "Observation":
                break
            if prior.get("type") == "Action":
                action = prior
                break
        if action is not None:
            history = [dict(item) for _, item in blocks[:position]]
            candidates.append(
                (
                    history,
                    dict(action),
                    dict(block),
                    original_index,
                    previous_observation,
                )
            )
        previous_observation = dict(block)
    return candidates


def _action_category(text: str) -> str:
    value = str(text).strip().lower()
    return value.split(maxsplit=1)[0] if value else "empty"


def _length_bin(length: int) -> str:
    if length < 128:
        return "short"
    if length < 512:
        return "medium"
    return "long"


def build_transition_examples(
    rows: Iterable[Dict[str, Any]],
    mode: str = "last",
    drop_leaky_think: bool = True,
    max_examples: Optional[int] = None,
    assignments: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[TransitionExample]:
    """Create stable causal fixtures from typed trajectories.

    `last` mirrors the training script's shared next-observation evaluation. `all` emits every
    action/observation transition and is intended for state-level and per-action analysis.
    """

    if mode not in {"last", "all"}:
        raise ValueError("mode must be 'last' or 'all'")

    examples: List[TransitionExample] = []
    seen_ids: set[str] = set()
    sorted_rows = sorted(rows, key=lambda row: str(row.get("trajectory_id", "")))
    for row in sorted_rows:
        trajectory_id = str(row.get("trajectory_id", "")).strip()
        if not trajectory_id:
            raise ValueError("Every trajectory must have a non-empty trajectory_id")
        candidates = _transition_candidates(row, drop_leaky_think)
        assignment = (assignments or {}).get(trajectory_id)
        if mode == "last" and candidates:
            candidates = candidates[-1:]
        for (
            history,
            action,
            observation,
            block_index,
            previous_observation,
        ) in candidates:
            target_step = _step(observation)
            context_value = {
                "goal": row.get("goal", ""),
                "history": history,
                "action": action,
            }
            example_id = f"{trajectory_id}:obs:{block_index}"
            if example_id in seen_ids:
                raise ValueError(f"Duplicate transition example ID: {example_id}")
            seen_ids.add(example_id)
            previous_state = None
            if previous_observation is not None:
                previous_state = _state_at_step(
                    row, _step(previous_observation), previous_observation
                )
            target_state = _state_at_step(row, target_step, observation)
            examples.append(
                TransitionExample(
                    example_id=example_id,
                    trajectory_id=trajectory_id,
                    environment=str(row.get("env", "unknown")),
                    source_split=str(row.get("split", "unknown")),
                    target_step=target_step,
                    goal=str(row.get("goal", "")),
                    history=history,
                    action=action,
                    target_observation=str(observation["text"]),
                    target_block_index=block_index,
                    context_hash=stable_hash(context_value),
                    task_id=(
                        str(assignment["task_id"])
                        if assignment and assignment.get("task_id") is not None
                        else _metadata_value(row, "task_id", "task_type", "task_name")
                    ),
                    eval_split=(
                        str(assignment["eval_split"])
                        if assignment and assignment.get("eval_split") is not None
                        else _eval_split(row)
                    ),
                    previous_state=previous_state,
                    target_state=target_state,
                    metadata={
                        "fixture_mode": mode,
                        "leaky_think_removed": drop_leaky_think,
                        "action_step": _step(action),
                        "action_category": _action_category(action.get("text", "")),
                        "context_blocks": len(history),
                        "context_length_bin": _length_bin(
                            sum(len(block.get("text", "")) for block in history)
                        ),
                        "target_characters": len(str(observation["text"])),
                        "target_length_bin": _length_bin(len(str(observation["text"]))),
                        "source_has_leakage_flags": bool(row.get("leakage_flags")),
                        "action_valid": (
                            action.get("metadata", {}).get("valid")
                            if isinstance(action.get("metadata"), dict)
                            else None
                        ),
                        "transition_validity_source": (
                            action.get("metadata", {}).get("validity_source")
                            if isinstance(action.get("metadata"), dict)
                            else None
                        ),
                        "simulator_revision": (
                            observation.get("metadata", {}).get("simulator_revision")
                            if isinstance(observation.get("metadata"), dict)
                            else None
                        ),
                        "reward": (
                            observation.get("metadata", {}).get("reward")
                            if isinstance(observation.get("metadata"), dict)
                            else None
                        ),
                        "completion": (
                            observation.get("metadata", {}).get("completion")
                            if isinstance(observation.get("metadata"), dict)
                            else None
                        ),
                        "novelty_group": assignment.get("novelty_group")
                        if assignment
                        else None,
                        "novelty_reason": assignment.get("novelty_reason")
                        if assignment
                        else None,
                    },
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


def fixture_manifest(
    examples: Sequence[TransitionExample],
    input_path: str,
    mode: str,
    drop_leaky_think: bool,
) -> Dict[str, Any]:
    counts = collections.Counter(example.environment for example in examples)
    split_counts = collections.Counter(
        example.eval_split or "unlabelled" for example in examples
    )
    task_counts = collections.Counter(
        example.task_id or "unlabelled" for example in examples
    )
    state_count = sum(example.target_state is not None for example in examples)
    history_lengths = [len(example.history) for example in examples]
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "input_path": str(Path(input_path)),
        "input_sha256": file_sha256(input_path),
        "fixture_mode": mode,
        "drop_leaky_think": drop_leaky_think,
        "example_count": len(examples),
        "example_ids_sha256": stable_hash([example.example_id for example in examples]),
        "environment_counts": dict(sorted(counts.items())),
        "eval_split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "structured_state_coverage": state_count / len(examples) if examples else 0.0,
        "history_blocks": {
            "min": min(history_lengths) if history_lengths else 0,
            "max": max(history_lengths) if history_lengths else 0,
            "mean": sum(history_lengths) / len(history_lengths)
            if history_lengths
            else 0.0,
        },
    }


def prepare_fixture(
    input_path: str,
    output_path: str,
    mode: str = "last",
    drop_leaky_think: bool = True,
    max_examples: Optional[int] = None,
    manifest_path: Optional[str] = None,
    split_manifest_path: Optional[str] = None,
    tokenizer: Any = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    assignments = None
    if split_manifest_path:
        from .split_manifest import load_assignments

        assignments = load_assignments(split_manifest_path)
    examples = build_transition_examples(
        read_jsonl(input_path),
        mode=mode,
        drop_leaky_think=drop_leaky_think,
        max_examples=None if tokenizer is not None else max_examples,
        assignments=assignments,
    )
    dropped_over_token_budget = 0
    token_lengths: List[int] = []
    if tokenizer is not None:
        if max_tokens is None or max_tokens <= 0:
            raise ValueError("max_tokens must be positive when tokenizer is supplied")
        filtered = []
        for example in examples:
            length = sum(
                len(tokenizer.encode(block["text"], add_special_tokens=False))
                for block in example.history
            ) + len(
                tokenizer.encode(example.target_observation, add_special_tokens=False)
            )
            if length > max_tokens:
                dropped_over_token_budget += 1
                continue
            token_lengths.append(length)
            filtered.append(example)
        examples = filtered[:max_examples] if max_examples is not None else filtered
        token_lengths = token_lengths[: len(examples)]
    if not examples:
        raise ValueError(
            f"No valid action-to-observation transitions found in {input_path}"
        )
    write_jsonl(output_path, (example.to_dict() for example in examples))
    manifest = fixture_manifest(examples, input_path, mode, drop_leaky_think)
    manifest["token_filter"] = {
        "enabled": tokenizer is not None,
        "max_tokens": max_tokens,
        "dropped_over_token_budget": dropped_over_token_budget,
        "min_tokens": min(token_lengths) if token_lengths else None,
        "max_observed_tokens": max(token_lengths) if token_lengths else None,
        "mean_tokens": (
            sum(token_lengths) / len(token_lengths) if token_lengths else None
        ),
    }
    manifest["split_manifest_path"] = split_manifest_path
    if split_manifest_path:
        manifest["split_manifest_sha256"] = file_sha256(split_manifest_path)
    write_json(manifest_path or f"{output_path}.manifest.json", manifest)
    return manifest
