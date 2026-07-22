"""Deterministic robustness fixtures that remain separate from model predictions."""

from __future__ import annotations

import copy
import random
import re
from typing import Any, Dict, Iterable, List, Sequence

from .schema import TransitionExample, stable_hash, write_jsonl


NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")
CORRUPTION_TYPES = (
    "deleted_block",
    "duplicated_block",
    "swapped_observations",
    "contradictory_observation",
    "numeric_value",
    "entity_name",
)


def _non_goal_indices(history: Sequence[Dict[str, Any]]) -> List[int]:
    return [index for index, block in enumerate(history) if block.get("type") != "Goal"]


def _observation_indices(history: Sequence[Dict[str, Any]]) -> List[int]:
    return [
        index
        for index, block in enumerate(history)
        if block.get("type") == "Observation"
    ]


def _replace_number(text: str) -> tuple[str, bool]:
    match = NUMBER_RE.search(text)
    if not match:
        return text, False
    value = float(match.group())
    replacement = str(int(value + 1)) if value.is_integer() else str(value + 1.0)
    return text[: match.start()] + replacement + text[match.end() :], True


def _replace_entity(text: str) -> tuple[str, bool]:
    ignored = {
        "the",
        "you",
        "your",
        "this",
        "that",
        "with",
        "from",
        "into",
        "observation",
    }
    for match in WORD_RE.finditer(text):
        if match.group().lower() not in ignored:
            return text[: match.start()] + "corrupted_entity" + text[
                match.end() :
            ], True
    return text, False


def corrupt_history(
    history: Sequence[Dict[str, Any]], corruption_type: str, rng: random.Random
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if corruption_type not in CORRUPTION_TYPES:
        raise ValueError(f"Unsupported corruption type {corruption_type!r}")
    blocks = copy.deepcopy(list(history))
    eligible = _non_goal_indices(blocks)
    observations = _observation_indices(blocks)
    if not eligible:
        raise ValueError("History has no corruptible blocks")
    details: Dict[str, Any] = {"corruption_type": corruption_type}
    if corruption_type == "deleted_block":
        index = rng.choice(eligible)
        details["removed"] = blocks.pop(index)
        details["index"] = index
    elif corruption_type == "duplicated_block":
        index = rng.choice(eligible)
        blocks.insert(index + 1, copy.deepcopy(blocks[index]))
        details["index"] = index
    elif corruption_type == "swapped_observations":
        if len(observations) < 2:
            raise ValueError("Need two observations to swap")
        left, right = observations[-2:]
        blocks[left], blocks[right] = blocks[right], blocks[left]
        details["indices"] = [left, right]
    elif corruption_type == "contradictory_observation":
        index = observations[-1] if observations else rng.choice(eligible)
        original = blocks[index]["text"]
        blocks[index]["text"] = (
            f"Contradiction: this observation is impossible. {original}"
        )
        details["index"] = index
    elif corruption_type == "numeric_value":
        candidates = [
            index for index in eligible if NUMBER_RE.search(blocks[index]["text"])
        ]
        if not candidates:
            raise ValueError("History has no numeric value")
        index = rng.choice(candidates)
        original = blocks[index]["text"]
        blocks[index]["text"], _ = _replace_number(original)
        details.update(
            {"index": index, "original": original, "corrupted": blocks[index]["text"]}
        )
    else:
        candidates = [
            index for index in eligible if WORD_RE.search(blocks[index]["text"])
        ]
        if not candidates:
            raise ValueError("History has no entity-like token")
        index = rng.choice(candidates)
        original = blocks[index]["text"]
        blocks[index]["text"], _ = _replace_entity(original)
        details.update(
            {"index": index, "original": original, "corrupted": blocks[index]["text"]}
        )
    return blocks, details


def generate_corruption_fixtures(
    fixtures: Sequence[TransitionExample],
    *,
    corruption_types: Sequence[str] = CORRUPTION_TYPES,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    output = []
    for fixture in fixtures:
        for corruption_type in corruption_types:
            try:
                corrupted, details = corrupt_history(
                    fixture.history, corruption_type, rng
                )
            except ValueError as exc:
                output.append(
                    {
                        "schema_version": "dflex-corruption-fixture-v1",
                        "case_id": f"{fixture.example_id}:{corruption_type}",
                        "example_id": fixture.example_id,
                        "corruption_type": corruption_type,
                        "status": "excluded",
                        "exclusion_reason": str(exc),
                    }
                )
                continue
            output.append(
                {
                    "schema_version": "dflex-corruption-fixture-v1",
                    "case_id": f"{fixture.example_id}:{corruption_type}",
                    "example_id": fixture.example_id,
                    "trajectory_id": fixture.trajectory_id,
                    "environment": fixture.environment,
                    "corruption_type": corruption_type,
                    "status": "ready",
                    "clean_context_hash": fixture.context_hash,
                    "corrupted_context_hash": stable_hash(corrupted),
                    "clean_history": fixture.history,
                    "corrupted_history": corrupted,
                    "action": fixture.action,
                    "target_observation": fixture.target_observation,
                    "details": details,
                    "simulator_grounded": False,
                }
            )
    return output


def generate_counterfactual_candidates(
    fixtures: Sequence[TransitionExample], *, seed: int = 0
) -> List[Dict[str, Any]]:
    """Generate minimal edits that must be simulator-validated before scoring."""

    rng = random.Random(seed)
    output = []
    for fixture in fixtures:
        history = copy.deepcopy(fixture.history)
        candidates = _non_goal_indices(history)
        if not candidates:
            continue
        rng.shuffle(candidates)
        edited = None
        edit_type = None
        details = None
        for index in candidates:
            replacement, changed = _replace_number(history[index]["text"])
            if changed:
                edited = copy.deepcopy(history)
                original = edited[index]["text"]
                edited[index]["text"] = replacement
                edit_type = "numeric_property"
                details = {"index": index, "original": original, "edited": replacement}
                break
        if edited is None:
            index = candidates[0]
            replacement, changed = _replace_entity(history[index]["text"])
            if not changed:
                continue
            edited = copy.deepcopy(history)
            original = edited[index]["text"]
            edited[index]["text"] = replacement
            edit_type = "entity_substitution"
            details = {"index": index, "original": original, "edited": replacement}
        output.append(
            {
                "schema_version": "dflex-counterfactual-candidate-v1",
                "case_id": f"{fixture.example_id}:{edit_type}",
                "example_id": fixture.example_id,
                "trajectory_id": fixture.trajectory_id,
                "environment": fixture.environment,
                "edit_type": edit_type,
                "base_history": fixture.history,
                "edited_history": edited,
                "action": fixture.action,
                "base_target": fixture.target_observation,
                "details": details,
                "requires_simulator_validation": True,
                "environment_valid": False,
                "status": "requires_simulator_validation",
                "matched_irrelevant_control_required": True,
            }
        )
    return output


def write_generated(path: str, records: Iterable[Dict[str, Any]]) -> None:
    write_jsonl(path, records)
