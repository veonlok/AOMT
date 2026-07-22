"""Validation and scoring for closed-loop observation rollout traces."""

from __future__ import annotations

import collections
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from .metrics import normalize_text, summarize, text_metrics
from .schema import RESULT_SCHEMA_VERSION


def validate_rollout(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["steps must be a non-empty list"]
    requested_horizon = record.get("horizon")
    if not isinstance(requested_horizon, int) or requested_horizon <= 0:
        errors.append("horizon must be a positive integer")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"step {index} is not an object")
            continue
        if not isinstance(step.get("prediction"), str) or not isinstance(
            step.get("target"), str
        ):
            errors.append(f"step {index} requires string prediction and target")
        if step.get("future_ground_truth_used") is not False:
            errors.append(
                f"step {index} must explicitly set future_ground_truth_used=false"
            )
        expected_source = "ground_truth_initial" if index == 0 else "model"
        if step.get("observation_context_source") != expected_source:
            errors.append(
                f"step {index} observation_context_source must be {expected_source!r}"
            )
    return errors


def score_rollouts(
    records: Iterable[Dict[str, Any]],
    *,
    strict: bool = True,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    valid_records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, record in enumerate(records):
        rollout_id = str(record.get("rollout_id", f"row-{ordinal}"))
        if rollout_id in seen:
            raise ValueError(f"Duplicate rollout_id: {rollout_id}")
        seen.add(rollout_id)
        errors = validate_rollout(record)
        if errors:
            invalid_records.append({"rollout_id": rollout_id, "errors": errors})
        else:
            valid_records.append(record)
    if strict and invalid_records:
        first = invalid_records[0]
        raise ValueError(
            f"Invalid rollout {first['rollout_id']}: {'; '.join(first['errors'])}"
        )

    step_rows: List[Dict[str, Any]] = []
    rollout_rows: List[Dict[str, Any]] = []
    for record in valid_records:
        per_step = [
            text_metrics(step["prediction"], step["target"]) for step in record["steps"]
        ]
        validity = [
            bool(step["state_valid"])
            if isinstance(step.get("state_valid"), bool)
            else bool(metrics["exact_match"])
            for step, metrics in zip(record["steps"], per_step)
        ]
        first_error = next(
            (index + 1 for index, valid in enumerate(validity) if not valid), None
        )
        repeated = [
            normalize_text(record["steps"][index]["prediction"])
            == normalize_text(record["steps"][index - 1]["prediction"])
            for index in range(1, len(record["steps"]))
        ]
        invalid_actions = [
            not step["valid_action"]
            for step in record["steps"]
            if isinstance(step.get("valid_action"), bool)
        ]
        requested = int(record["horizon"])
        rollout_rows.append(
            {
                "rollout_id": record["rollout_id"],
                "horizon": requested,
                "completion": min(1.0, len(per_step) / requested),
                "all_steps_exact": float(
                    len(per_step) >= requested
                    and all(row["exact_match"] for row in per_step[:requested])
                ),
                "mean_token_f1": sum(row["token_f1"] for row in per_step)
                / len(per_step),
                "first_token_f1": per_step[0]["token_f1"],
                "final_token_f1": per_step[-1]["token_f1"],
                "time_to_first_error": float(
                    first_error if first_error is not None else requested + 1
                ),
                "survived_horizon": float(
                    len(validity) >= requested and all(validity[:requested])
                ),
                "repeated_observation_rate": (
                    sum(repeated) / len(repeated) if repeated else 0.0
                ),
                "invalid_action_rate": (
                    sum(invalid_actions) / len(invalid_actions)
                    if invalid_actions
                    else None
                ),
                "validity_basis": (
                    "state_valid"
                    if all("state_valid" in step for step in record["steps"])
                    else "exact_text_proxy"
                ),
            }
        )
        for index, metrics in enumerate(per_step, 1):
            step_rows.append({"horizon": requested, "position": index, **metrics})

    by_horizon: Dict[str, Any] = {}
    grouped: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rollout_rows:
        grouped[row["horizon"]].append(row)
    for horizon, rows in sorted(grouped.items()):
        by_horizon[str(horizon)] = {
            "rollouts": len(rows),
            "completion": summarize(
                [row["completion"] for row in rows], bootstrap_samples, seed
            ),
            "all_steps_exact": summarize(
                [row["all_steps_exact"] for row in rows], bootstrap_samples, seed
            ),
            "mean_token_f1": summarize(
                [row["mean_token_f1"] for row in rows], bootstrap_samples, seed
            ),
            "final_token_f1": summarize(
                [row["final_token_f1"] for row in rows], bootstrap_samples, seed
            ),
            "error_accumulation": summarize(
                [row["final_token_f1"] - row["first_token_f1"] for row in rows],
                bootstrap_samples,
                seed,
            ),
            "time_to_first_error": summarize(
                [row["time_to_first_error"] for row in rows], bootstrap_samples, seed
            ),
            "survival_rate": summarize(
                [row["survived_horizon"] for row in rows], bootstrap_samples, seed
            ),
            "repeated_observation_rate": summarize(
                [row["repeated_observation_rate"] for row in rows],
                bootstrap_samples,
                seed,
            ),
            "invalid_action_rate": summarize(
                [
                    row["invalid_action_rate"]
                    for row in rows
                    if row["invalid_action_rate"] is not None
                ],
                bootstrap_samples,
                seed,
            ),
            "validity_bases": sorted({row["validity_basis"] for row in rows}),
        }

    by_position: Dict[str, Any] = {}
    position_groups: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in step_rows:
        position_groups[row["position"]].append(row)
    for position, rows in sorted(position_groups.items()):
        by_position[str(position)] = {
            "examples": len(rows),
            "token_f1": summarize(
                [row["token_f1"] for row in rows], bootstrap_samples, seed
            ),
            "exact_match": summarize(
                [row["exact_match"] for row in rows], bootstrap_samples, seed
            ),
        }

    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "evaluation": "closed_loop_observation_rollout",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "valid_rollouts": len(valid_records),
            "invalid_rollouts": len(invalid_records),
            "by_horizon": by_horizon,
            "by_position": by_position,
        },
        invalid_records,
    )
