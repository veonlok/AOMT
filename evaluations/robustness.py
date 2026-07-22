"""Scoring contracts for counterfactual consistency and corruption recovery."""

from __future__ import annotations

import collections
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .metrics import normalize_text, state_metrics, summarize, text_metrics
from .schema import RESULT_SCHEMA_VERSION


def score_counterfactuals(
    records: Iterable[Dict[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 0
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Score simulator-validated base/counterfactual prediction pairs.

    Invalid or ambiguous interventions are excluded with reasons, as required by the protocol.
    Validity must come from the environment adapter, not the language model under evaluation.
    """

    rows: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, record in enumerate(records):
        case_id = str(record.get("case_id", f"row-{ordinal}"))
        if case_id in seen:
            raise ValueError(f"Duplicate counterfactual case_id: {case_id}")
        seen.add(case_id)
        if record.get("environment_valid") is not True:
            reason = str(record.get("exclusion_reason", "not validated by environment"))
            excluded.append({"case_id": case_id, "reason": reason})
            continue
        validity_source = str(record.get("validity_source", "")).strip()
        if not validity_source:
            raise ValueError(f"Counterfactual {case_id} needs validity_source")
        required = (
            "base_prediction",
            "base_target",
            "counterfactual_prediction",
            "counterfactual_target",
        )
        if any(not isinstance(record.get(field), str) for field in required):
            raise ValueError(
                f"Counterfactual {case_id} requires string base/counterfactual predictions and targets"
            )
        base = text_metrics(record["base_prediction"], record["base_target"])
        counterfactual = text_metrics(
            record["counterfactual_prediction"], record["counterfactual_target"]
        )
        expected_changed = normalize_text(record["base_target"]) != normalize_text(
            record["counterfactual_target"]
        )
        prediction_changed = normalize_text(
            record["base_prediction"]
        ) != normalize_text(record["counterfactual_prediction"])
        row: Dict[str, Any] = {
            "case_id": case_id,
            "environment": str(record.get("environment", "unknown")),
            "edit_type": str(record.get("edit_type", "unknown")),
            "base_token_f1": base["token_f1"],
            "counterfactual_token_f1": counterfactual["token_f1"],
            "counterfactual_exact_match": counterfactual["exact_match"],
            "expected_consequence_changed": float(expected_changed),
            "prediction_changed": float(prediction_changed),
            "change_sensitivity_correct": float(prediction_changed == expected_changed),
        }
        predicted_state = record.get("counterfactual_predicted_state")
        target_state = record.get("counterfactual_target_state")
        previous_state = record.get("base_target_state")
        if isinstance(predicted_state, Mapping) and isinstance(target_state, Mapping):
            row.update(state_metrics(predicted_state, target_state, previous_state))
        rows.append(row)

    groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[row["edit_type"]].append(row)

    metric_names = (
        "base_token_f1",
        "counterfactual_token_f1",
        "counterfactual_exact_match",
        "expected_consequence_changed",
        "prediction_changed",
        "change_sensitivity_correct",
        "state_key_accuracy",
        "changed_key_accuracy",
    )

    def aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            metric: summarize(
                [item[metric] for item in items if metric in item],
                bootstrap_samples,
                seed,
            )
            for metric in metric_names
        }

    return (
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "evaluation": "counterfactual_consistency",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "valid_cases": len(rows),
            "excluded_cases": len(excluded),
            "metrics": aggregate(rows),
            "by_edit_type": {
                name: {"cases": len(items), "metrics": aggregate(items)}
                for name, items in sorted(groups.items())
            },
        },
        excluded,
    )


def score_corruption_recovery(
    records: Iterable[Dict[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 0
) -> Dict[str, Any]:
    """Score recovery after an injected, typed trajectory corruption."""

    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, record in enumerate(records):
        case_id = str(record.get("case_id", f"row-{ordinal}"))
        if case_id in seen:
            raise ValueError(f"Duplicate corruption case_id: {case_id}")
        seen.add(case_id)
        corruption_type = str(record.get("corruption_type", "")).strip()
        if not corruption_type:
            raise ValueError(f"Corruption {case_id} needs corruption_type")
        steps = record.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"Corruption {case_id} needs non-empty steps")
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or not isinstance(step.get("coherent"), bool):
                raise ValueError(
                    f"Corruption {case_id} step {index} needs boolean coherent"
                )
            if not isinstance(step.get("prediction"), str) or not isinstance(
                step.get("target"), str
            ):
                raise ValueError(
                    f"Corruption {case_id} step {index} needs prediction and target"
                )
        recovered_index = next(
            (index for index, step in enumerate(steps) if step["coherent"]), None
        )
        quality = [
            text_metrics(step["prediction"], step["target"])["token_f1"]
            for step in steps
        ]
        post_recovery_quality = (
            quality[recovered_index:] if recovered_index is not None else []
        )
        rows.append(
            {
                "case_id": case_id,
                "corruption_type": corruption_type,
                "decode": str(record.get("decode", "unspecified")),
                "recovered": float(recovered_index is not None),
                "steps_to_recovery": float(recovered_index + 1)
                if recovered_index is not None
                else None,
                "post_recovery_token_f1": (
                    sum(post_recovery_quality) / len(post_recovery_quality)
                    if post_recovery_quality
                    else None
                ),
                "full_trace_token_f1": sum(quality) / len(quality),
            }
        )

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(row["corruption_type"], row["decode"])].append(row)

    def aggregate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "recovery_rate": summarize(
                [item["recovered"] for item in items], bootstrap_samples, seed
            ),
            "steps_to_recovery_recovered_only": summarize(
                [
                    item["steps_to_recovery"]
                    for item in items
                    if item["steps_to_recovery"] is not None
                ],
                bootstrap_samples,
                seed,
            ),
            "post_recovery_token_f1": summarize(
                [
                    item["post_recovery_token_f1"]
                    for item in items
                    if item["post_recovery_token_f1"] is not None
                ],
                bootstrap_samples,
                seed,
            ),
            "full_trace_token_f1": summarize(
                [item["full_trace_token_f1"] for item in items], bootstrap_samples, seed
            ),
        }

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation": "corruption_recovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(rows),
        "metrics": aggregate(rows),
        "by_corruption_and_decode": {
            f"{corruption_type}/{decode}": {
                "cases": len(items),
                "metrics": aggregate(items),
            }
            for (corruption_type, decode), items in sorted(groups.items())
        },
    }
