"""Score per-example model predictions against frozen transition fixtures."""

from __future__ import annotations

import collections
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .metrics import state_metrics, summarize, text_metrics
from .schema import RESULT_SCHEMA_VERSION, TransitionExample, stable_hash


TEXT_METRICS = (
    "exact_match",
    "token_precision",
    "token_recall",
    "token_f1",
    "character_similarity",
    "number_set_match",
)
STATE_METRICS = (
    "state_key_accuracy",
    "state_key_coverage",
    "state_exact_match",
    "changed_key_accuracy",
    "stable_key_accuracy",
)


def _prediction_index(
    predictions: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for prediction in predictions:
        example_id = str(prediction.get("example_id", "")).strip()
        if not example_id:
            raise ValueError("Every prediction must have a non-empty example_id")
        if example_id in index:
            raise ValueError(f"Duplicate prediction for {example_id}")
        index[example_id] = prediction
    return index


def _summaries(
    rows: Sequence[Dict[str, Any]], bootstrap_samples: int, seed: int
) -> Dict[str, Any]:
    result = {
        key: summarize(
            [row[key] for row in rows if key in row], bootstrap_samples, seed
        )
        for key in TEXT_METRICS + STATE_METRICS
    }
    nll_rows = [row for row in rows if isinstance(row.get("nll"), (int, float))]
    result["nll_per_example"] = summarize(
        [float(row["nll"]) for row in nll_rows], bootstrap_samples, seed
    )
    weighted_tokens = sum(int(row.get("target_tokens", 0)) for row in nll_rows)
    result["nll_per_token_micro"] = (
        sum(float(row["nll"]) * int(row.get("target_tokens", 0)) for row in nll_rows)
        / weighted_tokens
        if weighted_tokens
        else None
    )
    return result


def _group_results(
    scored: Sequence[Dict[str, Any]], bootstrap_samples: int, seed: int
) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "environment": collections.defaultdict(list),
        "eval_split": collections.defaultdict(list),
        "task_id": collections.defaultdict(list),
        "action_category": collections.defaultdict(list),
        "target_length_bin": collections.defaultdict(list),
        "context_length_bin": collections.defaultdict(list),
        "source_leakage": collections.defaultdict(list),
    }
    for row in scored:
        groups["environment"][row["environment"]].append(row)
        groups["eval_split"][row.get("eval_split") or "unlabelled"].append(row)
        groups["task_id"][row.get("task_id") or "unlabelled"].append(row)
        groups["action_category"][row.get("action_category") or "unlabelled"].append(
            row
        )
        groups["target_length_bin"][
            row.get("target_length_bin") or "unlabelled"
        ].append(row)
        groups["context_length_bin"][
            row.get("context_length_bin") or "unlabelled"
        ].append(row)
        groups["source_leakage"][row.get("source_leakage") or "clean"].append(row)
    return {
        dimension: {
            name: {
                "examples": len(items),
                "metrics": _summaries(items, bootstrap_samples, seed),
            }
            for name, items in sorted(values.items())
        }
        for dimension, values in groups.items()
    }


def score_predictions(
    fixtures: Sequence[TransitionExample],
    predictions: Iterable[Dict[str, Any]],
    *,
    allow_missing: bool = False,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prediction_by_id = _prediction_index(predictions)
    fixture_ids = {fixture.example_id for fixture in fixtures}
    unexpected = sorted(set(prediction_by_id) - fixture_ids)
    missing = sorted(fixture_ids - set(prediction_by_id))
    if missing and not allow_missing:
        raise ValueError(
            f"Missing predictions for {len(missing)} fixtures; first IDs: {missing[:5]}"
        )

    scored: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for fixture in fixtures:
        prediction_record = prediction_by_id.get(fixture.example_id)
        if prediction_record is None:
            continue
        prediction_context_hash = prediction_record.get("context_hash")
        if (
            prediction_context_hash is not None
            and prediction_context_hash != fixture.context_hash
        ):
            raise ValueError(
                f"Prediction {fixture.example_id} was generated from a different context hash"
            )
        prediction_text = prediction_record.get("prediction")
        if not isinstance(prediction_text, str):
            raise ValueError(
                f"Prediction {fixture.example_id} has no string 'prediction'"
            )
        row: Dict[str, Any] = {
            "example_id": fixture.example_id,
            "trajectory_id": fixture.trajectory_id,
            "environment": fixture.environment,
            "eval_split": fixture.eval_split,
            "task_id": fixture.task_id,
            "action_category": fixture.metadata.get("action_category"),
            "target_length_bin": fixture.metadata.get("target_length_bin"),
            "context_length_bin": fixture.metadata.get("context_length_bin"),
            "source_leakage": (
                "flagged"
                if fixture.metadata.get("source_has_leakage_flags")
                else "clean"
            ),
            **text_metrics(prediction_text, fixture.target_observation),
        }
        if isinstance(prediction_record.get("nll"), (int, float)):
            row["nll"] = float(prediction_record["nll"])
            row["target_tokens"] = int(prediction_record.get("target_tokens", 0))
        predicted_state = prediction_record.get("predicted_state")
        if fixture.target_state is not None and isinstance(predicted_state, Mapping):
            row.update(
                state_metrics(
                    predicted_state, fixture.target_state, fixture.previous_state
                )
            )
        for field in ("valid_preconditions", "rule_consistent", "valid_action"):
            if isinstance(prediction_record.get(field), bool):
                row[field] = float(prediction_record[field])
        scored.append(row)
        if row["exact_match"] < 1.0:
            failures.append(
                {
                    "example_id": fixture.example_id,
                    "trajectory_id": fixture.trajectory_id,
                    "action": fixture.action.get("text", ""),
                    "prediction": prediction_text,
                    "target": fixture.target_observation,
                    "token_f1": row["token_f1"],
                    "context_hash": fixture.context_hash,
                }
            )

    validation_metrics: Dict[str, Any] = {}
    for field in ("valid_preconditions", "rule_consistent", "valid_action"):
        validation_metrics[field] = summarize(
            [row[field] for row in scored if field in row], bootstrap_samples, seed
        )

    state_examples = sum("state_key_accuracy" in row for row in scored)
    report = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fixture_count": len(fixtures),
        "prediction_count": len(prediction_by_id),
        "scored_count": len(scored),
        "coverage": len(scored) / len(fixtures) if fixtures else 0.0,
        "missing_example_ids": missing,
        "unexpected_example_ids": unexpected,
        "fixture_ids_sha256": stable_hash([fixture.example_id for fixture in fixtures]),
        "structured_state": {
            "fixture_coverage": (
                sum(fixture.target_state is not None for fixture in fixtures)
                / len(fixtures)
                if fixtures
                else 0.0
            ),
            "scored_examples": state_examples,
            "status": "available"
            if state_examples
            else "unavailable_no_labels_or_predictions",
        },
        "metrics": _summaries(scored, bootstrap_samples, seed),
        "environment_validation": validation_metrics,
        "groups": _group_results(scored, bootstrap_samples, seed),
    }
    failures.sort(key=lambda item: (item["token_f1"], item["example_id"]))
    return report, failures
