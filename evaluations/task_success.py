"""Aggregate ID/OOD task execution results without rewarding uniformly poor models."""

from __future__ import annotations

import collections
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .metrics import summarize
from .schema import RESULT_SCHEMA_VERSION


def _split_name(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "in_distribution": "id",
        "indistribution": "id",
        "in": "id",
        "out_of_distribution": "ood",
        "outofdistribution": "ood",
        "out": "ood",
    }
    result = aliases.get(normalized, normalized)
    if result not in {"id", "ood"}:
        raise ValueError(f"eval_split must be ID or OOD, got {value!r}")
    return result


def _normalized_score(record: Dict[str, Any]) -> float:
    if isinstance(record.get("normalized_score"), (int, float)):
        value = float(record["normalized_score"])
    elif isinstance(record.get("score"), (int, float)) and isinstance(
        record.get("max_score"), (int, float)
    ):
        maximum = float(record["max_score"])
        if maximum <= 0:
            raise ValueError("max_score must be positive")
        value = float(record["score"]) / maximum
    elif isinstance(record.get("success"), bool):
        value = float(record["success"])
    else:
        raise ValueError(
            "Each episode needs normalized_score, score/max_score, or boolean success"
        )
    if not math.isfinite(value):
        raise ValueError("Episode score must be finite")
    return value


def _gap_interval(
    id_scores: Sequence[float], ood_scores: Sequence[float], samples: int, seed: int
) -> List[float | None]:
    if not id_scores or not ood_scores:
        return [None, None]
    rng = random.Random(seed)
    gaps = []
    for _ in range(samples):
        id_mean = sum(rng.choice(id_scores) for _ in id_scores) / len(id_scores)
        ood_mean = sum(rng.choice(ood_scores) for _ in ood_scores) / len(ood_scores)
        gaps.append(id_mean - ood_mean)
    gaps.sort()
    low = gaps[max(0, int(0.025 * len(gaps)))]
    high = gaps[min(len(gaps) - 1, max(0, int(math.ceil(0.975 * len(gaps))) - 1))]
    return [low, high]


def _aggregate(
    rows: Sequence[Dict[str, Any]], bootstrap_samples: int, seed: int
) -> Dict[str, Any]:
    id_rows = [row for row in rows if row["eval_split"] == "id"]
    ood_rows = [row for row in rows if row["eval_split"] == "ood"]
    id_scores = [row["normalized_score"] for row in id_rows]
    ood_scores = [row["normalized_score"] for row in ood_rows]
    id_mean = sum(id_scores) / len(id_scores) if id_scores else None
    ood_mean = sum(ood_scores) / len(ood_scores) if ood_scores else None
    success_values = [row["success"] for row in rows if row.get("success") is not None]
    return {
        "episodes": len(rows),
        "id": summarize(id_scores, bootstrap_samples, seed),
        "ood": summarize(ood_scores, bootstrap_samples, seed),
        "ood_gap": {
            "value": id_mean - ood_mean
            if id_mean is not None and ood_mean is not None
            else None,
            "ci95": _gap_interval(id_scores, ood_scores, bootstrap_samples, seed),
            "definition": "mean_id_normalized_score - mean_ood_normalized_score",
        },
        "absolute_ood_score": ood_mean,
        "success_rate": summarize(success_values, bootstrap_samples, seed),
        "status": "complete"
        if id_scores and ood_scores
        else "incomplete_missing_id_or_ood",
    }


def score_task_success(
    records: Iterable[Dict[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 0
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, int]] = set()
    for ordinal, record in enumerate(records):
        episode_id = str(record.get("episode_id", "")).strip()
        condition = str(record.get("condition", "")).strip()
        environment = str(record.get("environment", "")).strip()
        if not episode_id or not condition or not environment:
            raise ValueError(
                f"Episode row {ordinal} requires episode_id, condition, and environment"
            )
        run_seed = int(record.get("seed", 0))
        key = (episode_id, condition, run_seed)
        if key in seen:
            raise ValueError(f"Duplicate episode/condition/seed: {key}")
        seen.add(key)
        rows.append(
            {
                "episode_id": episode_id,
                "condition": condition,
                "environment": environment,
                "eval_split": _split_name(record.get("eval_split")),
                "normalized_score": _normalized_score(record),
                "success": float(record["success"])
                if isinstance(record.get("success"), bool)
                else None,
                "seed": run_seed,
                "task_id": str(record.get("task_id", "unlabelled")),
            }
        )

    by_condition: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    by_condition_environment: Dict[Tuple[str, str], List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_condition_environment[(row["condition"], row["environment"])].append(row)

    result_conditions: Dict[str, Any] = {}
    for condition, condition_rows in sorted(by_condition.items()):
        environments = {
            environment: _aggregate(group_rows, bootstrap_samples, seed)
            for (group_condition, environment), group_rows in sorted(
                by_condition_environment.items()
            )
            if group_condition == condition
        }
        result_conditions[condition] = {
            "aggregate": _aggregate(condition_rows, bootstrap_samples, seed),
            "environments": environments,
            "seeds": sorted({row["seed"] for row in condition_rows}),
        }

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation": "id_ood_task_success",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": len(rows),
        "conditions": result_conditions,
        "interpretation_guard": (
            "A smaller OOD gap is evidence of transfer only when absolute OOD score is competitive."
        ),
    }
