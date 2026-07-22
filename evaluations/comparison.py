"""Multi-condition, multi-seed comparison with paired hierarchical uncertainty."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .metrics import text_metrics
from .schema import (
    RESULT_SCHEMA_VERSION,
    TransitionExample,
    read_jsonl,
    stable_hash,
    write_json,
)


def parse_run_spec(value: str) -> Tuple[str, int, str]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        raise ValueError("Run specs must be CONDITION:SEED:PREDICTIONS.jsonl")
    return parts[0], int(parts[1]), parts[2]


def _prediction_rows(
    fixtures: Sequence[TransitionExample], path: str, condition: str, seed: int
) -> List[Dict[str, Any]]:
    by_id = {str(row.get("example_id")): row for row in read_jsonl(path)}
    if len(by_id) != len(fixtures):
        raise ValueError(
            f"{condition} seed {seed}: expected {len(fixtures)} predictions, got {len(by_id)}"
        )
    rows = []
    for fixture in fixtures:
        prediction = by_id.get(fixture.example_id)
        if prediction is None:
            raise ValueError(f"{condition} seed {seed}: missing {fixture.example_id}")
        if prediction.get("context_hash") not in (None, fixture.context_hash):
            raise ValueError(
                f"{condition} seed {seed}: context mismatch for {fixture.example_id}"
            )
        text = prediction.get("prediction")
        if not isinstance(text, str):
            raise ValueError(
                f"{condition} seed {seed}: missing text for {fixture.example_id}"
            )
        metrics = text_metrics(text, fixture.target_observation)
        rows.append(
            {
                "condition": condition,
                "seed": seed,
                "example_id": fixture.example_id,
                "cluster": fixture.task_id or fixture.trajectory_id,
                "nll": float(prediction["nll"])
                if isinstance(prediction.get("nll"), (int, float))
                else None,
                "target_tokens": int(prediction.get("target_tokens", 0)),
                "token_f1": metrics["token_f1"],
                "exact_match": metrics["exact_match"],
            }
        )
    return rows


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else float("nan")


def _condition_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    seeds: Dict[str, Any] = {}
    for seed, items in sorted(by_seed.items()):
        nll_items = [item for item in items if item["nll"] is not None]
        token_total = sum(item["target_tokens"] for item in nll_items)
        seeds[str(seed)] = {
            "examples": len(items),
            "macro_nll": _mean(item["nll"] for item in nll_items),
            "micro_token_nll": (
                sum(item["nll"] * item["target_tokens"] for item in nll_items)
                / token_total
                if token_total
                else None
            ),
            "token_f1": _mean(item["token_f1"] for item in items),
            "exact_match": _mean(item["exact_match"] for item in items),
        }
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "across_seed": {
            metric: _mean(
                value[metric] for value in seeds.values() if value[metric] is not None
            )
            for metric in ("macro_nll", "micro_token_nll", "token_f1", "exact_match")
        },
    }


def _hierarchical_interval(
    differences: Dict[int, Dict[str, List[float]]], *, samples: int, seed: int
) -> Dict[str, Any]:
    common_seeds = sorted(differences)
    if not common_seeds:
        return {"mean": None, "ci95": [None, None], "win_rate": None, "pairs": 0}
    observed = [
        value
        for clusters in differences.values()
        for values in clusters.values()
        for value in values
    ]
    rng = random.Random(seed)
    boot = []
    for _ in range(samples):
        sampled_values: List[float] = []
        for sampled_seed in (rng.choice(common_seeds) for _ in common_seeds):
            clusters = differences[sampled_seed]
            names = list(clusters)
            for sampled_cluster in (rng.choice(names) for _ in names):
                sampled_values.extend(clusters[sampled_cluster])
        boot.append(_mean(sampled_values))
    boot.sort()
    low = boot[max(0, int(math.floor(0.025 * len(boot))))]
    high = boot[min(len(boot) - 1, max(0, int(math.ceil(0.975 * len(boot))) - 1))]
    return {
        "mean": _mean(observed),
        "ci95": [low, high],
        "win_rate": sum(value > 0 for value in observed) / len(observed),
        "pairs": len(observed),
        "seed_count": len(common_seeds),
        "cluster_count": sum(len(clusters) for clusters in differences.values()),
    }


def _paired_comparison(
    primary_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    primary = {(row["seed"], row["example_id"]): row for row in primary_rows}
    baseline = {(row["seed"], row["example_id"]): row for row in baseline_rows}
    common = sorted(set(primary) & set(baseline))
    result = {}
    for metric, direction in (("nll", -1.0), ("token_f1", 1.0), ("exact_match", 1.0)):
        differences: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key in common:
            left, right = primary[key], baseline[key]
            if left[metric] is None or right[metric] is None:
                continue
            improvement = direction * (left[metric] - right[metric])
            differences[left["seed"]][left["cluster"]].append(improvement)
        result[metric] = {
            "orientation": "positive_means_primary_better",
            **_hierarchical_interval(differences, samples=bootstrap_samples, seed=seed),
        }
    return result


def compare_runs(
    fixtures: Sequence[TransitionExample],
    run_specs: Sequence[Tuple[str, int, str]],
    *,
    primary: str,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    all_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    sources = []
    seen_runs = set()
    for condition, run_seed, path in run_specs:
        key = (condition, run_seed)
        if key in seen_runs:
            raise ValueError(f"Duplicate condition/seed run: {key}")
        seen_runs.add(key)
        rows = _prediction_rows(fixtures, path, condition, run_seed)
        all_rows[condition].extend(rows)
        sources.append(
            {
                "condition": condition,
                "seed": run_seed,
                "path": path,
                "sha256": __import__("hashlib")
                .sha256(Path(path).read_bytes())
                .hexdigest(),
            }
        )
    if primary not in all_rows:
        raise ValueError(f"Primary condition {primary!r} is not present")
    comparisons = {
        baseline: _paired_comparison(
            all_rows[primary], rows, bootstrap_samples=bootstrap_samples, seed=seed
        )
        for baseline, rows in sorted(all_rows.items())
        if baseline != primary
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation": "multi_run_paired_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary": primary,
        "fixture_count": len(fixtures),
        "fixture_ids_sha256": stable_hash([fixture.example_id for fixture in fixtures]),
        "sources": sources,
        "conditions": {
            condition: _condition_summary(rows)
            for condition, rows in sorted(all_rows.items())
        },
        "paired_improvements": comparisons,
        "bootstrap": {
            "method": "hierarchical_seed_then_task_cluster",
            "samples": bootstrap_samples,
            "seed": seed,
        },
    }


def compare_to_file(
    fixtures: Sequence[TransitionExample],
    run_specs: Sequence[Tuple[str, int, str]],
    output_path: str,
    *,
    primary: str,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    report = compare_runs(
        fixtures,
        run_specs,
        primary=primary,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    write_json(output_path, report)
    return report
