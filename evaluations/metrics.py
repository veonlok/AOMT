"""Deterministic text, state, aggregation, and uncertainty metrics."""

from __future__ import annotations

import math
import random
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).lower()
    return " ".join(value.split())


def text_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(normalize_text(text))


def _multiset_overlap(left: Sequence[str], right: Sequence[str]) -> int:
    remaining: Dict[str, int] = {}
    for token in right:
        remaining[token] = remaining.get(token, 0) + 1
    overlap = 0
    for token in left:
        if remaining.get(token, 0) > 0:
            overlap += 1
            remaining[token] -= 1
    return overlap


def text_metrics(prediction: str, target: str) -> Dict[str, float]:
    pred_norm = normalize_text(prediction)
    target_norm = normalize_text(target)
    pred_tokens = text_tokens(prediction)
    target_tokens = text_tokens(target)
    overlap = _multiset_overlap(pred_tokens, target_tokens)
    precision = overlap / len(pred_tokens) if pred_tokens else float(not target_tokens)
    recall = overlap / len(target_tokens) if target_tokens else float(not pred_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_numbers = NUMBER_RE.findall(pred_norm)
    target_numbers = NUMBER_RE.findall(target_norm)
    return {
        "exact_match": float(pred_norm == target_norm),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "character_similarity": SequenceMatcher(None, pred_norm, target_norm).ratio(),
        "number_set_match": float(sorted(pred_numbers) == sorted(target_numbers)),
    }


def flatten_state(value: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in sorted(value, key=str):
        path = f"{prefix}.{key}" if prefix else str(key)
        item = value[key]
        if isinstance(item, Mapping):
            result.update(flatten_state(item, path))
        else:
            result[path] = item
    return result


def _canonical_state_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return tuple(sorted((_canonical_state_value(item) for item in value), key=repr))
    if isinstance(value, tuple):
        return tuple(_canonical_state_value(item) for item in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return normalize_text(str(value))


def state_metrics(
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    predicted = flatten_state(prediction)
    expected = flatten_state(target)
    prior = flatten_state(previous) if previous else {}
    if not expected:
        raise ValueError("Target state has no scorable keys")
    correct = {
        key: key in predicted
        and _canonical_state_value(predicted[key])
        == _canonical_state_value(expected[key])
        for key in expected
    }
    changed_keys = [
        key
        for key in expected
        if key not in prior
        or _canonical_state_value(prior[key]) != _canonical_state_value(expected[key])
    ]
    stable_keys = [key for key in expected if key not in changed_keys]
    return {
        "state_key_accuracy": sum(correct.values()) / len(expected),
        "state_key_coverage": sum(key in predicted for key in expected) / len(expected),
        "state_exact_match": float(
            all(correct.values()) and len(predicted) == len(expected)
        ),
        "changed_key_accuracy": (
            sum(correct[key] for key in changed_keys) / len(changed_keys)
            if changed_keys
            else float("nan")
        ),
        "stable_key_accuracy": (
            sum(correct[key] for key in stable_keys) / len(stable_keys)
            if stable_keys
            else float("nan")
        ),
    }


def finite(values: Iterable[float]) -> List[float]:
    return [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]


def bootstrap_interval(
    values: Sequence[float],
    samples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    clean = finite(values)
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return clean[0], clean[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(clean) for _ in clean) / len(clean))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(means) - 1, int(math.floor(tail * len(means)))))
    high_index = max(
        0, min(len(means) - 1, int(math.ceil((1.0 - tail) * len(means))) - 1)
    )
    return means[low_index], means[high_index]


def summarize(
    values: Sequence[float], bootstrap_samples: int = 2000, seed: int = 0
) -> Dict[str, Any]:
    clean = finite(values)
    if not clean:
        return {"count": 0, "mean": None, "stddev": None, "ci95": [None, None]}
    low, high = bootstrap_interval(clean, samples=bootstrap_samples, seed=seed)
    return {
        "count": len(clean),
        "mean": sum(clean) / len(clean),
        "stddev": statistics.stdev(clean) if len(clean) > 1 else 0.0,
        "ci95": [low, high],
    }
