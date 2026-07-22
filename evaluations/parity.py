"""Prove that frozen fixtures and the training-time next-observation mask are identical."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np

from .fixtures import build_transition_examples


def _tokenize_blocks(tokenizer: Any, blocks: List[Dict[str, Any]]) -> np.ndarray:
    tokens: List[int] = []
    for block in blocks:
        tokens.extend(tokenizer.encode(block["text"], add_special_tokens=False))
    return np.asarray(tokens, dtype=np.int64)


def check_nextobs_parity(
    rows: Iterable[Dict[str, Any]], tokenizer: Any, *, max_len: int = 8192
) -> Dict[str, Any]:
    """Compare exact context and target IDs from both independent implementations."""

    import train_llada_rb as training

    checked = 0
    dropped = 0
    mismatches: List[Dict[str, Any]] = []
    for row in rows:
        fixture_items = build_transition_examples(
            [row], mode="last", drop_leaky_think=True
        )
        training_example = training.row_to_arrays(
            row, tokenizer, max_len=max_len, drop_leaky_think=True
        )
        if training_example is None or not fixture_items:
            dropped += 1
            continue
        mask = training.next_obs_mask(training_example)
        if mask is None:
            dropped += 1
            continue
        context_mask, target_mask = mask
        fixture = fixture_items[0]
        fixture_context = _tokenize_blocks(tokenizer, fixture.history)
        fixture_target = np.asarray(
            tokenizer.encode(fixture.target_observation, add_special_tokens=False),
            dtype=np.int64,
        )
        training_context = training_example.ids[context_mask]
        training_target = training_example.ids[target_mask]
        context_equal = np.array_equal(fixture_context, training_context)
        target_equal = np.array_equal(fixture_target, training_target)
        checked += 1
        if not context_equal or not target_equal:
            mismatches.append(
                {
                    "trajectory_id": training_example.traj_id,
                    "context_equal": context_equal,
                    "target_equal": target_equal,
                    "fixture_context_tokens": len(fixture_context),
                    "training_context_tokens": len(training_context),
                    "fixture_target_tokens": len(fixture_target),
                    "training_target_tokens": len(training_target),
                }
            )
    return {
        "schema_version": "dflex-nextobs-parity-v1",
        "checked": checked,
        "dropped": dropped,
        "mismatch_count": len(mismatches),
        "passed": checked > 0 and not mismatches,
        "mismatches": mismatches,
    }
