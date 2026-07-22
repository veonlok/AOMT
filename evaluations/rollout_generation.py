"""Generate leakage-controlled closed-loop observation rollouts."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, List, Protocol, Sequence, Tuple

from .fixtures import _clean_blocks
from .schema import ROLLOUT_SCHEMA_VERSION, stable_hash, write_jsonl


class ObservationPredictor(Protocol):
    def predict_observation(
        self,
        *,
        goal: str,
        history: Sequence[Dict[str, Any]],
        action: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str | Dict[str, Any]: ...


def load_predictor(spec: str) -> ObservationPredictor:
    """Load `module:object`; call it without arguments when it is a factory."""

    if ":" not in spec:
        raise ValueError("Predictor must be specified as module:object")
    module_name, object_name = spec.split(":", 1)
    value = getattr(importlib.import_module(module_name), object_name)
    if isinstance(value, type):
        value = value()
    elif callable(value) and not hasattr(value, "predict_observation"):
        value = value()
    if not hasattr(value, "predict_observation"):
        raise TypeError(f"{spec} does not provide predict_observation")
    return value


def _action_observation_pairs(
    row: Dict[str, Any], *, drop_leaky_think: bool = True
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]]]:
    clean = [block for _, block in _clean_blocks(row, drop_leaky_think)]
    pairs: List[Tuple[int, int]] = []
    previous_observation = -1
    for observation_index, block in enumerate(clean):
        if block.get("type") != "Observation":
            continue
        action_index = None
        for index in range(observation_index - 1, previous_observation, -1):
            if clean[index].get("type") == "Action":
                action_index = index
                break
        if action_index is not None:
            pairs.append((action_index, observation_index))
        previous_observation = observation_index
    return clean, pairs


def _prediction_value(value: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, str):
        return {"prediction": value}
    if not isinstance(value, dict) or not isinstance(value.get("prediction"), str):
        raise TypeError(
            "predict_observation must return a string or a dict with 'prediction'"
        )
    return dict(value)


def generate_oracle_action_rollouts(
    rows: Iterable[Dict[str, Any]],
    predictor: ObservationPredictor,
    *,
    horizons: Sequence[int] = (3, 5, 10),
    stride: int = 1,
    max_rollouts: int | None = None,
    include_context: bool = True,
) -> List[Dict[str, Any]]:
    """Roll out model observations while replaying only recorded future actions.

    After the initial state, no recorded observation or Think block is placed in model context.
    Targets remain outside predictor inputs and are attached only after each prediction returns.
    """

    if stride <= 0:
        raise ValueError("stride must be positive")
    normalized_horizons = sorted({int(value) for value in horizons})
    if not normalized_horizons or normalized_horizons[0] <= 0:
        raise ValueError("horizons must contain positive integers")
    output: List[Dict[str, Any]] = []
    for row in rows:
        trajectory_id = str(row.get("trajectory_id", ""))
        clean, pairs = _action_observation_pairs(row)
        for horizon in normalized_horizons:
            for start in range(0, max(0, len(pairs) - horizon + 1), stride):
                start_action_index = pairs[start][0]
                history = [dict(block) for block in clean[:start_action_index]]
                steps = []
                for offset in range(horizon):
                    action_index, observation_index = pairs[start + offset]
                    action = dict(clean[action_index])
                    target = dict(clean[observation_index])
                    context_before_action = [dict(block) for block in history]
                    context_hash = stable_hash(
                        {
                            "goal": row.get("goal", ""),
                            "history": context_before_action,
                            "action": action,
                        }
                    )
                    response = _prediction_value(
                        predictor.predict_observation(
                            goal=str(row.get("goal", "")),
                            history=context_before_action,
                            action=action,
                            metadata={
                                "trajectory_id": trajectory_id,
                                "rollout_start": start,
                                "rollout_offset": offset,
                                "horizon": horizon,
                            },
                        )
                    )
                    step = {
                        "position": offset + 1,
                        "action": action.get("text", ""),
                        "prediction": response.pop("prediction"),
                        "target": target.get("text", ""),
                        "context_hash": context_hash,
                        "observation_context_source": (
                            "ground_truth_initial" if offset == 0 else "model"
                        ),
                        "future_ground_truth_used": False,
                        **response,
                    }
                    if include_context:
                        step["context"] = context_before_action
                    steps.append(step)
                    history.append(action)
                    history.append(
                        {
                            "step": target.get("step", -1),
                            "type": "Observation",
                            "text": step["prediction"],
                            "metadata": {"source": "model"},
                        }
                    )
                output.append(
                    {
                        "schema_version": ROLLOUT_SCHEMA_VERSION,
                        "rollout_id": f"{trajectory_id}:start-{start}:h{horizon}",
                        "trajectory_id": trajectory_id,
                        "environment": str(row.get("env", "unknown")),
                        "horizon": horizon,
                        "mode": "oracle_action_dynamics",
                        "simulator_grounded": False,
                        "steps": steps,
                    }
                )
                if max_rollouts is not None and len(output) >= max_rollouts:
                    return output
    return output


def generate_rollouts_to_file(
    rows: Iterable[Dict[str, Any]],
    predictor: ObservationPredictor,
    output_path: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    records = generate_oracle_action_rollouts(rows, predictor, **kwargs)
    write_jsonl(output_path, records)
    return {
        "schema_version": ROLLOUT_SCHEMA_VERSION,
        "rollouts": len(records),
        "output_path": output_path,
        "simulator_grounded": False,
        "mode": "oracle_action_dynamics",
    }
