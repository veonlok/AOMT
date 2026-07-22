"""Strict adapter boundary between offline trajectories and environment simulators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class AdapterCapabilities:
    environment: str
    simulator_grounded: bool
    can_reset: bool
    can_step: bool
    can_extract_state: bool
    can_validate_transition: bool
    can_generate_counterfactuals: bool
    label_scope: str
    limitation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnvironmentAdapter(Protocol):
    capabilities: AdapterCapabilities

    def reset(self, task_id: str, seed: int) -> Any: ...

    def step(self, action: str) -> Any: ...

    def extract_state(self, state: Any) -> Dict[str, Any]: ...

    def validate_transition(
        self, before: Dict[str, Any], action: str, after: Dict[str, Any]
    ) -> Dict[str, Any]: ...


class MetadataOnlyAdapter:
    """Expose available trajectory metadata without claiming simulator grounding."""

    def __init__(self, environment: str, label_scope: str, limitation: str):
        self.capabilities = AdapterCapabilities(
            environment=environment,
            simulator_grounded=False,
            can_reset=False,
            can_step=False,
            can_extract_state=False,
            can_validate_transition=False,
            can_generate_counterfactuals=False,
            label_scope=label_scope,
            limitation=limitation,
        )

    def trajectory_metadata(self, row: Dict[str, Any]) -> Dict[str, Any]:
        labels = row.get("state_labels")
        return dict(labels) if isinstance(labels, dict) else {}

    def _unavailable(self, operation: str) -> None:
        raise RuntimeError(
            f"{self.capabilities.environment} adapter cannot {operation}: "
            f"{self.capabilities.limitation}"
        )

    def reset(self, task_id: str, seed: int) -> Any:
        del task_id, seed
        self._unavailable("reset the simulator")

    def step(self, action: str) -> Any:
        del action
        self._unavailable("step the simulator")

    def extract_state(self, state: Any) -> Dict[str, Any]:
        del state
        self._unavailable("extract step-aligned state")

    def validate_transition(
        self, before: Dict[str, Any], action: str, after: Dict[str, Any]
    ) -> Dict[str, Any]:
        del before, action, after
        self._unavailable("validate transitions")


def default_adapter(environment: str) -> MetadataOnlyAdapter:
    environment = environment.lower()
    if environment == "alfworld":
        return MetadataOnlyAdapter(
            environment,
            "trajectory",
            "downloaded labels contain task_type, PDDL parameters, and scene metadata but no live simulator",
        )
    if environment == "webshop":
        return MetadataOnlyAdapter(
            environment,
            "partial_trajectory",
            "downloaded labels are partial source metadata and no WebShop execution backend is installed",
        )
    if environment == "scienceworld":
        return MetadataOnlyAdapter(
            environment,
            "none",
            "downloaded state_labels are empty and no ScienceWorld simulator is installed",
        )
    raise ValueError(f"Unknown environment {environment!r}")
