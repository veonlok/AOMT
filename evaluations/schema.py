"""Versioned records and safe JSON helpers used by all evaluations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


FIXTURE_SCHEMA_VERSION = "dflex-transition-v1"
PREDICTION_SCHEMA_VERSION = "dflex-prediction-v1"
RESULT_SCHEMA_VERSION = "dflex-evaluation-v1"
ROLLOUT_SCHEMA_VERSION = "dflex-rollout-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: os.PathLike[str] | str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _atomic_write(path: os.PathLike[str] | str, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: os.PathLike[str] | str, value: Any) -> None:
    _atomic_write(
        path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def write_jsonl(path: os.PathLike[str] | str, values: Iterable[Dict[str, Any]]) -> None:
    content = "".join(canonical_json(value) + "\n" for value in values)
    _atomic_write(path, content)


@dataclass(frozen=True)
class TransitionExample:
    """One causal action-to-observation evaluation example."""

    example_id: str
    trajectory_id: str
    environment: str
    source_split: str
    target_step: int
    goal: str
    history: List[Dict[str, Any]]
    action: Dict[str, Any]
    target_observation: str
    target_block_index: int
    context_hash: str
    task_id: Optional[str] = None
    eval_split: Optional[str] = None
    previous_state: Optional[Dict[str, Any]] = None
    target_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = FIXTURE_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "TransitionExample":
        if value.get("schema_version") != FIXTURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported fixture schema {value.get('schema_version')!r}; "
                f"expected {FIXTURE_SCHEMA_VERSION!r}"
            )
        fields = {
            "example_id",
            "trajectory_id",
            "environment",
            "source_split",
            "target_step",
            "goal",
            "history",
            "action",
            "target_observation",
            "target_block_index",
            "context_hash",
            "task_id",
            "eval_split",
            "previous_state",
            "target_state",
            "metadata",
            "schema_version",
        }
        return cls(**{key: value[key] for key in fields if key in value})
