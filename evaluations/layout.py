"""Discover the canonical dataset and model layout without hard-coded legacy paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import file_sha256


DATASET_VARIANTS = {
    "scienceworld": ("scienceworld-compact-v2", "scienceworld"),
    "alfworld": ("alfworld-success-v2", "alfworld"),
    "webshop": ("webshop",),
}


def discover_dataset_root(workspace: str | os.PathLike[str] = ".") -> Path:
    root = Path(workspace)
    candidates = []
    if os.environ.get("AOMT_DATA_ROOT"):
        candidates.append(Path(os.environ["AOMT_DATA_ROOT"]))
    candidates.extend([root / "cp2107-textworld-trajectories", root / "data"])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "No dataset root found; set AOMT_DATA_ROOT or add cp2107-textworld-trajectories"
    )


def resolve_dataset_variant(
    dataset_root: str | os.PathLike[str], environment: str
) -> Path:
    root = Path(dataset_root)
    if environment not in DATASET_VARIANTS:
        raise ValueError(f"Unsupported environment: {environment}")
    for name in DATASET_VARIANTS[environment]:
        candidate = root / name
        if all(
            (candidate / f"{split}.jsonl").is_file()
            for split in ("train", "validation", "test")
        ):
            return candidate
    raise FileNotFoundError(f"No complete {environment} dataset under {root}")


def _base_weight_status(base_dir: Path) -> Dict[str, Any]:
    index_path = base_dir / "model.safetensors.index.json"
    expected = []
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        expected = sorted(set(index.get("weight_map", {}).values()))
    present = [name for name in expected if (base_dir / name).is_file()]
    missing = [name for name in expected if name not in present]
    return {
        "path": str(base_dir),
        "index_present": index_path.is_file(),
        "expected_shards": len(expected),
        "present_shards": len(present),
        "missing_shards": missing,
        "weight_bytes_present": sum(
            (base_dir / name).stat().st_size for name in present
        ),
        "inference_ready": bool(expected) and not missing,
    }


def workspace_preflight(
    workspace: str | os.PathLike[str] = ".",
    *,
    dataset_root: Optional[str] = None,
    models_root: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(workspace)
    data_root = Path(dataset_root) if dataset_root else discover_dataset_root(root)
    model_root = Path(models_root) if models_root else root / "models"
    datasets = {}
    for environment in DATASET_VARIANTS:
        try:
            variant = resolve_dataset_variant(data_root, environment)
            datasets[environment] = {
                "variant": variant.name,
                "path": str(variant),
                "ready": True,
            }
        except FileNotFoundError as exc:
            datasets[environment] = {"ready": False, "error": str(exc)}
    adapter_root = model_root / "AOMT"
    adapter_items = {}
    if adapter_root.is_dir():
        for path in sorted(adapter_root.iterdir()):
            config_path = path / "adapter_config.json"
            weight_path = path / "adapter_model.safetensors"
            if (
                not path.is_dir()
                or not config_path.is_file()
                or not weight_path.is_file()
            ):
                continue
            with config_path.open("r", encoding="utf-8") as handle:
                config = json.load(handle)
            adapter_items[path.name] = {
                "path": str(path),
                "base_model": config.get("base_model_name_or_path"),
                "peft_version": config.get("peft_version"),
                "rank": config.get("r"),
                "config_sha256": file_sha256(config_path),
                "weight_bytes": weight_path.stat().st_size,
                "weight_sha256": file_sha256(weight_path),
            }
    adapters = sorted(adapter_items)
    base = _base_weight_status(model_root / "LLaDA2.0-mini")
    return {
        "schema_version": "aomt-workspace-preflight-v1",
        "workspace": str(root.resolve()),
        "dataset_root": str(data_root),
        "datasets": datasets,
        "models_root": str(model_root),
        "adapters": {
            "path": str(adapter_root),
            "count": len(adapters),
            "names": adapters,
            "items": adapter_items,
        },
        "base_model": base,
        "evaluation_ready": all(item["ready"] for item in datasets.values())
        and bool(adapters)
        and base["inference_ready"],
    }
