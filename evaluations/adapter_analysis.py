"""Static integrity and parameter-space diagnostics for a collection of LoRA adapters."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .schema import write_json


def analyze_adapters(adapter_root: str, output_path: str) -> Dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("adapter analysis requires safetensors") from exc

    root = Path(adapter_root)
    vectors = {}
    adapters = {}
    reference_keys = None
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        weights = directory / "adapter_model.safetensors"
        config_path = directory / "adapter_config.json"
        if not weights.is_file() or not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        with safe_open(str(weights), framework="numpy") as handle:
            keys = list(handle.keys())
            arrays = [
                handle.get_tensor(key).astype(np.float32, copy=False) for key in keys
            ]
        vector = np.concatenate([array.reshape(-1) for array in arrays])
        vectors[directory.name] = vector
        if reference_keys is None:
            reference_keys = keys
        adapters[directory.name] = {
            "tensor_count": len(keys),
            "parameter_count": int(vector.size),
            "l2_norm": float(np.linalg.norm(vector)),
            "mean_absolute_weight": float(np.mean(np.abs(vector))),
            "max_absolute_weight": float(np.max(np.abs(vector))),
            "rank": config.get("r"),
            "lora_alpha": config.get("lora_alpha"),
            "base_model": config.get("base_model_name_or_path"),
            "tensor_schema_matches": keys == reference_keys,
        }

    names = sorted(vectors)
    pairwise = {}
    nearest = {}
    for left in names:
        pairwise[left] = {}
        best = None
        left_norm = float(np.linalg.norm(vectors[left]))
        for right in names:
            right_norm = float(np.linalg.norm(vectors[right]))
            cosine = float(
                np.dot(vectors[left], vectors[right]) / (left_norm * right_norm)
            )
            pairwise[left][right] = cosine
            if left != right and (best is None or cosine > best[1]):
                best = (right, cosine)
        nearest[left] = (
            {
                "adapter": best[0],
                "cosine_similarity": best[1],
            }
            if best
            else None
        )

    norms = [adapters[name]["l2_norm"] for name in names]
    report = {
        "schema_version": "aomt-adapter-analysis-v1",
        "analysis_type": "static_parameter_space_diagnostic",
        "behavioral_performance": False,
        "guard": (
            "Raw LoRA parameter similarity is an integrity/optimization diagnostic and must not "
            "be interpreted as task performance without base-model inference."
        ),
        "adapter_root": str(root),
        "adapter_count": len(names),
        "common_tensor_schema": all(
            item["tensor_schema_matches"] for item in adapters.values()
        ),
        "adapters": adapters,
        "nearest_parameter_neighbors": nearest,
        "pairwise_cosine_similarity": pairwise,
        "l2_norm_summary": {
            "min": min(norms) if norms else None,
            "max": max(norms) if norms else None,
            "mean": sum(norms) / len(norms) if norms else None,
            "finite": all(math.isfinite(value) for value in norms),
        },
    }
    write_json(output_path, report)
    return report
