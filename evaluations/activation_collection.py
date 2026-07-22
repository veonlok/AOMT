"""Collect frozen checkpoint activations for leakage-controlled linear probes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .llada_nextobs import _tokenize_blocks
from .provenance import collect_provenance
from .schema import TransitionExample, write_json


def _pool_hidden(hidden: Any, valid_length: int, mode: str) -> Any:
    if valid_length <= 0:
        raise ValueError("valid_length must be positive")
    if mode == "final_context_token":
        return hidden[0, valid_length - 1]
    if mode == "mean_context":
        return hidden[0, :valid_length].mean(dim=0)
    raise ValueError(f"Unsupported pooling mode {mode!r}")


def collect_llada_activations(
    fixtures: Sequence[TransitionExample],
    *,
    model_dir: str,
    output_path: str,
    adapter_dir: Optional[str] = None,
    layers: Sequence[int] = (-1,),
    pooling: str = "final_context_token",
    max_len: int = 8192,
    device: str = "cuda",
    dtype: str = "bfloat16",
    fixture_path: Optional[str] = None,
    dataset_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "activation collection requires torch and transformers"
        ) from exc
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype=dtype_by_name[dtype]
    )
    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading an adapter requires peft") from exc
        model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.to(device)
    model.eval()
    collected: Dict[int, List[np.ndarray]] = {layer: [] for layer in layers}
    example_ids = []
    skipped = []
    with torch.no_grad():
        for fixture in fixtures:
            context_ids = _tokenize_blocks(tokenizer, fixture.history)
            if not context_ids or len(context_ids) > max_len:
                skipped.append(fixture.example_id)
                continue
            input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)
            width = len(context_ids)
            valid = torch.ones((1, width), dtype=torch.bool, device=device)
            attention_dtype = (
                dtype_by_name[dtype] if device.startswith("cuda") else torch.float32
            )
            attention_mask = (
                valid[:, None, None, :].expand(1, 1, width, width).to(attention_dtype)
            )
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = output.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden_states")
            for layer in layers:
                pooled = _pool_hidden(hidden_states[layer], width, pooling)
                collected[layer].append(pooled.float().cpu().numpy())
            example_ids.append(fixture.example_id)
    arrays = {
        f"layer_{layer}": np.stack(values)
        for layer, values in collected.items()
        if values
    }
    if not arrays:
        raise ValueError("No activations were collected")
    np.savez_compressed(output_path, example_ids=np.asarray(example_ids), **arrays)
    report = {
        "schema_version": "dflex-activation-collection-v1",
        "output_path": output_path,
        "examples": len(example_ids),
        "skipped_example_ids": skipped,
        "layers": list(layers),
        "pooling": pooling,
        "target_observation_visible": False,
        "checkpoint": adapter_dir or model_dir,
        "provenance": collect_provenance(
            fixture_path=fixture_path,
            checkpoint=adapter_dir or model_dir,
            dataset_manifest=dataset_manifest,
            extra={"layers": list(layers), "pooling": pooling},
        ),
    }
    write_json(f"{output_path}.manifest.json", report)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report
