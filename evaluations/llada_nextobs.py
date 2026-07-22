"""Run masked next-observation evaluation on a LLaDA checkpoint."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import (
    PREDICTION_SCHEMA_VERSION,
    TransitionExample,
    stable_hash,
    write_json,
    write_jsonl,
)
from .provenance import collect_provenance


def _tokenize_blocks(tokenizer: Any, blocks: Sequence[Dict[str, Any]]) -> List[int]:
    tokens: List[int] = []
    for block in blocks:
        tokens.extend(tokenizer.encode(str(block["text"]), add_special_tokens=False))
    return [int(token) for token in tokens]


def _decode_masked_target(
    model: Any,
    input_ids: Any,
    valid: Any,
    *,
    target_start: int,
    target_end: int,
    mask_id: int,
    pad_id: int,
    attention_dtype: Any,
    mode: str,
    steps: int,
    initial_logits: Any,
) -> Tuple[Any, int]:
    """Fill a fixed-length masked target without ever exposing reference target tokens."""

    import torch

    if mode == "one_shot_argmax":
        return initial_logits[0, target_start:target_end].argmax(dim=-1), 1
    if mode not in {"iterative_diffusion", "semi_autoregressive"}:
        raise ValueError(f"Unsupported decode mode {mode!r}")
    sequence = input_ids.clone()
    remaining = torch.zeros_like(sequence, dtype=torch.bool)
    remaining[:, target_start:target_end] = True
    target_length = target_end - target_start
    budget = max(1, min(int(steps), target_length))
    logits = initial_logits
    forward_passes = 1
    for iteration in range(budget):
        target_logits = logits[0, target_start:target_end].clone()
        target_logits[:, mask_id] = -float("inf")
        target_logits[:, pad_id] = -float("inf")
        guesses = target_logits.argmax(dim=-1)
        remaining_positions = torch.where(remaining[0, target_start:target_end])[0]
        if not len(remaining_positions):
            break
        steps_left = budget - iteration
        fill_count = max(1, math.ceil(len(remaining_positions) / steps_left))
        if mode == "semi_autoregressive":
            chosen = remaining_positions[:fill_count]
        else:
            confidence = target_logits[remaining_positions].max(dim=-1).values
            chosen = remaining_positions[
                confidence.topk(min(fill_count, len(confidence))).indices
            ]
        absolute = chosen + target_start
        sequence[0, absolute] = guesses[chosen]
        remaining[0, absolute] = False
        if remaining.any():
            width = sequence.shape[1]
            attention_mask = (
                valid[:, None, None, :].expand(1, 1, width, width).to(attention_dtype)
            )
            logits = model(
                input_ids=sequence, attention_mask=attention_mask
            ).logits.float()
            forward_passes += 1
    return sequence[0, target_start:target_end], forward_passes


def run_llada_nextobs(
    fixtures: Sequence[TransitionExample],
    *,
    model_dir: str,
    output_path: str,
    adapter_dir: Optional[str] = None,
    batch_size: int = 1,
    max_len: int = 8192,
    device: str = "cuda",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    condition: str = "unknown",
    run_seed: int = 0,
    decode: str = "one_shot_argmax",
    decode_steps: int = 16,
    fixture_path: Optional[str] = None,
    dataset_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Score every fixture with exactly the causal context stored in that fixture.

    NLL always comes from the initial fully masked target, matching `val_nextobs` in
    `train_llada_rb.py`. Decoding can be one-shot, confidence-ordered iterative diffusion, or
    left-to-right semi-autoregressive filling. Reference text supplies target length for this
    fixed-length benchmark but is never supplied as token context.
    """

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("run-llada requires torch and transformers") from exc

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_by_name:
        raise ValueError(f"Unsupported dtype {dtype!r}")
    torch_dtype = dtype_by_name[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if mask_id is None:
        raise ValueError("Tokenizer has no mask_token_id")
    if mask_id == pad_id:
        raise ValueError("mask_token_id and pad_token_id must be distinct")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype
    )
    if adapter_dir:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Loading an adapter requires peft") from exc
        model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.to(device)
    model.eval()

    prepared: List[Tuple[TransitionExample, List[int], List[int]]] = []
    skipped: List[Dict[str, Any]] = []
    for fixture in fixtures:
        context_ids = _tokenize_blocks(tokenizer, fixture.history)
        target_ids = [
            int(token)
            for token in tokenizer.encode(
                fixture.target_observation, add_special_tokens=False
            )
        ]
        if not context_ids or not target_ids:
            skipped.append({"example_id": fixture.example_id, "reason": "empty_tokens"})
            continue
        if len(context_ids) + len(target_ids) > max_len:
            skipped.append(
                {
                    "example_id": fixture.example_id,
                    "reason": "exceeds_max_len",
                    "tokens": len(context_ids) + len(target_ids),
                }
            )
            continue
        prepared.append((fixture, context_ids, target_ids))

    predictions: List[Dict[str, Any]] = []
    started = time.time()
    with torch.no_grad():
        for offset in range(0, len(prepared), batch_size):
            batch = prepared[offset : offset + batch_size]
            width = max(len(context) + len(target) for _, context, target in batch)
            input_ids = torch.full(
                (len(batch), width), pad_id, dtype=torch.long, device=device
            )
            labels = torch.full(
                (len(batch), width), -100, dtype=torch.long, device=device
            )
            valid = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
            spans: List[Tuple[int, int]] = []
            for row_index, (_, context_ids, target_ids) in enumerate(batch):
                start = len(context_ids)
                end = start + len(target_ids)
                input_ids[row_index, :start] = torch.tensor(
                    context_ids, dtype=torch.long, device=device
                )
                input_ids[row_index, start:end] = mask_id
                labels[row_index, start:end] = torch.tensor(
                    target_ids, dtype=torch.long, device=device
                )
                valid[row_index, :end] = True
                spans.append((start, end))
            attention_dtype = (
                torch_dtype if device.startswith("cuda") else torch.float32
            )
            attention_mask = (
                valid[:, None, None, :]
                .expand(len(batch), 1, width, width)
                .to(attention_dtype)
            )
            batch_started = time.time()
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float()
            token_losses = functional.cross_entropy(
                logits.transpose(1, 2), labels, reduction="none", ignore_index=-100
            )
            elapsed_ms = (time.time() - batch_started) * 1000.0
            for row_index, ((fixture, _, target_ids), (start, end)) in enumerate(
                zip(batch, spans)
            ):
                decode_started = time.time()
                prediction_tensor, forward_passes = _decode_masked_target(
                    model,
                    input_ids[row_index : row_index + 1],
                    valid[row_index : row_index + 1],
                    target_start=start,
                    target_end=end,
                    mask_id=mask_id,
                    pad_id=pad_id,
                    attention_dtype=attention_dtype,
                    mode=decode,
                    steps=decode_steps,
                    initial_logits=logits[row_index : row_index + 1],
                )
                decode_elapsed_ms = (time.time() - decode_started) * 1000.0
                prediction_ids = prediction_tensor.tolist()
                prediction_text = tokenizer.decode(
                    prediction_ids, skip_special_tokens=True
                )
                nll = float(token_losses[row_index, start:end].mean().item())
                predictions.append(
                    {
                        "schema_version": PREDICTION_SCHEMA_VERSION,
                        "example_id": fixture.example_id,
                        "context_hash": fixture.context_hash,
                        "prediction": prediction_text,
                        "nll": nll,
                        "target_tokens": len(target_ids),
                        "decode": decode,
                        "decode_steps": decode_steps,
                        "forward_passes": forward_passes,
                        "target_length_source": "reference_token_count",
                        "future_ground_truth_used": False,
                        "latency_ms_share": (
                            elapsed_ms / len(batch) + decode_elapsed_ms
                        ),
                        "condition": condition,
                        "seed": run_seed,
                    }
                )

    write_jsonl(output_path, predictions)
    summary = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "fixtures": len(fixtures),
        "fixture_ids_sha256": stable_hash([fixture.example_id for fixture in fixtures]),
        "predictions": len(predictions),
        "skipped": skipped,
        "seconds": time.time() - started,
        "output_path": output_path,
        "model_dir": model_dir,
        "adapter_dir": adapter_dir,
        "batch_size": batch_size,
        "max_len": max_len,
        "device": device,
        "dtype": dtype,
        "decode": decode,
        "decode_steps": decode_steps,
        "target_length_source": "reference_token_count",
        "future_ground_truth_used": False,
        "condition": condition,
        "seed": run_seed,
        "provenance": collect_provenance(
            fixture_path=fixture_path,
            checkpoint=adapter_dir or model_dir,
            dataset_manifest=dataset_manifest,
            extra={"condition": condition, "seed": run_seed, "decode": decode},
        ),
    }
    write_json(f"{output_path}.manifest.json", summary)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return summary
