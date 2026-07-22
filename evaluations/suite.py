"""Execute a matched checkpoint suite and produce its paired comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .comparison import compare_to_file
from .llada_nextobs import run_llada_nextobs
from .schema import TransitionExample


def run_evaluation_suite(
    fixtures: Sequence[TransitionExample],
    runs: Sequence[Tuple[str, int, str]],
    *,
    fixtures_path: str,
    model_dir: str,
    output_dir: str,
    primary: str,
    dataset_manifest: str | None = None,
    decode: str = "one_shot_argmax",
    decode_steps: int = 16,
    batch_size: int = 1,
    max_len: int = 8192,
    device: str = "cuda",
    dtype: str = "bfloat16",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> Dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    prediction_specs = []
    run_summaries = []
    for condition, seed, adapter in runs:
        run_dir = destination / f"{condition}-s{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = run_dir / f"nextobs-{decode}.jsonl"
        summary = run_llada_nextobs(
            fixtures,
            model_dir=model_dir,
            adapter_dir=adapter,
            output_path=str(predictions),
            batch_size=batch_size,
            max_len=max_len,
            device=device,
            dtype=dtype,
            condition=condition,
            run_seed=seed,
            decode=decode,
            decode_steps=decode_steps,
            fixture_path=fixtures_path,
            dataset_manifest=dataset_manifest,
        )
        run_summaries.append(summary)
        prediction_specs.append((condition, seed, str(predictions)))
    comparison_path = destination / f"comparison-{decode}.json"
    comparison = compare_to_file(
        fixtures,
        prediction_specs,
        str(comparison_path),
        primary=primary,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        "runs": run_summaries,
        "comparison_path": str(comparison_path),
        "comparison": comparison,
    }
