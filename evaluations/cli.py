"""Command-line interface for preparing, running, and scoring D-Flex evaluations."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from .comparison import compare_to_file, parse_run_spec
from .adapter_analysis import analyze_adapters
from .activation_collection import collect_llada_activations
from .coverage import audit_datasets
from .fixtures import prepare_fixture
from .generators import (
    CORRUPTION_TYPES,
    generate_corruption_fixtures,
    generate_counterfactual_candidates,
)
from .llada_nextobs import run_llada_nextobs
from .layout import workspace_preflight
from .parity import check_nextobs_parity
from .provenance import collect_provenance, create_dataset_manifest
from .probes import evaluate_probes_to_file
from .rollouts import score_rollouts
from .rollout_generation import generate_rollouts_to_file, load_predictor
from .robustness import score_corruption_recovery, score_counterfactuals
from .schema import TransitionExample, read_jsonl, write_json, write_jsonl
from .scorer import score_predictions
from .suite import run_evaluation_suite
from .split_manifest import create_novelty_manifest
from .task_success import score_task_success


def _fixtures(path: str) -> List[TransitionExample]:
    return [TransitionExample.from_dict(row) for row in read_jsonl(path)]


def _prepare(args: argparse.Namespace) -> None:
    tokenizer = None
    if args.model_dir:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "token-filtered fixture preparation requires transformers"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, trust_remote_code=True
        )
    manifest = prepare_fixture(
        args.input,
        args.output,
        mode=args.mode,
        drop_leaky_think=not args.keep_leaky_think,
        max_examples=args.max_examples,
        manifest_path=args.manifest,
        split_manifest_path=args.split_manifest,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )
    summary = {key: value for key, value in manifest.items() if key != "task_counts"}
    print(json.dumps(summary, indent=2, sort_keys=True))


def _score(args: argparse.Namespace) -> None:
    report, failures = score_predictions(
        _fixtures(args.fixtures),
        read_jsonl(args.predictions),
        allow_missing=args.allow_missing,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["run"] = _run_metadata(args)
    write_json(args.output, report)
    if args.failures:
        write_jsonl(args.failures, failures)
    print(json.dumps(report, indent=2, sort_keys=True))


def _score_rollouts(args: argparse.Namespace) -> None:
    report, invalid = score_rollouts(
        read_jsonl(args.rollouts),
        strict=not args.allow_invalid,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["run"] = _run_metadata(args)
    write_json(args.output, report)
    if args.invalid:
        write_jsonl(args.invalid, invalid)
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_llada(args: argparse.Namespace) -> None:
    summary = run_llada_nextobs(
        _fixtures(args.fixtures),
        model_dir=args.model_dir,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device,
        dtype=args.dtype,
        condition=args.condition,
        run_seed=args.run_seed,
        decode=args.decode,
        decode_steps=args.decode_steps,
        fixture_path=args.fixtures,
        dataset_manifest=args.dataset_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _score_task_success(args: argparse.Namespace) -> None:
    report = score_task_success(
        read_jsonl(args.episodes),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _score_counterfactuals(args: argparse.Namespace) -> None:
    report, excluded = score_counterfactuals(
        read_jsonl(args.cases), bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    report["run"] = _run_metadata(args)
    write_json(args.output, report)
    if args.excluded:
        write_jsonl(args.excluded, excluded)
    print(json.dumps(report, indent=2, sort_keys=True))


def _score_corruptions(args: argparse.Namespace) -> None:
    report = score_corruption_recovery(
        read_jsonl(args.cases), bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    report["run"] = _run_metadata(args)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def _compare(args: argparse.Namespace) -> None:
    report = compare_to_file(
        _fixtures(args.fixtures),
        [parse_run_spec(value) for value in args.run],
        args.output,
        primary=args.primary,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _parity(args: argparse.Namespace) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "parity requires transformers for the production tokenizer"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    report = check_nextobs_parity(
        read_jsonl(args.input), tokenizer, max_len=args.max_len
    )
    report["provenance"] = collect_provenance(
        fixture_path=args.input,
        checkpoint=args.model_dir,
        dataset_manifest=args.dataset_manifest,
        extra={
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_vocab_size": len(tokenizer),
            "mask_token_id": tokenizer.mask_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_len": args.max_len,
        },
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


def _dataset_manifest(args: argparse.Namespace) -> None:
    report = create_dataset_manifest(
        args.data_root, args.output, repo_id=args.repo_id, revision=args.revision
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _run_suite(args: argparse.Namespace) -> None:
    report = run_evaluation_suite(
        _fixtures(args.fixtures),
        [parse_run_spec(value) for value in args.run],
        fixtures_path=args.fixtures,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        primary=args.primary,
        dataset_manifest=args.dataset_manifest,
        decode=args.decode,
        decode_steps=args.decode_steps,
        batch_size=args.batch_size,
        max_len=args.max_len,
        device=args.device,
        dtype=args.dtype,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _generate_rollouts(args: argparse.Namespace) -> None:
    report = generate_rollouts_to_file(
        read_jsonl(args.input),
        load_predictor(args.predictor),
        args.output,
        horizons=args.horizon or [3, 5, 10],
        stride=args.stride,
        max_rollouts=args.max_rollouts,
        include_context=not args.omit_context,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _create_splits(args: argparse.Namespace) -> None:
    report = create_novelty_manifest(
        args.train,
        args.evaluation,
        args.output,
        environment=args.environment,
        strategy=args.strategy,
        length_quantile=args.length_quantile,
    )
    summary = {key: value for key, value in report.items() if key != "assignments"}
    print(json.dumps(summary, indent=2, sort_keys=True))


def _generate_corruptions(args: argparse.Namespace) -> None:
    records = generate_corruption_fixtures(
        _fixtures(args.fixtures),
        corruption_types=args.corruption_type or CORRUPTION_TYPES,
        seed=args.seed,
    )
    write_jsonl(args.output, records)
    print(json.dumps({"records": len(records), "output": args.output}, indent=2))


def _generate_counterfactuals(args: argparse.Namespace) -> None:
    records = generate_counterfactual_candidates(
        _fixtures(args.fixtures), seed=args.seed
    )
    write_jsonl(args.output, records)
    print(json.dumps({"records": len(records), "output": args.output}, indent=2))


def _score_probes(args: argparse.Namespace) -> None:
    report = evaluate_probes_to_file(
        args.activations, args.labels, args.output, seed=args.seed
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _collect_activations(args: argparse.Namespace) -> None:
    report = collect_llada_activations(
        _fixtures(args.fixtures),
        model_dir=args.model_dir,
        adapter_dir=args.adapter_dir,
        output_path=args.output,
        layers=args.layer or [-1],
        pooling=args.pooling,
        max_len=args.max_len,
        device=args.device,
        dtype=args.dtype,
        fixture_path=args.fixtures,
        dataset_manifest=args.dataset_manifest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _audit_datasets(args: argparse.Namespace) -> None:
    inputs = []
    for value in args.input:
        environment, path = value.split(":", 1)
        inputs.append((environment, path))
    report = audit_datasets(inputs, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


def _preflight(args: argparse.Namespace) -> None:
    report = workspace_preflight(
        args.workspace, dataset_root=args.dataset_root, models_root=args.models_root
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready and not report["evaluation_ready"]:
        raise SystemExit(2)


def _analyze_adapters(args: argparse.Namespace) -> None:
    report = analyze_adapters(args.adapter_root, args.output)
    summary = {
        key: value
        for key, value in report.items()
        if key != "pairwise_cosine_similarity"
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "condition": args.condition,
        "seed": args.run_seed,
        "checkpoint": args.checkpoint,
    }


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--checkpoint")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze causal transition fixtures")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest")
    prepare.add_argument("--mode", choices=["last", "all"], default="last")
    prepare.add_argument("--keep-leaky-think", action="store_true")
    prepare.add_argument("--max-examples", type=int)
    prepare.add_argument("--split-manifest")
    prepare.add_argument("--model-dir")
    prepare.add_argument("--max-tokens", type=int, default=8192)
    prepare.set_defaults(handler=_prepare)

    score = subparsers.add_parser("score", help="score per-example prediction JSONL")
    score.add_argument("--fixtures", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--failures")
    score.add_argument("--allow-missing", action="store_true")
    score.add_argument("--bootstrap-samples", type=int, default=2000)
    score.add_argument("--seed", type=int, default=0)
    _add_run_arguments(score)
    score.set_defaults(handler=_score)

    rollouts = subparsers.add_parser(
        "score-rollouts", help="score closed-loop trace JSONL"
    )
    rollouts.add_argument("--rollouts", required=True)
    rollouts.add_argument("--output", required=True)
    rollouts.add_argument("--invalid")
    rollouts.add_argument("--allow-invalid", action="store_true")
    rollouts.add_argument("--bootstrap-samples", type=int, default=2000)
    rollouts.add_argument("--seed", type=int, default=0)
    _add_run_arguments(rollouts)
    rollouts.set_defaults(handler=_score_rollouts)

    task_success = subparsers.add_parser(
        "score-task-success", help="aggregate absolute ID/OOD scores and the OOD gap"
    )
    task_success.add_argument("--episodes", required=True)
    task_success.add_argument("--output", required=True)
    task_success.add_argument("--bootstrap-samples", type=int, default=2000)
    task_success.add_argument("--seed", type=int, default=0)
    task_success.set_defaults(handler=_score_task_success)

    counterfactuals = subparsers.add_parser(
        "score-counterfactuals", help="score simulator-validated counterfactual pairs"
    )
    counterfactuals.add_argument("--cases", required=True)
    counterfactuals.add_argument("--output", required=True)
    counterfactuals.add_argument("--excluded")
    counterfactuals.add_argument("--bootstrap-samples", type=int, default=2000)
    counterfactuals.add_argument("--seed", type=int, default=0)
    _add_run_arguments(counterfactuals)
    counterfactuals.set_defaults(handler=_score_counterfactuals)

    corruptions = subparsers.add_parser(
        "score-corruptions", help="score typed corruption-recovery traces"
    )
    corruptions.add_argument("--cases", required=True)
    corruptions.add_argument("--output", required=True)
    corruptions.add_argument("--bootstrap-samples", type=int, default=2000)
    corruptions.add_argument("--seed", type=int, default=0)
    _add_run_arguments(corruptions)
    corruptions.set_defaults(handler=_score_corruptions)

    llada = subparsers.add_parser(
        "run-llada", help="run masked next-observation scoring"
    )
    llada.add_argument("--fixtures", required=True)
    llada.add_argument("--model-dir", required=True)
    llada.add_argument("--adapter-dir")
    llada.add_argument("--output", required=True)
    llada.add_argument("--batch-size", type=int, default=1)
    llada.add_argument("--max-len", type=int, default=8192)
    llada.add_argument("--device", default="cuda")
    llada.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    llada.add_argument("--condition", required=True)
    llada.add_argument("--run-seed", type=int, required=True)
    llada.add_argument(
        "--decode",
        choices=["one_shot_argmax", "iterative_diffusion", "semi_autoregressive"],
        default="one_shot_argmax",
    )
    llada.add_argument("--decode-steps", type=int, default=16)
    llada.add_argument("--dataset-manifest")
    llada.set_defaults(handler=_run_llada)

    compare = subparsers.add_parser(
        "compare", help="compare multiple conditions and seeds with paired uncertainty"
    )
    compare.add_argument("--fixtures", required=True)
    compare.add_argument(
        "--run",
        action="append",
        required=True,
        help="CONDITION:SEED:PREDICTIONS.jsonl; repeat for every run",
    )
    compare.add_argument("--primary", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--bootstrap-samples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(handler=_compare)

    parity = subparsers.add_parser(
        "check-parity",
        help="compare training and standalone next-observation token masks",
    )
    parity.add_argument("--input", required=True)
    parity.add_argument("--model-dir", required=True)
    parity.add_argument("--output", required=True)
    parity.add_argument("--max-len", type=int, default=8192)
    parity.add_argument("--dataset-manifest")
    parity.set_defaults(handler=_parity)

    dataset_manifest = subparsers.add_parser(
        "dataset-manifest", help="hash and record an immutable dataset revision"
    )
    dataset_manifest.add_argument("--data-root", default="data")
    dataset_manifest.add_argument("--repo-id", required=True)
    dataset_manifest.add_argument("--revision", required=True)
    dataset_manifest.add_argument("--output", required=True)
    dataset_manifest.set_defaults(handler=_dataset_manifest)

    suite = subparsers.add_parser(
        "run-suite", help="evaluate all checkpoints and create a paired comparison"
    )
    suite.add_argument("--fixtures", required=True)
    suite.add_argument("--model-dir", required=True)
    suite.add_argument(
        "--run",
        action="append",
        required=True,
        help="CONDITION:SEED:ADAPTER_DIR; repeat for every checkpoint",
    )
    suite.add_argument("--primary", required=True)
    suite.add_argument("--output-dir", required=True)
    suite.add_argument("--dataset-manifest")
    suite.add_argument(
        "--decode",
        choices=["one_shot_argmax", "iterative_diffusion", "semi_autoregressive"],
        default="one_shot_argmax",
    )
    suite.add_argument("--decode-steps", type=int, default=16)
    suite.add_argument("--batch-size", type=int, default=1)
    suite.add_argument("--max-len", type=int, default=8192)
    suite.add_argument("--device", default="cuda")
    suite.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    suite.add_argument("--bootstrap-samples", type=int, default=2000)
    suite.add_argument("--seed", type=int, default=0)
    suite.set_defaults(handler=_run_suite)

    generate_rollouts = subparsers.add_parser(
        "generate-rollouts",
        help="generate closed-loop observations with recorded future actions",
    )
    generate_rollouts.add_argument("--input", required=True)
    generate_rollouts.add_argument("--predictor", required=True, help="module:object")
    generate_rollouts.add_argument("--output", required=True)
    generate_rollouts.add_argument("--horizon", action="append", type=int)
    generate_rollouts.add_argument("--stride", type=int, default=1)
    generate_rollouts.add_argument("--max-rollouts", type=int)
    generate_rollouts.add_argument("--omit-context", action="store_true")
    generate_rollouts.set_defaults(handler=_generate_rollouts)

    splits = subparsers.add_parser(
        "create-splits",
        help="label evaluation trajectories ID/OOD by training-set novelty",
    )
    splits.add_argument("--train", required=True)
    splits.add_argument("--evaluation", action="append", required=True)
    splits.add_argument("--environment", required=True)
    splits.add_argument(
        "--strategy",
        choices=["task_family", "goal_template", "scene", "length"],
        default="task_family",
    )
    splits.add_argument("--length-quantile", type=float, default=0.95)
    splits.add_argument("--output", required=True)
    splits.set_defaults(handler=_create_splits)

    generate_corruptions = subparsers.add_parser(
        "generate-corruptions", help="create deterministic typed corruption fixtures"
    )
    generate_corruptions.add_argument("--fixtures", required=True)
    generate_corruptions.add_argument(
        "--corruption-type", action="append", choices=list(CORRUPTION_TYPES)
    )
    generate_corruptions.add_argument("--seed", type=int, default=0)
    generate_corruptions.add_argument("--output", required=True)
    generate_corruptions.set_defaults(handler=_generate_corruptions)

    generate_counterfactuals = subparsers.add_parser(
        "generate-counterfactuals",
        help="create minimal edits requiring simulator validation",
    )
    generate_counterfactuals.add_argument("--fixtures", required=True)
    generate_counterfactuals.add_argument("--seed", type=int, default=0)
    generate_counterfactuals.add_argument("--output", required=True)
    generate_counterfactuals.set_defaults(handler=_generate_counterfactuals)

    probes = subparsers.add_parser(
        "score-probes", help="fit layer-wise linear probes with leakage controls"
    )
    probes.add_argument("--activations", required=True)
    probes.add_argument("--labels", required=True)
    probes.add_argument("--output", required=True)
    probes.add_argument("--seed", type=int, default=0)
    probes.set_defaults(handler=_score_probes)

    activations = subparsers.add_parser(
        "collect-activations", help="collect frozen hidden states for linear probes"
    )
    activations.add_argument("--fixtures", required=True)
    activations.add_argument("--model-dir", required=True)
    activations.add_argument("--adapter-dir")
    activations.add_argument("--output", required=True)
    activations.add_argument("--layer", action="append", type=int)
    activations.add_argument(
        "--pooling",
        choices=["final_context_token", "mean_context"],
        default="final_context_token",
    )
    activations.add_argument("--max-len", type=int, default=8192)
    activations.add_argument("--device", default="cuda")
    activations.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    activations.add_argument("--dataset-manifest")
    activations.set_defaults(handler=_collect_activations)

    readiness = subparsers.add_parser(
        "audit-readiness", help="audit which evaluations each dataset can support"
    )
    readiness.add_argument(
        "--input", action="append", required=True, help="ENVIRONMENT:PATH.jsonl"
    )
    readiness.add_argument("--output", required=True)
    readiness.set_defaults(handler=_audit_datasets)

    preflight = subparsers.add_parser(
        "preflight", help="discover datasets, adapters, and base-model shard readiness"
    )
    preflight.add_argument("--workspace", default=".")
    preflight.add_argument("--dataset-root")
    preflight.add_argument("--models-root")
    preflight.add_argument("--output")
    preflight.add_argument("--require-ready", action="store_true")
    preflight.set_defaults(handler=_preflight)

    adapter_analysis = subparsers.add_parser(
        "analyze-adapters",
        help="validate and compare LoRA weights without base inference",
    )
    adapter_analysis.add_argument("--adapter-root", default="models/AOMT")
    adapter_analysis.add_argument("--output", required=True)
    adapter_analysis.set_defaults(handler=_analyze_adapters)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
