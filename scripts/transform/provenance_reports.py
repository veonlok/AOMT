#!/usr/bin/env python3
"""Provenance + model-slice reporting (split out of the core dataset audit).

These reports document where the published artifacts came from and stub out the
per-slice model comparison. They depend on a cached ``artifacts/hf_source_probe.json``
and on local ``runs/*/metrics.json``, and they predate the gated multi-environment
dataset now available on Hugging Face — so their "HF exposes only a model repo"
framing is historical. Kept as a separate entry point rather than deleted.

Run standalone:  python -m scripts.transform.provenance_reports
Emits: reports/source_of_truth_report.md, reports/model_slice_report.md,
       artifacts/snapshot_diff.json, artifacts/run_metrics_summary.json
"""

from __future__ import annotations

import json

from scripts.utils.common import (
    ARTIFACTS_DIR,
    REPORTS_DIR,
    ROOT,
    ensure_dir,
    load_split_rows,
    read_json,
    render_markdown_table,
)


def infer_objective_set(run_name: str) -> str:
    if "rb" in run_name or "random" in run_name:
        return "random_block"
    if "dar" in run_name or "causal" in run_name:
        return "d_ar"
    if "dflex" in run_name:
        return "d_flex"
    return "unknown"


def load_run_metrics():
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        return []
    rows = []
    for entry in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        metrics_path = entry / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = read_json(metrics_path)
        rows.append(
            {
                "run_name": entry.name,
                "objective_set": metrics.get("objective_set") or infer_objective_set(entry.name),
                "final_val_loss": metrics.get("final_val_loss"),
                "best_val_nextobs_loss": metrics.get("best_val_nextobs_loss"),
                "final_val_nextobs_loss": metrics.get("final_val_nextobs_loss"),
                "schedule": metrics.get("schedule"),
                "minutes": metrics.get("minutes"),
            }
        )
    return rows


def build_source_of_truth_report(hf_probe, local_summary):
    sibling_folders = sorted(
        {
            sibling["rfilename"].split("/")[0]
            for sibling in hf_probe.get("model_api", {}).get("siblings", [])
            if sibling["rfilename"].split("/")[0] not in {".gitattributes", "README.md", ""}
        }
    )
    return f"""# Source of Truth Report

## Summary

- Requested HF URL: `{hf_probe['requested_url']}`
- Public repo type resolved via API: `{hf_probe.get('model_api', {}).get('repoType') or hf_probe.get('public_repo_type', 'unknown')}`
- Public repo id: `{hf_probe.get('model_api', {}).get('id', 'unknown')}`
- Local trajectory snapshot rows: {local_summary['total_rows']}
- Local environments present: {", ".join(local_summary['env_counts'].keys()) or "none"}

## Hugging Face Findings

- The provided public URL resolves to a **model repo**, not a public dataset repo.
- The cached API probe shows the repo exposes adapter folders such as: {", ".join(sibling_folders)}.
- The public `README.md` in the model repo is empty in the fetched HTML page.
- A probe to `{hf_probe['dataset_api_probe']['url']}` returned HTTP {hf_probe['dataset_api_probe']['status']}, so there is no public dataset payload available to this audit from that endpoint.

## Implication For This Audit

- Hugging Face is still used as the **provenance source** for published artifacts and adapter inventory.
- The executable trajectory audit runs on the local JSONL snapshot because that is the accessible corpus in this checkout.
- The harness is written to preserve `env_name` and multi-environment fields so it can ingest ALFWorld/WebShop data later without redesign.

## Local Snapshot Inventory

- Train rows: {local_summary['split_counts']['train']}
- Validation rows: {local_summary['split_counts']['validation']}
- Test rows: {local_summary['split_counts']['test']}
- SHA256 train: `{local_summary['file_hashes']['train']}`
- SHA256 validation: `{local_summary['file_hashes']['validation']}`
- SHA256 test: `{local_summary['file_hashes']['test']}`
"""


def build_model_slice_report(run_metrics, hf_probe):
    run_rows = [
        {
            "run_name": run["run_name"],
            "objective_set": run["objective_set"],
            "final_val_loss": run["final_val_loss"] if run["final_val_loss"] is not None else "",
            "best_val_nextobs_loss": run["best_val_nextobs_loss"] if run["best_val_nextobs_loss"] is not None else "",
            "final_val_nextobs_loss": run["final_val_nextobs_loss"] if run["final_val_nextobs_loss"] is not None else "",
            "minutes": run["minutes"] if run["minutes"] is not None else "",
        }
        for run in run_metrics
    ]
    available_adapters = sorted(
        {
            sibling["rfilename"].split("/")[0]
            for sibling in hf_probe.get("model_api", {}).get("siblings", [])
            if sibling["rfilename"].split("/")[0] not in {"", ".gitattributes", "README.md"}
        }
    )
    runs_md = (
        render_markdown_table(
            run_rows,
            [
                "run_name",
                "objective_set",
                "final_val_loss",
                "best_val_nextobs_loss",
                "final_val_nextobs_loss",
                "minutes",
            ],
        )
        if run_rows
        else "No local run metrics were found."
    )
    adapters_md = "\n".join(f"- `{name}`" for name in available_adapters)

    return f"""# Model Slice Report

## Status

This execution produced the dataset-side audit harness and split redesign artifacts. A full per-slice D-Random vs D-Flex error analysis was **not executed yet** because the current checkout does not include:

- a local base LLaDA runtime with Python/transformers available on PATH
- per-trajectory evaluation outputs for the published adapters
- local ALFWorld/WebShop evaluation traces to join against the dataset slices

## Available Local Run Metrics

{runs_md}

## Published Hugging Face Adapter Inventory

{adapters_md}

## Next Step

Use the generated `trajectory_features.csv` and split manifests as the join key layer for future per-trajectory evaluation once the adapter evaluation runtime is available. The intended slice comparisons are:

- seen-template vs unseen-template
- seen-family vs unseen-family
- short vs medium vs long trajectories
- ambiguity-heavy vs clean trajectories
- ScienceWorld vs ALFWorld vs WebShop downstream transfer
"""


def main():
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(REPORTS_DIR)

    hf_probe_path = ARTIFACTS_DIR / "hf_source_probe.json"
    if not hf_probe_path.exists():
        raise FileNotFoundError(f"Missing cached HF probe at {hf_probe_path}")
    hf_probe = read_json(hf_probe_path)

    _rows, local_summary = load_split_rows()
    run_metrics = load_run_metrics()

    (ARTIFACTS_DIR / "snapshot_diff.json").write_text(
        json.dumps(
            {
                "status": "hf_dataset_unavailable",
                "requested_hf_url": hf_probe["requested_url"],
                "public_repo_type": hf_probe.get("model_api", {}).get("repoType")
                or hf_probe.get("public_repo_type", "unknown"),
                "dataset_api_probe": hf_probe["dataset_api_probe"],
                "local_snapshot": {
                    "total_rows": local_summary["total_rows"],
                    "split_counts": local_summary["split_counts"],
                    "env_counts": local_summary["env_counts"],
                    "file_hashes": local_summary["file_hashes"],
                },
                "conclusion": "The public Hugging Face endpoint available during this execution exposes model adapters, not public trajectory JSONL files, so only provenance reconciliation was possible.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "run_metrics_summary.json").write_text(
        json.dumps(run_metrics, indent=2), encoding="utf-8"
    )

    (REPORTS_DIR / "source_of_truth_report.md").write_text(
        build_source_of_truth_report(hf_probe, local_summary), encoding="utf-8"
    )
    (REPORTS_DIR / "model_slice_report.md").write_text(
        build_model_slice_report(run_metrics, hf_probe), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "reports": [
                    str((REPORTS_DIR / "source_of_truth_report.md").relative_to(ROOT)).replace("\\", "/"),
                    str((REPORTS_DIR / "model_slice_report.md").relative_to(ROOT)).replace("\\", "/"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
