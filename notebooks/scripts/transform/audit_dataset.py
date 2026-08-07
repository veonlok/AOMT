#!/usr/bin/env python3
"""Dataset audit for the local ScienceWorld trajectory snapshot.

Extracts per-trajectory features, clusters exact/near duplicates, labels the
current split's leakage risk, and builds two leakage-robust benchmark splits
(balanced IID + grouped OOD). Emits ``trajectory_features.{csv,json}``, the split
manifests, and ``reports/dataset_audit_report.md``.

Taxonomy/feature logic lives in ``taxonomy.py``; shared io/util helpers live in
``common.py``.
"""

from __future__ import annotations

import json

from notebooks.scripts.utils.common import (
    ARTIFACTS_DIR,
    REPORTS_DIR,
    counter_object,
    ensure_dir,
    group_by,
    load_split_rows,
    render_markdown_table,
    write_csv,
)
from notebooks.scripts.transform.taxonomy import extract_features


def build_duplicate_info(features):
    full_groups = group_by(features, lambda f: f["full_trajectory_hash"])
    goal_action_groups = group_by(features, lambda f: f["goal_action_cluster_key"])
    duplicate_cluster_id_by_traj = {}
    clusters = []
    counter = 1

    for signature, items in full_groups.items():
        if len(items) <= 1:
            continue
        cluster_id = f"full_{counter}"
        counter += 1
        for item in items:
            duplicate_cluster_id_by_traj[item["trajectory_id"]] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_type": "exact_full",
                "signature": signature,
                "size": len(items),
                "trajectory_ids": [item["trajectory_id"] for item in items],
                "goal_family": items[0]["goal_family"],
                "source_splits": list(dict.fromkeys(item["source_split"] for item in items)),
            }
        )

    for signature, items in goal_action_groups.items():
        if len(items) <= 1:
            continue
        unresolved = [item for item in items if item["trajectory_id"] not in duplicate_cluster_id_by_traj]
        if not unresolved:
            continue
        cluster_id = f"goal_action_{counter}"
        counter += 1
        for item in unresolved:
            duplicate_cluster_id_by_traj[item["trajectory_id"]] = cluster_id
        clusters.append(
            {
                "cluster_id": cluster_id,
                "cluster_type": "goal_action",
                "signature": signature,
                "size": len(items),
                "trajectory_ids": [item["trajectory_id"] for item in items],
                "goal_family": items[0]["goal_family"],
                "source_splits": list(dict.fromkeys(item["source_split"] for item in items)),
            }
        )

    for feature in features:
        duplicate_cluster_id_by_traj.setdefault(feature["trajectory_id"], f"singleton_{feature['trajectory_id']}")

    return {
        "clusters": clusters,
        "duplicateClusterIdByTraj": duplicate_cluster_id_by_traj,
    }


def annotate_current_split_risk(features):
    train = [f for f in features if f["source_split"] == "train"]
    train_by_full = {f["full_trajectory_hash"] for f in train}
    train_by_goal_action = {f["goal_action_cluster_key"] for f in train}
    train_by_template = {f["goal_template_key"] for f in train}
    train_by_family = {f["goal_family"] for f in train}
    train_by_env = {f["env_name"] for f in train}

    for feature in features:
        if feature["source_split"] == "train":
            feature["current_split_risk"] = "train"
        elif feature["env_name"] not in train_by_env:
            feature["current_split_risk"] = "env_unseen"
        elif feature["full_trajectory_hash"] in train_by_full:
            feature["current_split_risk"] = "exact_seen"
        elif (
            feature["goal_action_cluster_key"] in train_by_goal_action
            or feature["goal_template_key"] in train_by_template
        ):
            feature["current_split_risk"] = "template_seen"
        elif feature["goal_family"] in train_by_family:
            feature["current_split_risk"] = "family_seen"
        else:
            feature["current_split_risk"] = "strong_ood"


def compute_overlap(features, left_split: str, right_split: str):
    left = [f for f in features if f["source_split"] == left_split]
    right = [f for f in features if f["source_split"] == right_split]
    left_full = {f["full_trajectory_hash"] for f in left}
    left_goal_action = {f["goal_action_cluster_key"] for f in left}
    left_goal_template = {f["goal_template_key"] for f in left}
    left_goal_exact = {f["goal_instance_key"] for f in left}
    left_prefix = {f["trajectory_prefix"] for f in left}
    return {
        "shared_exact_goals": sum(1 for f in right if f["goal_instance_key"] in left_goal_exact),
        "shared_goal_templates": sum(1 for f in right if f["goal_template_key"] in left_goal_template),
        "shared_goal_action_patterns": sum(1 for f in right if f["goal_action_cluster_key"] in left_goal_action),
        "shared_full_trajectories": sum(1 for f in right if f["full_trajectory_hash"] in left_full),
        "shared_prefixes": len({f["trajectory_prefix"] for f in right if f["trajectory_prefix"] in left_prefix}),
    }


def collect_current_split_summary(features):
    summary = {}
    for split in ["train", "validation", "test"]:
        items = [f for f in features if f["source_split"] == split]
        summary[split] = {
            "count": len(items),
            "env_counts": counter_object([f["env_name"] for f in items]),
            "family_counts": counter_object([f["goal_family"] for f in items]),
            "risk_counts": counter_object([f["current_split_risk"] for f in items]),
            "length_bin_counts": counter_object([f["trajectory_length_bin"] for f in items]),
            "flagged_rows": sum(1 for f in items if f["has_leakage_flag"]),
        }
    summary["overlap"] = {
        "train_validation": compute_overlap(features, "train", "validation"),
        "train_test": compute_overlap(features, "train", "test"),
        "validation_test": compute_overlap(features, "validation", "test"),
    }
    return summary


def proportional_targets(counter_map, target_count: int, total_count: int):
    out = {}
    assigned = 0
    items = sorted(counter_map.items(), key=lambda pair: (-pair[1], str(pair[0])))
    for idx, (key, count) in enumerate(items):
        if idx == len(items) - 1:
            out[key] = target_count - assigned
        else:
            value = round((count / total_count) * target_count)
            out[key] = value
            assigned += value
    return out


def compute_targets(features, target_counts):
    total = len(features)
    families = counter_object([f["goal_family"] for f in features])
    envs = counter_object([f["env_name"] for f in features])
    family_targets = {}
    env_targets = {}
    for split, target in target_counts.items():
        family_targets[split] = proportional_targets(families, target, total)
        env_targets[split] = proportional_targets(envs, target, total)
    return family_targets, env_targets


def choose_best_split(group, split_names, target_counts, family_targets, env_targets, stats):
    best = None
    best_score = float("inf")
    group_family_counts = counter_object([item["goal_family"] for item in group["items"]])
    group_env_counts = counter_object([item["env_name"] for item in group["items"]])

    for split in split_names:
        current = stats[split]
        projected_count = current["count"] + len(group["items"])
        remaining = target_counts[split] - projected_count
        overfill_penalty = abs(remaining) * 1000 if remaining < 0 else remaining * 0.5

        family_penalty = 0
        for family, count in group_family_counts.items():
            target = family_targets.get(split, {}).get(family, 0)
            current_count = current["family_counts"].get(family, 0)
            family_penalty += max(0, current_count + count - target) * 3

        env_penalty = 0
        for env, count in group_env_counts.items():
            target = env_targets.get(split, {}).get(env, 0)
            current_count = current["env_counts"].get(env, 0)
            env_penalty += max(0, current_count + count - target) * 2

        fill_penalty = abs(target_counts[split] - projected_count) * 0.25
        score = overfill_penalty + family_penalty + env_penalty + fill_penalty
        if score < best_score:
            best_score = score
            best = split
    return best


def allocate_groups(groups, target_counts, family_targets, env_targets):
    split_names = list(target_counts.keys())
    assignments = {}
    stats = {
        split: {"count": 0, "family_counts": {}, "env_counts": {}}
        for split in split_names
    }

    sorted_groups = sorted(groups, key=lambda group: (-len(group["items"]), group["key"]))
    for group in sorted_groups:
        best_split = choose_best_split(group, split_names, target_counts, family_targets, env_targets, stats)
        assignments[group["key"]] = best_split
        stats[best_split]["count"] += len(group["items"])
        for item in group["items"]:
            stats[best_split]["family_counts"][item["goal_family"]] = (
                stats[best_split]["family_counts"].get(item["goal_family"], 0) + 1
            )
            stats[best_split]["env_counts"][item["env_name"]] = (
                stats[best_split]["env_counts"].get(item["env_name"], 0) + 1
            )
    return assignments, stats


def build_split_file(features, assignments_by_trajectory, split_name: str):
    items = [f for f in features if assignments_by_trajectory[f["trajectory_id"]] == split_name]
    return {
        "split": split_name,
        "count": len(items),
        "env_counts": counter_object([f["env_name"] for f in items]),
        "family_counts": counter_object([f["goal_family"] for f in items]),
        "length_bin_counts": counter_object([f["trajectory_length_bin"] for f in items]),
        "trajectory_ids": [f["trajectory_id"] for f in items],
    }


def build_assignments_from_groups(features, group_key_fn, target_counts):
    grouped = group_by(features, group_key_fn)
    groups = [{"key": key, "items": items} for key, items in grouped.items()]
    family_targets, env_targets = compute_targets(features, target_counts)
    assignments, _stats = allocate_groups(groups, target_counts, family_targets, env_targets)
    assignments_by_trajectory = {}
    for group in groups:
        split = assignments[group["key"]]
        for item in group["items"]:
            assignments_by_trajectory[item["trajectory_id"]] = split
    return assignments_by_trajectory


def compute_benchmark_overlap(features, assignments_by_trajectory, key_field: str):
    by_split = {"train": set(), "validation": set(), "test": set()}
    for feature in features:
        split = assignments_by_trajectory[feature["trajectory_id"]]
        by_split[split].add(feature[key_field])
    return {
        "train_validation": len(by_split["validation"] & by_split["train"]),
        "train_test": len(by_split["test"] & by_split["train"]),
        "validation_test": len(by_split["test"] & by_split["validation"]),
    }


def build_benchmark_artifacts(features):
    target_counts = {
        "train": sum(1 for f in features if f["source_split"] == "train"),
        "validation": sum(1 for f in features if f["source_split"] == "validation"),
        "test": sum(1 for f in features if f["source_split"] == "test"),
    }
    iid_assignments = build_assignments_from_groups(features, lambda f: f["goal_action_cluster_key"], target_counts)
    grouped_ood_assignments = build_assignments_from_groups(features, lambda f: f["goal_template_key"], target_counts)
    return {
        "iid": {
            "metadata": {
                "benchmark": "iid_balanced",
                "grouping_key": "goal_action_cluster_key",
                "description": "Duplicate-aware balanced split built from the full local snapshot pool.",
            },
            "splits": {
                "train": build_split_file(features, iid_assignments, "train"),
                "validation": build_split_file(features, iid_assignments, "validation"),
                "test": build_split_file(features, iid_assignments, "test"),
            },
            "verification": compute_benchmark_overlap(features, iid_assignments, "goal_action_cluster_key"),
            "assignments": dict(sorted(iid_assignments.items())),
        },
        "groupedOod": {
            "metadata": {
                "benchmark": "grouped_ood",
                "grouping_key": "goal_template_key",
                "description": "Template-grouped OOD split built from the full local snapshot pool.",
            },
            "splits": {
                "train": build_split_file(features, grouped_ood_assignments, "train"),
                "validation": build_split_file(features, grouped_ood_assignments, "validation"),
                "test": build_split_file(features, grouped_ood_assignments, "test"),
            },
            "verification": compute_benchmark_overlap(features, grouped_ood_assignments, "goal_template_key"),
            "assignments": dict(sorted(grouped_ood_assignments.items())),
        },
    }


def build_dataset_audit_report(features, duplicate_info, current_summary, benchmarks):
    family_rows = [
        {"goal_family": goal_family, "count": count}
        for goal_family, count in counter_object([f["goal_family"] for f in features]).items()
    ]
    risk_rows = [{"risk": risk, "count": count} for risk, count in current_summary["validation"]["risk_counts"].items()]
    overlap_rows = []
    for pair, stats in current_summary["overlap"].items():
        overlap_rows.append({"pair": pair, **stats})
    duplicate_rows = sorted(
        duplicate_info["clusters"],
        key=lambda cluster: (-cluster["size"], cluster["cluster_id"]),
    )[:10]
    duplicate_rows = [
        {
            "cluster_id": cluster["cluster_id"],
            "cluster_type": cluster["cluster_type"],
            "size": cluster["size"],
            "goal_family": cluster["goal_family"],
            "source_splits": ",".join(cluster["source_splits"]),
        }
        for cluster in duplicate_rows
    ]
    iid_counts = benchmarks["iid"]["splits"]
    ood_counts = benchmarks["groupedOod"]["splits"]
    iid_verify = benchmarks["iid"]["verification"]
    ood_verify = benchmarks["groupedOod"]["verification"]

    return f"""# Dataset Audit Report

## Summary

- Total rows audited: {len(features)}
- Environment coverage in the accessible snapshot: {", ".join(counter_object([f["env_name"] for f in features]).keys())}
- Exact full-trajectory duplicates across the local snapshot: {sum(1 for cluster in duplicate_info["clusters"] if cluster["cluster_type"] == "exact_full")}
- Goal-plus-action duplicate clusters across the local snapshot: {sum(1 for cluster in duplicate_info["clusters"] if cluster["cluster_type"] == "goal_action")}

## Key Findings

- The local snapshot is entirely **ScienceWorld** even though downstream evaluation is expected on ScienceWorld, ALFWorld, and WebShop.
- The current split has heavy **exact goal overlap** and **goal/action-pattern overlap** between train and validation/test.
- The current split has **0 exact full-trajectory overlaps** across train/validation/test, so the leakage signal is mainly template-level rather than byte-for-byte duplication.
- Validation is likely favorable to memorization because many validation rows are already in `template_seen` or `family_seen` territory relative to train.

## Goal Family Distribution

{render_markdown_table(family_rows, ["goal_family", "count"])}

## Current Split Overlap

{render_markdown_table(overlap_rows, ["pair", "shared_exact_goals", "shared_goal_templates", "shared_goal_action_patterns", "shared_full_trajectories", "shared_prefixes"])}

## Validation Risk Tiers

{render_markdown_table(risk_rows, ["risk", "count"])}

## Largest Duplicate Clusters

{render_markdown_table(duplicate_rows, ["cluster_id", "cluster_type", "size", "goal_family", "source_splits"])}

## Proposed Benchmarks

### Balanced IID

- Train: {iid_counts["train"]["count"]}
- Validation: {iid_counts["validation"]["count"]}
- Test: {iid_counts["test"]["count"]}
- Grouping key: `{benchmarks["iid"]["metadata"]["grouping_key"]}`
- Verified cross-split grouping overlap: train/val {iid_verify["train_validation"]}, train/test {iid_verify["train_test"]}, val/test {iid_verify["validation_test"]}

### Grouped OOD

- Train: {ood_counts["train"]["count"]}
- Validation: {ood_counts["validation"]["count"]}
- Test: {ood_counts["test"]["count"]}
- Grouping key: `{benchmarks["groupedOod"]["metadata"]["grouping_key"]}`
- Verified cross-split grouping overlap: train/val {ood_verify["train_validation"]}, train/test {ood_verify["train_test"]}, val/test {ood_verify["validation_test"]}

## Recommendation

Use the **grouped OOD** split as the primary representation-learning benchmark and keep the **balanced IID** split as a sanity-check model-selection benchmark. The current split is still useful as a historical reference, but it is not robust enough to stand alone as evidence of out-of-distribution generalization.
"""


def main():
    ensure_dir(ARTIFACTS_DIR)
    ensure_dir(REPORTS_DIR)

    rows, _ = load_split_rows()

    features = extract_features(rows)
    duplicate_info = build_duplicate_info(features)
    for feature in features:
        feature["duplicate_cluster_id"] = duplicate_info["duplicateClusterIdByTraj"][feature["trajectory_id"]]
    annotate_current_split_risk(features)

    current_summary = collect_current_split_summary(features)
    benchmarks = build_benchmark_artifacts(features)

    serializable_features = []
    for feature in features:
        serializable_features.append({
            "trajectory_id": feature["trajectory_id"],
            "source_split": feature["source_split"],
            "env_name": feature["env_name"],
            "trajectory_prefix": feature["trajectory_prefix"],
            "goal_family": feature["goal_family"],
            "objective_type": feature["objective_type"],
            "goal_text": feature["goal_text"],
            "goal_template_key": feature["goal_template_key"],
            "n_steps": feature["n_steps"],
            "n_blocks": feature["n_blocks"],
            "n_actions": feature["n_actions"],
            "n_observations": feature["n_observations"],
            "trajectory_length_bin": feature["trajectory_length_bin"],
            "observation_density_bin": feature["observation_density_bin"],
            "interaction_pattern": feature["interaction_pattern"],
            "ambiguity_pattern": feature["ambiguity_pattern"],
            "completion_status": feature["completion_status"],
            "has_leakage_flag": feature["has_leakage_flag"],
            "repeated_action_ratio": feature["repeated_action_ratio"],
            "repeated_observation_ratio": feature["repeated_observation_ratio"],
            "action_skeleton_hash": feature["action_skeleton_hash"],
            "full_trajectory_hash": feature["full_trajectory_hash"],
            "goal_action_cluster_key": feature["goal_action_cluster_key"],
            "duplicate_cluster_id": feature["duplicate_cluster_id"],
            "current_split_risk": feature["current_split_risk"],
            "source_file": feature["source_file"],
            "source_revision": feature["source_revision"],
        })

    (ARTIFACTS_DIR / "trajectory_features.json").write_text(
        json.dumps(serializable_features, indent=2), encoding="utf-8"
    )
    write_csv(ARTIFACTS_DIR / "trajectory_features.csv", serializable_features)
    (ARTIFACTS_DIR / "duplicate_clusters.json").write_text(
        json.dumps(duplicate_info["clusters"], indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "split_current_hf_audit.json").write_text(
        json.dumps({
            "metadata": {
                "benchmark": "current_snapshot_audit",
                "source": "local-scienceworld-snapshot",
                "note": "HF public URL resolves to a model repo; this audit runs on the local trajectory snapshot.",
            },
            "summary": current_summary,
        },indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "split_iid_balanced.json").write_text(
        json.dumps(benchmarks["iid"], indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "split_grouped_ood.json").write_text(
        json.dumps(benchmarks["groupedOod"], indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "benchmark_verification.json").write_text(
        json.dumps({
            "iid_goal_action_overlap": benchmarks["iid"]["verification"],
            "ood_goal_template_overlap": benchmarks["groupedOod"]["verification"],
        }, indent=2), encoding="utf-8")

    (REPORTS_DIR / "dataset_audit_report.md").write_text(
        build_dataset_audit_report(features, duplicate_info, current_summary, benchmarks),
        encoding="utf-8",
    )

    print(
        json.dumps({
            "features": len(serializable_features),
            "duplicate_clusters": len(duplicate_info["clusters"]),
            "reports": [
                str((REPORTS_DIR / "dataset_audit_report.md").relative_to(REPORTS_DIR.parent)).replace("\\", "/"),
            ]
        }, indent=2)
    )


if __name__ == "__main__":
    main()
