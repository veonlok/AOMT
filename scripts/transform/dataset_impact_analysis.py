#!/usr/bin/env python3
"""
dataset_impact_analysis.py -- Dataset leakage and D-Random vs D-Flex impact harness.

Answers:
  1. WHY does D-Random memorize? (template leakage, not trajectory length)
  2. Is memorization driven by sample type or trajectory length?
  3. What do training curves reveal about D-Random vs D-Flex?
  4. What would stratified sampling look like vs the current random split?
  5. What is the per-family / per-component impact on memorization risk?

Outputs:
  reports/dataset_impact_report.md
  artifacts/memorization_scores.csv
  artifacts/stratified_split_sim.json

Run:
  python -m scripts.transform.dataset_impact_analysis
  (Requires artifacts/trajectory_features.csv and split JSON artifacts to exist.
   Run `python -m scripts.transform.audit_dataset` first if they are missing.)
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.utils.common import (
    ARTIFACTS_DIR,
    REPORTS_DIR,
    ROOT,
    counter_object as counter_obj,
    read_json,
    write_csv,
)
from scripts.transform.impact_report import build_report

RUNS_DIR = ROOT / "runs"


# ── I/O helper (CSV reader is specific to this script) ────────────────────────
def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ── 1. Feature loading ────────────────────────────────────────────────────────
def load_features():
    path = ARTIFACTS_DIR / "trajectory_features.csv"
    rows = read_csv(path)
    for row in rows:
        for field in ("n_steps", "n_blocks", "n_actions", "n_observations"):
            row[field] = int(row.get(field) or 0)
        for field in ("repeated_action_ratio", "repeated_observation_ratio"):
            try:
                row[field] = float(row.get(field) or 0)
            except (ValueError, TypeError):
                row[field] = 0.0
        row["has_leakage_flag"] = row.get("has_leakage_flag", "False") == "True"
    return rows


# ── 2. Training log parsing ───────────────────────────────────────────────────
STEP_RE = re.compile(r"\[step\s+(\d+)/\d+\]\s+train=([\d.]+)\s+val=([\d.]+)")

def parse_log(path: Path):
    steps, train_losses, val_losses = [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))
    return {"steps": steps, "train": train_losses, "val": val_losses}

def load_training_logs():
    candidates = {
        "drandom_const (rb_const20)": "rb_const20_640764.out",
        "drandom_curric (rb_curric)": "rb_curric_640765.out",
        "causal20": "causal20_640766.out",
    }
    logs = {}
    for label, filename in candidates.items():
        p = RUNS_DIR / filename
        if p.exists():
            logs[label] = parse_log(p)
    return logs


# ── 3. Memorization risk scoring ──────────────────────────────────────────────
def compute_memorization_scores(features):
    train = [f for f in features if f["source_split"] == "train"]
    val = [f for f in features if f["source_split"] == "validation"]

    template_train_freq = Counter(f["goal_template_key"] for f in train)
    family_train_freq = Counter(f["goal_family"] for f in train)
    prefix_train = {f["trajectory_prefix"] for f in train}

    total_train = len(train)
    scores = []
    for f in val:
        tmpl_count = template_train_freq.get(f["goal_template_key"], 0)
        fam_count = family_train_freq.get(f["goal_family"], 0)
        tmpl_rate = tmpl_count / total_train if total_train else 0.0
    
        if tmpl_count >= 10:
            mem_tier = "high"
        elif tmpl_count >= 3:
            mem_tier = "medium"
        elif tmpl_count >= 1:
            mem_tier = "low"
        else:
            mem_tier = "zero_ood"

        scores.append({
            "trajectory_id": f["trajectory_id"],
            "goal_family": f["goal_family"],
            "trajectory_length_bin": f["trajectory_length_bin"],
            "n_actions": f["n_actions"],
            "current_split_risk": f["current_split_risk"],
            "interaction_pattern": f.get("interaction_pattern", ""),
            "template_train_count": tmpl_count,
            "template_train_rate": round(tmpl_rate, 4),
            "family_train_count": fam_count,
            "memorization_tier": mem_tier,
            "same_prefix_in_train": f["trajectory_prefix"] in prefix_train,
        })

    return scores, template_train_freq, family_train_freq


# ── 4. Length-stratified memorization analysis ────────────────────────────────
def length_stratified_analysis(features, memorization_scores):
    mem_by_id = {s["trajectory_id"]: s for s in memorization_scores}
    val = [f for f in features if f["source_split"] == "validation"]

    by_length = defaultdict(list)
    for f in val:
        by_length[f["trajectory_length_bin"]].append(f)

    rows = []
    for length_bin in ["short", "medium", "long"]:
        items = by_length.get(length_bin, [])
        if not items:
            continue
        n = len(items)
        template_seen = sum(1 for f in items if f["current_split_risk"] == "template_seen")
        family_seen = sum(1 for f in items if f["current_split_risk"] == "family_seen")
        strong_ood = sum(1 for f in items if f["current_split_risk"] == "strong_ood")
        tmpl_counts = [
            mem_by_id[f["trajectory_id"]]["template_train_count"]
            for f in items
            if f["trajectory_id"] in mem_by_id
        ]
        avg_tmpl = sum(tmpl_counts) / len(tmpl_counts) if tmpl_counts else 0.0
        high_mem = sum(1 for c in tmpl_counts if c >= 10)
        rows.append({
            "length_bin": length_bin,
            "val_count": n,
            "template_seen_pct": f"{100 * template_seen / n:.1f}%",
            "family_seen": family_seen,
            "strong_ood": strong_ood,
            "avg_template_train_count": round(avg_tmpl, 1),
            "high_mem_count": high_mem,
        })
    return rows


# ── 5. Family coverage and template overlap ───────────────────────────────────
def family_coverage_analysis(features):
    train = [f for f in features if f["source_split"] == "train"]
    val = [f for f in features if f["source_split"] == "validation"]

    train_by_fam = counter_obj([f["goal_family"] for f in train])
    val_by_fam = counter_obj([f["goal_family"] for f in val])

    train_tmpl_by_fam = defaultdict(set)
    for f in train:
        train_tmpl_by_fam[f["goal_family"]].add(f["goal_template_key"])

    val_tmpl_by_fam = defaultdict(set)
    for f in val:
        val_tmpl_by_fam[f["goal_family"]].add(f["goal_template_key"])

    all_families = sorted(set(list(train_by_fam) + list(val_by_fam)))
    rows = []
    for fam in all_families:
        tr_count = train_by_fam.get(fam, 0)
        vl_count = val_by_fam.get(fam, 0)
        tr_tmpls = train_tmpl_by_fam.get(fam, set())
        vl_tmpls = val_tmpl_by_fam.get(fam, set())
        overlap = len(tr_tmpls & vl_tmpls)
        vl_total = len(vl_tmpls)
        overlap_pct = f"{100 * overlap // vl_total}%" if vl_total else "—"
        rows.append({
            "goal_family": fam,
            "train": tr_count,
            "val": vl_count,
            "train_templates": len(tr_tmpls),
            "val_templates": vl_total,
            "val_tmpl_in_train": overlap,
            "overlap_pct": overlap_pct,
        })
    return sorted(rows, key=lambda r: -r["train"])


# ── 6. Split simulation: current vs IID vs grouped OOD ───────────────────────
def split_simulation(features):
    all_by_id = {f["trajectory_id"]: f for f in features}

    def risk_distribution(val_ids, train_ids):
        train_tmpls = {all_by_id[i]["goal_template_key"] for i in train_ids if i in all_by_id}
        train_fams = {all_by_id[i]["goal_family"] for i in train_ids if i in all_by_id}
        train_hashes = {all_by_id[i]["full_trajectory_hash"] for i in train_ids if i in all_by_id}
        risks = Counter()
        for tid in val_ids:
            f = all_by_id.get(tid)
            if f is None:
                continue
            if f["full_trajectory_hash"] in train_hashes:
                risks["exact_seen"] += 1
            elif f["goal_template_key"] in train_tmpls:
                risks["template_seen"] += 1
            elif f["goal_family"] in train_fams:
                risks["family_seen"] += 1
            else:
                risks["strong_ood"] += 1
        return dict(risks)

    current_risk = Counter(
        f["current_split_risk"] for f in features if f["source_split"] == "validation"
    )
    current_val_count = sum(current_risk.values())

    iid = read_json(ARTIFACTS_DIR / "split_iid_balanced.json")
    ood = read_json(ARTIFACTS_DIR / "split_grouped_ood.json")

    iid_risk = risk_distribution(
        iid["splits"]["validation"]["trajectory_ids"],
        iid["splits"]["train"]["trajectory_ids"],
    )
    ood_risk = risk_distribution(
        ood["splits"]["validation"]["trajectory_ids"],
        ood["splits"]["train"]["trajectory_ids"],
    )

    return {
        "current": {
            "val_count": current_val_count, 
            "risk": dict(current_risk)
        },
        "iid_balanced": {
            "val_count": iid["splits"]["validation"]["count"],
            "risk": iid_risk,
        },
        "grouped_ood": {
            "val_count": ood["splits"]["validation"]["count"],
            "risk": ood_risk,
        },
    }


# ── 7. D-Flex objective analysis ──────────────────────────────────────────────
def objective_analysis():
    best_path = None
    for candidate in ["llada_dflex_full", "dflex_full"]:
        p = RUNS_DIR / candidate / "metrics.json"
        if p.exists():
            best_path = p
            break
    if best_path is None:
        return {}
    return read_json(best_path)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    logs = load_training_logs()
    memo_scores, _, _ = compute_memorization_scores(features)
    length_rows = length_stratified_analysis(features, memo_scores)
    family_rows = family_coverage_analysis(features)
    sim = split_simulation(features)
    obj_m = objective_analysis()

    write_csv(ARTIFACTS_DIR / "memorization_scores.csv", memo_scores)
    (ARTIFACTS_DIR / "stratified_split_sim.json").write_text(
        json.dumps(sim, indent=2), encoding="utf-8"
    )

    report = build_report(features, memo_scores, logs, length_rows, family_rows, sim, obj_m)
    (REPORTS_DIR / "dataset_impact_report.md").write_text(report, encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "features_loaded": len(features),
        "memorization_scores": len(memo_scores),
        "run_logs_parsed": list(logs.keys()),
        "outputs": [
            "reports/dataset_impact_report.md",
            "artifacts/memorization_scores.csv",
            "artifacts/stratified_split_sim.json",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
