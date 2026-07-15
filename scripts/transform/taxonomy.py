#!/usr/bin/env python3
"""Portable trajectory taxonomy + per-trajectory feature extraction.

This is the reusable core of the dataset audit: goal-family / objective-type /
interaction-pattern classifiers, goal-template normalization, and
``extract_features`` which turns a raw ``blocks`` trajectory into the flat feature
dict consumed by the audit pipeline and by ``notebooks/trajectory_taxonomy.ipynb``.

Kept free of pipeline/orchestration logic so it can be imported standalone."""

from __future__ import annotations

from scripts.utils.common import sha256_text


ROOM_WORDS = [
    "kitchen",
    "bathroom",
    "bedroom",
    "workshop",
    "greenhouse",
    "foundry",
    "living room",
    "outside",
    "hallway",
    "art studio",
]

COLOR_WORDS = ["blue", "orange", "red", "yellow", "green", "violet", "black", "white"]


def normalize_whitespace(text) -> str:
    return " ".join(str(text or "").split())


def normalize_goal_template(goal: str) -> str:
    out = normalize_whitespace(goal).lower()
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        out = out.replace(f"unknown substance {ch.lower()}", "unknown substance <var>")
    for room in sorted(ROOM_WORDS, key=len, reverse=True):
        out = out.replace(room, "<room>")
    for color in COLOR_WORDS:
        out = out.replace(color, "<color>")

    normalized = []
    token = ""
    for ch in out:
        if ch.isdigit() or ch in ".-":
            token += ch
            continue
        if token:
            normalized.append("<num>")
            token = ""
        normalized.append(ch)
    if token:
        normalized.append("<num>")
    return "".join(normalized)


def classify_goal_family(goal: str) -> str:
    g = normalize_whitespace(goal).lower()
    if "electrically conductive" in g:
        return "conductivity"
    if "measure the temperature of" in g:
        return "temperature_measurement"
    if "measure the melting point of" in g:
        return "melting_point"
    if "find the animal with the shortest life span, then the longest" in g:
        return "lifespan_dual"
    if "find the animal with the longest life span, then the shortest" in g:
        return "lifespan_dual"
    if "find the animal with the shortest life span" in g:
        return "lifespan_shortest"
    if "find the animal with the longest life span" in g:
        return "lifespan_longest"
    if "grow a " in g and " from seed" in g:
        return "plant_growth"
    if "find a(n) animal" in g:
        return "find_animal"
    if "find a(n) plant" in g:
        return "find_plant"
    if "find a(n) non-living thing" in g:
        return "find_nonliving"
    if "find a(n) living thing" in g:
        return "find_living"
    if "move it to the" in g:
        return "object_relocation"
    # --- families that were previously bucketed as "other" ---
    if "grow a " in g and ("several plants" in g or "cross-pollinat" in g):
        return "plant_reproduction"
    if "use chemistry to create" in g:
        return "chemistry_mixing"
    if "change the state of matter" in g or any(w in g for w in ("melt ", "boil ", "freeze ")):
        return "change_state_of_matter"
    if "turn on" in g and "light bulb" in g:
        return "power_device"
    if "focus on the" in g and "life stage" in g:
        return "focus_lifestage"
    return "other"


def classify_objective_type(goal_family: str) -> str:
    if goal_family.startswith("find_"):
        return "retrieval"
    if goal_family.startswith("lifespan") or goal_family == "focus_lifestage":
        return "comparison"
    if goal_family == "conductivity":
        return "classification"
    if goal_family in {"temperature_measurement", "melting_point"}:
        return "measurement"
    if goal_family in {"plant_growth", "plant_reproduction", "chemistry_mixing",
                       "change_state_of_matter", "power_device"}:
        return "multi_step_manipulation"
    if goal_family == "object_relocation":
        return "manipulation"
    return "other"


def classify_length_bin(action_count: int) -> str:
    if action_count <= 8:
        return "short"
    if action_count <= 20:
        return "medium"
    return "long"


def classify_observation_density(avg_obs_chars: float) -> str:
    if avg_obs_chars < 80:
        return "sparse"
    if avg_obs_chars < 220:
        return "medium"
    return "dense"


def repeated_action_ratio(actions) -> float:
    if len(actions) <= 1:
        return 0.0
    repeated = sum(1 for i in range(1, len(actions)) if actions[i] == actions[i - 1])
    return repeated / (len(actions) - 1)


def repeated_observation_ratio(observations) -> float:
    if len(observations) <= 1:
        return 0.0
    repeated = sum(1 for i in range(1, len(observations)) if observations[i] == observations[i - 1])
    return repeated / (len(observations) - 1)


def classify_interaction_pattern(actions, observations) -> str:
    lower_actions = [a.lower() for a in actions]
    look_around = sum(1 for a in lower_actions if a == "look around")
    teleports = sum(1 for a in lower_actions if a.startswith("teleport to "))
    waits = sum(1 for a in lower_actions if a in {"wait", "wait1"})
    numeric_actions = sum(1 for a in lower_actions if a.isdigit())
    ambiguous_obs = sum(1 for obs in observations if "Ambiguous request" in obs)

    if ambiguous_obs > 0 or numeric_actions > 0:
        return "repair_or_disambiguation"
    if waits >= 3:
        return "stateful_waiting"
    if look_around / max(len(actions), 1) >= 0.35 or teleports >= 3:
        return "search_heavy"
    if repeated_action_ratio(lower_actions) >= 0.3:
        return "repetitive"
    return "directed"


def classify_ambiguity_pattern(actions, observations) -> str:
    lower_actions = [a.lower() for a in actions]
    numeric_actions = sum(1 for a in lower_actions if a.isdigit())
    ambiguous_obs = sum(1 for obs in observations if "Ambiguous request" in obs)
    if ambiguous_obs > 0 or numeric_actions > 0:
        return "disambiguation"
    if repeated_observation_ratio(observations) >= 0.25:
        return "repeated_observation"
    return "none"


def guess_completion_status(row, actions, observations) -> str:
    if not actions:
        return "unknown"
    lower_goal = normalize_whitespace(row.get("goal", "")).lower()
    final_action = actions[-1].lower()
    if "move it to the" in lower_goal and final_action.startswith("move "):
        return "assumed_success_demo"
    if "focus on" in lower_goal and final_action.startswith("focus on "):
        return "assumed_success_demo"
    if (
        "electrically conductive" in lower_goal
        or "temperature is above" in lower_goal
        or "temperature is below" in lower_goal
        or "melting point" in lower_goal
    ) and final_action.startswith("move "):
        return "assumed_success_demo"
    if "grow a " in lower_goal and any(
        marker in obs.lower() for marker in ["grown", "reproducing stage", "banana."]
        for obs in observations
    ):
        return "assumed_success_demo"
    return "unknown"


def extract_features(rows):
    features = []
    for row in rows:
        blocks = row.get("blocks") or []
        goal_text = normalize_whitespace(row.get("goal", ""))
        goal_template_key = normalize_goal_template(goal_text)
        goal_family = classify_goal_family(goal_text)
        objective_type = classify_objective_type(goal_family)
        actions = [normalize_whitespace(block.get("text", "")) for block in blocks if block.get("type") == "Action"]
        observations = [str(block.get("text", "")) for block in blocks if block.get("type") == "Observation"]
        thinks = [str(block.get("text", "")) for block in blocks if block.get("type") == "Think"]
        obs_lengths = [len(obs) for obs in observations]
        avg_obs_chars = sum(obs_lengths) / len(obs_lengths) if obs_lengths else 0.0
        steps_present = {
            int(block.get("step"))
            for block in blocks
            if isinstance(block.get("step"), int) and int(block.get("step")) > 0
        }
        trajectory_prefix = str(row.get("trajectory_id", "")).split("_")[0] or "unknown"
        action_skeleton = " || ".join(actions)
        full_trajectory_text = goal_text + "\n" + "\n".join(
            f"{block.get('type')}|{normalize_whitespace(block.get('text', ''))}" for block in blocks
        )

        features.append(
            {
                "trajectory_id": str(row.get("trajectory_id", "")),
                "source_split": row.get("split") or row.get("_sourceSplit"),
                "env_name": row.get("env", "unknown"),
                "trajectory_prefix": trajectory_prefix,
                "goal_text": goal_text,
                "goal_family": goal_family,
                "objective_type": objective_type,
                "goal_instance_key": goal_text.lower(),
                "goal_template_key": goal_template_key,
                "n_steps": len(steps_present),
                "n_blocks": len(blocks),
                "n_actions": len(actions),
                "n_observations": len(observations),
                "n_thinks": len(thinks),
                "total_block_chars": sum(len(str(block.get("text", ""))) for block in blocks),
                "avg_observation_chars": round(avg_obs_chars, 2),
                "trajectory_length_bin": classify_length_bin(len(actions)),
                "observation_density_bin": classify_observation_density(avg_obs_chars),
                "interaction_pattern": classify_interaction_pattern(actions, observations),
                "ambiguity_pattern": classify_ambiguity_pattern(actions, observations),
                "completion_status": guess_completion_status(row, actions, observations),
                "has_leakage_flag": bool(row.get("leakage_flags")),
                "action_skeleton_hash": sha256_text(action_skeleton),
                "full_trajectory_hash": sha256_text(full_trajectory_text),
                "goal_action_cluster_key": sha256_text(goal_template_key + "\n" + action_skeleton),
                "repeated_action_ratio": round(repeated_action_ratio([a.lower() for a in actions]), 3),
                "repeated_observation_ratio": round(repeated_observation_ratio(observations), 3),
                "actions": actions,
                "observations": observations,
                "leakage_flags": row.get("leakage_flags") or [],
                "source_file": row.get("_sourceFile"),
                "source_revision": "local-scienceworld-snapshot",
                "raw_row": row,
            }
        )
    return features
