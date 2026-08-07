#!/usr/bin/env python3
"""Cross-environment goal-family / objective-type / action-verb taxonomy.

This module owns the environment-specific classifiers for ScienceWorld,
ALFWorld, and WebShop and provides a small per-environment dispatch so
``notebooks/trajectory_taxonomy.ipynb`` can label all three environments
consistently:

    goal_family(env, goal, state_labels) -> str
    objective_type(env, goal_family)     -> str
    action_verb(action, env)             -> str

ALFWorld ships exact ALFRED `task_type` labels in ``state_labels`` — those are used
as the canonical family when present; the keyword classifiers are the fallback.
ScienceWorld uses a compact snapshot-specific keyword taxonomy defined here so the
env dispatch keeps all dataset-specific rules in one place.
"""

from __future__ import annotations

from notebooks.scripts.transform import taxonomy


# ----------------------------------------------------------------- ScienceWorld
def gf_scienceworld(goal: str) -> str:
    g = taxonomy.normalize_whitespace(goal).lower()
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
    if "dominant or recessive trait" in g:
        return "trait_dominance"
    if "friction" in g and "inclined planes" in g:
        return "friction_comparison"
    if "steepest angle" in g or "shallowest angle" in g:
        return "inclined_plane_angle"
    if "grow a " in g and ("several plants" in g or "cross-pollinat" in g):
        return "plant_reproduction"
    if "use chemistry to create" in g:
        return "chemistry_mixing"
    if "change the state of matter" in g or any(w in g for w in ("melt ", "boil ", "freeze ")):
        return "change_state_of_matter"
    if "turn on" in g and "create an electrical circuit" in g and (
        "electric " in g or "light bulb" in g
    ):
        return "power_device"
    if "focus on the" in g and "life stage" in g:
        return "focus_lifestage"
    return "other"


def ot_scienceworld(goal_family: str) -> str:
    if goal_family.startswith("find_"):
        return "retrieval"
    if goal_family.startswith("lifespan") or goal_family == "focus_lifestage":
        return "comparison"
    if goal_family == "conductivity":
        return "classification"
    if goal_family in {"temperature_measurement", "melting_point"}:
        return "measurement"
    if goal_family == "trait_dominance":
        return "classification"
    if goal_family in {"friction_comparison", "inclined_plane_angle"}:
        return "comparison"
    if goal_family in {"plant_growth", "plant_reproduction", "chemistry_mixing",
                       "change_state_of_matter", "power_device"}:
        return "multi_step_manipulation"
    if goal_family == "object_relocation":
        return "manipulation"
    return "other"


# --------------------------------------------------------------------- ALFWorld
def gf_alfworld(goal: str) -> str:
    g = goal.lower()
    if "clean" in g:                       
        return "clean_and_place"
    if "heat" in g or "hot" in g:          
        return "heat_and_place"
    if "cool" in g or "cold" in g:         
        return "cool_and_place"
    if "look at" in g or "examine" in g or "desklamp" in g: 
        return "look_at_in_light"
    if "two " in g or " two" in g:         
        return "pick_two_and_place"
    if "put" in g or "place" in g or "move" in g: 
        return "pick_and_place"
    return "other"


def ot_alfworld(goal_family: str) -> str:
    return "examine" if "look_at" in goal_family else "manipulation"


# ---------------------------------------------------------------------- WebShop
def gf_webshop(goal: str) -> str:
    g = goal.lower()
    if ("$" in goal) or ("price lower than" in g) or ("dollar" in g) or ("under " in g and "$" in goal):
        return "budget_constrained"
    if any(w in g for w in [" or ", "either", "compare"]):          return "comparison_shopping"
    if any(w in g for w in ["size", "color", "colour", "pack of", "count of", "scent", "flavor"]):
        return "option_selection"
    if len(g.split()) >= 14:                                        return "attribute_constrained_search"
    return "plain_retrieval"


def ot_webshop(goal_family: str) -> str:
    return "commerce"


# ---------------------------------------------------- per-environment dispatch
FAMILY_CLASSIFIERS = {
    "scienceworld": gf_scienceworld,
    "alfworld": gf_alfworld,
    "webshop": gf_webshop,
}

OBJECTIVE_CLASSIFIERS = {
    "scienceworld": ot_scienceworld,
    "alfworld": ot_alfworld,
    "webshop": ot_webshop,
}


def goal_family(env: str, goal: str, state_labels=None) -> str:
    """Canonical goal family for a trajectory. ALFWorld's exact `task_type` label
    (from ``state_labels``) wins when present; otherwise the env classifier runs."""
    sl = state_labels or {}
    if env == "alfworld" and sl.get("task_type"):
        return sl["task_type"]
    return FAMILY_CLASSIFIERS.get(env, taxonomy.classify_goal_family)(goal)


def objective_type(env: str, family: str) -> str:
    return OBJECTIVE_CLASSIFIERS.get(env, taxonomy.classify_objective_type)(family)


def action_verb(action: str, env: str) -> str:
    """Coarse verb bucket for an action string, per environment.
    WebShop actions are ``search[...]`` / ``click[...]``; the text-world envs use a
    space-delimited verb, with a few multi-word prefixes normalized."""
    a = action.strip().lower()
    if not a:
        return "(empty)"
    if a.isdigit():          # numeric reply to an "Ambiguous request" disambiguation prompt
        return "disambiguate"
    if env == "webshop":
        if a.startswith("search["): return "search"
        if a.startswith("click["):  return "click"
        return a.split("[")[0].split(" ")[0]
    for prefix in ("teleport to", "go to", "pick up", "look around", "look at",
                   "put down", "move to", "connect"):
        if a.startswith(prefix):
            return prefix
    return a.split(" ")[0]
