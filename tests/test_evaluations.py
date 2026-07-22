from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluations.comparison import compare_runs, parse_run_spec
from evaluations.activation_collection import _pool_hidden
from evaluations.coverage import audit_rows
from evaluations.environment_adapters import default_adapter
from evaluations.fixtures import build_transition_examples
from evaluations.generators import (
    generate_corruption_fixtures,
    generate_counterfactual_candidates,
)
from evaluations.llada_nextobs import _decode_masked_target
from evaluations.layout import workspace_preflight
from evaluations.metrics import state_metrics, text_metrics
from evaluations.rollouts import score_rollouts, validate_rollout
from evaluations.rollout_generation import generate_oracle_action_rollouts
from evaluations.robustness import score_corruption_recovery, score_counterfactuals
from evaluations.parity import check_nextobs_parity
from evaluations.probes import evaluate_probes
from evaluations.schema import write_jsonl
from evaluations.split_manifest import build_novelty_manifest, goal_template
from evaluations.scorer import score_predictions
from evaluations.task_success import score_task_success


def trajectory():
    return {
        "trajectory_id": "task-1-run-1",
        "env": "scienceworld",
        "split": "validation",
        "goal": "Put the apple in the box.",
        "metadata": {"task_id": "put-object", "eval_split": "ood"},
        "leakage_flags": [{"step": 3, "type": "think_mentions_future"}],
        "state_labels": {
            "by_step": {
                "2": {"room": "kitchen", "inventory": []},
                "4": {"room": "kitchen", "inventory": ["apple"]},
            }
        },
        "blocks": [
            {
                "step": 0,
                "type": "Goal",
                "text": "Put the apple in the box.",
                "metadata": {},
            },
            {"step": 1, "type": "Think", "text": "I should look.", "metadata": {}},
            {"step": 1, "type": "Action", "text": "look around", "metadata": {}},
            {
                "step": 2,
                "type": "Observation",
                "text": "An apple is on the table.",
                "metadata": {},
            },
            {
                "step": 3,
                "type": "Think",
                "text": "Future leak: I picked it up.",
                "metadata": {},
            },
            {"step": 3, "type": "Action", "text": "pick up apple", "metadata": {}},
            {
                "step": 4,
                "type": "Observation",
                "text": "The apple is in the inventory.",
                "metadata": {},
            },
        ],
    }


class FakeTokenizer:
    def __init__(self):
        self.vocabulary = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        for token in text.lower().split():
            if token not in self.vocabulary:
                self.vocabulary[token] = len(self.vocabulary) + 10
            result.append(self.vocabulary[token])
        return result


class FixtureTests(unittest.TestCase):
    def test_last_transition_is_causal_and_removes_leaky_think(self):
        examples = build_transition_examples([trajectory()], mode="last")
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.action["text"], "pick up apple")
        self.assertEqual(example.target_observation, "The apple is in the inventory.")
        self.assertFalse(
            any("Future leak" in block["text"] for block in example.history)
        )
        self.assertEqual(example.eval_split, "ood")
        self.assertEqual(example.previous_state["inventory"], [])
        self.assertEqual(example.target_state["inventory"], ["apple"])

    def test_all_mode_emits_each_action_observation_pair(self):
        examples = build_transition_examples([trajectory()], mode="all")
        self.assertEqual([example.target_step for example in examples], [2, 4])
        self.assertEqual(len({example.example_id for example in examples}), 2)

    def test_training_and_fixture_nextobs_tokens_are_identical(self):
        report = check_nextobs_parity([trajectory()], FakeTokenizer(), max_len=1024)
        self.assertTrue(report["passed"])
        self.assertEqual(report["checked"], 1)
        self.assertEqual(report["mismatch_count"], 0)

    def test_split_assignment_is_attached_to_fixture(self):
        row = trajectory()
        assignments = {
            row["trajectory_id"]: {
                "task_id": "held-out-template",
                "eval_split": "ood",
                "novelty_group": "goal:x",
                "novelty_reason": "unseen_goal_template",
            }
        }
        fixture = build_transition_examples(
            [row], mode="last", assignments=assignments
        )[0]
        self.assertEqual(fixture.eval_split, "ood")
        self.assertEqual(fixture.task_id, "held-out-template")
        self.assertEqual(fixture.metadata["novelty_reason"], "unseen_goal_template")

    def test_block_level_state_labels_are_preserved(self):
        row = trajectory()
        row["state_labels"] = {}
        observations = [
            block for block in row["blocks"] if block["type"] == "Observation"
        ]
        observations[0]["state_labels"] = {"inventory": []}
        observations[1]["state_labels"] = {"inventory": ["apple"]}
        example = build_transition_examples([row], mode="last")[0]
        self.assertEqual(example.previous_state, {"inventory": []})
        self.assertEqual(example.target_state, {"inventory": ["apple"]})


class MetricTests(unittest.TestCase):
    def test_text_metrics_are_normalized_but_number_sensitive(self):
        same = text_metrics(" Temperature is 10 C ", "temperature IS 10 c")
        different = text_metrics("temperature is 11 c", "temperature is 10 c")
        self.assertEqual(same["exact_match"], 1.0)
        self.assertEqual(same["number_set_match"], 1.0)
        self.assertEqual(different["number_set_match"], 0.0)

    def test_state_metrics_separate_changes_from_stable_keys(self):
        result = state_metrics(
            {"room": "kitchen", "inventory": ["apple"]},
            {"room": "kitchen", "inventory": ["apple"]},
            {"room": "kitchen", "inventory": []},
        )
        self.assertEqual(result["state_key_accuracy"], 1.0)
        self.assertEqual(result["changed_key_accuracy"], 1.0)
        self.assertEqual(result["stable_key_accuracy"], 1.0)

    def test_prediction_scoring_requires_complete_coverage(self):
        fixtures = build_transition_examples([trajectory()], mode="all")
        with self.assertRaisesRegex(ValueError, "Missing predictions"):
            score_predictions(fixtures, [], bootstrap_samples=20)

    def test_prediction_scoring_reports_state_and_groups(self):
        fixture = build_transition_examples([trajectory()], mode="last")[0]
        report, failures = score_predictions(
            [fixture],
            [
                {
                    "example_id": fixture.example_id,
                    "prediction": fixture.target_observation,
                    "predicted_state": fixture.target_state,
                    "nll": 1.25,
                    "target_tokens": 7,
                }
            ],
            bootstrap_samples=20,
        )
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["structured_state"]["status"], "available")
        self.assertEqual(report["metrics"]["exact_match"]["mean"], 1.0)
        self.assertIn("ood", report["groups"]["eval_split"])
        self.assertEqual(failures, [])

    def test_context_hash_mismatch_is_rejected(self):
        fixture = build_transition_examples([trajectory()], mode="last")[0]
        with self.assertRaisesRegex(ValueError, "different context hash"):
            score_predictions(
                [fixture],
                [
                    {
                        "example_id": fixture.example_id,
                        "context_hash": "wrong",
                        "prediction": fixture.target_observation,
                    }
                ],
                bootstrap_samples=20,
            )


class TaskSuccessTests(unittest.TestCase):
    def test_reports_absolute_ood_score_with_gap(self):
        records = [
            {
                "episode_id": "id-1",
                "condition": "dflex",
                "environment": "scienceworld",
                "eval_split": "id",
                "seed": 0,
                "normalized_score": 0.9,
                "success": True,
            },
            {
                "episode_id": "ood-1",
                "condition": "dflex",
                "environment": "scienceworld",
                "eval_split": "ood",
                "seed": 0,
                "normalized_score": 0.7,
                "success": True,
            },
        ]
        report = score_task_success(records, bootstrap_samples=20)
        aggregate = report["conditions"]["dflex"]["aggregate"]
        self.assertAlmostEqual(aggregate["ood_gap"]["value"], 0.2)
        self.assertEqual(aggregate["absolute_ood_score"], 0.7)
        self.assertEqual(aggregate["status"], "complete")


class SplitManifestTests(unittest.TestCase):
    def test_goal_templates_normalize_ids_and_numbers(self):
        self.assertEqual(
            goal_template("Put unknown substance B in box 12"),
            "put unknown substance <id> in box <id>",
        )

    def test_unseen_template_is_ood(self):
        training = [{**trajectory(), "goal": "Pick up apple 1"}]
        seen = {**trajectory(), "trajectory_id": "seen", "goal": "Pick up apple 2"}
        unseen = {**trajectory(), "trajectory_id": "unseen", "goal": "Heat water"}
        manifest = build_novelty_manifest(
            training,
            [seen, unseen],
            environment="scienceworld",
            strategy="goal_template",
        )
        assignments = {row["trajectory_id"]: row for row in manifest["assignments"]}
        self.assertEqual(assignments["seen"]["eval_split"], "id")
        self.assertEqual(assignments["unseen"]["eval_split"], "ood")


class ComparisonTests(unittest.TestCase):
    def test_parse_run_spec_preserves_windows_path(self):
        self.assertEqual(
            parse_run_spec(r"dflex:2:C:\runs\predictions.jsonl"),
            ("dflex", 2, r"C:\runs\predictions.jsonl"),
        )

    def test_paired_comparison_uses_common_examples_and_seeds(self):
        fixture = build_transition_examples([trajectory()], mode="last")[0]
        with TemporaryDirectory() as temporary:
            paths = {}
            for condition, nll, prediction in (
                ("dflex", 1.0, fixture.target_observation),
                ("dar", 2.0, "wrong observation"),
            ):
                for run_seed in (0, 1):
                    path = Path(temporary) / f"{condition}-{run_seed}.jsonl"
                    write_jsonl(
                        path,
                        [
                            {
                                "example_id": fixture.example_id,
                                "context_hash": fixture.context_hash,
                                "prediction": prediction,
                                "nll": nll,
                                "target_tokens": 5,
                            }
                        ],
                    )
                    paths[(condition, run_seed)] = str(path)
            report = compare_runs(
                [fixture],
                [(condition, seed, path) for (condition, seed), path in paths.items()],
                primary="dflex",
                bootstrap_samples=20,
            )
        nll = report["paired_improvements"]["dar"]["nll"]
        self.assertEqual(nll["mean"], 1.0)
        self.assertEqual(nll["seed_count"], 2)
        self.assertEqual(nll["win_rate"], 1.0)


class DecodeTests(unittest.TestCase):
    def test_iterative_and_semi_autoregressive_fill_every_target(self):
        import torch

        class Result:
            def __init__(self, logits):
                self.logits = logits

        class Model:
            def __call__(self, input_ids, attention_mask):
                del attention_mask
                logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
                for position in range(input_ids.shape[1]):
                    logits[:, position, 3 + position % 3] = 10.0 + position
                return Result(logits)

        model = Model()
        inputs = torch.tensor([[7, 1, 1, 1, 1]])
        valid = torch.ones_like(inputs, dtype=torch.bool)
        initial = model(inputs, None).logits
        for mode in ("iterative_diffusion", "semi_autoregressive"):
            decoded, passes = _decode_masked_target(
                model,
                inputs,
                valid,
                target_start=1,
                target_end=5,
                mask_id=1,
                pad_id=0,
                attention_dtype=torch.float32,
                mode=mode,
                steps=2,
                initial_logits=initial,
            )
            self.assertEqual(len(decoded), 4)
            self.assertNotIn(1, decoded.tolist())
            self.assertEqual(passes, 2)


class ProbeTests(unittest.TestCase):
    def test_layer_probe_and_controls_are_reported(self):
        import numpy as np

        with TemporaryDirectory() as temporary:
            ids = np.array([f"e-{index}" for index in range(40)])
            labels = np.array([index % 2 for index in range(40)])
            features = np.column_stack((labels * 4.0 - 2.0, np.arange(40) * 0.001))
            activation_path = Path(temporary) / "activations.npz"
            np.savez(activation_path, example_ids=ids, layer_0=features)
            records = [
                {
                    "example_id": f"e-{index}",
                    "split": "train" if index < 20 else "test",
                    "group_id": f"group-{index}",
                    "labels": {"inventory_flag": int(labels[index])},
                    "text": f"example text class {labels[index]}",
                }
                for index in range(40)
            ]
            report = evaluate_probes(str(activation_path), records, seed=0)
        target = report["targets"]["inventory_flag"]
        self.assertEqual(target["status"], "complete")
        self.assertEqual(target["layers"]["layer_0"]["accuracy"], 1.0)
        self.assertIn("majority", target["controls"])
        self.assertIn("bag_of_words_target_redacted", target["controls"])

    def test_probe_group_leakage_is_rejected(self):
        import numpy as np

        with TemporaryDirectory() as temporary:
            activation_path = Path(temporary) / "activations.npz"
            np.savez(
                activation_path,
                example_ids=np.array(["a", "b"]),
                layer_0=np.array([[0.0], [1.0]]),
            )
            records = [
                {
                    "example_id": "a",
                    "split": "train",
                    "group_id": "same",
                    "labels": {"x": 0},
                },
                {
                    "example_id": "b",
                    "split": "test",
                    "group_id": "same",
                    "labels": {"x": 1},
                },
            ]
            with self.assertRaisesRegex(ValueError, "group leakage"):
                evaluate_probes(str(activation_path), records)

    def test_activation_pooling_never_uses_future_positions(self):
        import torch

        hidden = torch.tensor([[[1.0], [2.0], [100.0]]])
        self.assertEqual(_pool_hidden(hidden, 2, "final_context_token").item(), 2.0)
        self.assertEqual(_pool_hidden(hidden, 2, "mean_context").item(), 1.5)


class CoverageTests(unittest.TestCase):
    def test_readiness_distinguishes_trajectory_and_step_labels(self):
        row = trajectory()
        report = audit_rows([row], "scienceworld")
        self.assertTrue(report["capabilities"]["next_observation"])
        self.assertTrue(report["capabilities"]["structured_next_state"])
        self.assertFalse(report["capabilities"]["simulator_task_success"])

    def test_plan_only_data_is_not_next_state_ready(self):
        row = {
            "trajectory_id": "plan-only",
            "env": "alfworld",
            "split": "validation",
            "goal": "do task",
            "metadata": {"offline_plan_only": True},
            "state_labels": {"task_type": "pick_and_place"},
            "blocks": [
                {"step": 0, "type": "Goal", "text": "do task"},
                {"step": 1, "type": "Observation", "text": "initial state"},
                {"step": 1, "type": "Action", "text": "go kitchen"},
            ],
        }
        report = audit_rows([row], "alfworld")
        self.assertFalse(report["capabilities"]["next_observation"])
        self.assertEqual(report["metadata_flags"]["offline_plan_only"], 1)


class LayoutTests(unittest.TestCase):
    def test_preflight_prefers_evaluation_ready_dataset_variants(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "cp2107-textworld-trajectories"
            for variant in (
                "scienceworld-compact-v2",
                "alfworld-success-v2",
                "webshop",
            ):
                directory = data_root / variant
                directory.mkdir(parents=True)
                for split in ("train", "validation", "test"):
                    (directory / f"{split}.jsonl").write_text("{}\n", encoding="utf-8")
            base = root / "models" / "LLaDA2.0-mini"
            base.mkdir(parents=True)
            (base / "model.safetensors.index.json").write_text(
                '{"weight_map":{"x":"model-1.safetensors"}}', encoding="utf-8"
            )
            (base / "model-1.safetensors").write_bytes(b"weight")
            adapter = root / "models" / "AOMT" / "dflex"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            report = workspace_preflight(root)
            self.assertEqual(
                report["datasets"]["scienceworld"]["variant"],
                "scienceworld-compact-v2",
            )
            self.assertTrue(report["evaluation_ready"])


class RobustnessTests(unittest.TestCase):
    def test_counterfactuals_require_environment_validation(self):
        cases = [
            {
                "case_id": "cf-1",
                "environment": "scienceworld",
                "environment_valid": True,
                "validity_source": "scienceworld-replay-v1",
                "edit_type": "remove_prerequisite",
                "base_prediction": "the door opens",
                "base_target": "the door opens",
                "counterfactual_prediction": "the door stays closed",
                "counterfactual_target": "the door stays closed",
            }
        ]
        report, excluded = score_counterfactuals(cases, bootstrap_samples=20)
        self.assertEqual(excluded, [])
        self.assertEqual(report["metrics"]["change_sensitivity_correct"]["mean"], 1.0)

    def test_invalid_counterfactuals_are_excluded(self):
        report, excluded = score_counterfactuals(
            [
                {
                    "case_id": "cf-invalid",
                    "environment_valid": False,
                    "exclusion_reason": "simulator rejected edit",
                }
            ],
            bootstrap_samples=20,
        )
        self.assertEqual(report["valid_cases"], 0)
        self.assertEqual(len(excluded), 1)

    def test_corruption_recovery_reports_first_coherent_step(self):
        report = score_corruption_recovery(
            [
                {
                    "case_id": "corrupt-1",
                    "corruption_type": "impossible_observation",
                    "decode": "iterative_diffusion",
                    "steps": [
                        {
                            "coherent": False,
                            "prediction": "wrong",
                            "target": "state one",
                        },
                        {
                            "coherent": True,
                            "prediction": "state two",
                            "target": "state two",
                        },
                        {
                            "coherent": True,
                            "prediction": "state three",
                            "target": "state three",
                        },
                    ],
                }
            ],
            bootstrap_samples=20,
        )
        self.assertEqual(report["metrics"]["recovery_rate"]["mean"], 1.0)
        self.assertEqual(
            report["metrics"]["steps_to_recovery_recovered_only"]["mean"], 2.0
        )

    def test_corruption_generation_is_deterministic_and_non_mutating(self):
        fixture = build_transition_examples([trajectory()], mode="last")[0]
        original = [dict(block) for block in fixture.history]
        first = generate_corruption_fixtures(
            [fixture], corruption_types=["deleted_block", "duplicated_block"], seed=7
        )
        second = generate_corruption_fixtures(
            [fixture], corruption_types=["deleted_block", "duplicated_block"], seed=7
        )
        self.assertEqual(first, second)
        self.assertEqual(fixture.history, original)
        self.assertTrue(all(row["status"] == "ready" for row in first))

    def test_counterfactual_candidates_cannot_be_scored_before_validation(self):
        fixture = build_transition_examples([trajectory()], mode="last")[0]
        candidates = generate_counterfactual_candidates([fixture], seed=0)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["requires_simulator_validation"])
        self.assertFalse(candidates[0]["environment_valid"])
        self.assertEqual(candidates[0]["status"], "requires_simulator_validation")

    def test_metadata_adapter_refuses_simulator_claims(self):
        adapter = default_adapter("alfworld")
        self.assertFalse(adapter.capabilities.simulator_grounded)
        self.assertEqual(adapter.capabilities.label_scope, "trajectory")
        with self.assertRaisesRegex(RuntimeError, "cannot reset"):
            adapter.reset("task", 0)


class RolloutTests(unittest.TestCase):
    def rollout(self):
        return {
            "rollout_id": "r1",
            "horizon": 3,
            "steps": [
                {
                    "prediction": "state one",
                    "target": "state one",
                    "observation_context_source": "ground_truth_initial",
                    "future_ground_truth_used": False,
                },
                {
                    "prediction": "state two",
                    "target": "state two",
                    "observation_context_source": "model",
                    "future_ground_truth_used": False,
                },
                {
                    "prediction": "wrong",
                    "target": "state three",
                    "observation_context_source": "model",
                    "future_ground_truth_used": False,
                },
            ],
        }

    def test_rollout_requires_model_feedback(self):
        record = self.rollout()
        record["steps"][1]["observation_context_source"] = "ground_truth"
        self.assertTrue(validate_rollout(record))
        with self.assertRaisesRegex(ValueError, "Invalid rollout"):
            score_rollouts([record], bootstrap_samples=20)

    def test_rollout_scores_horizon_and_error_accumulation(self):
        report, invalid = score_rollouts([self.rollout()], bootstrap_samples=20)
        self.assertEqual(invalid, [])
        self.assertEqual(report["by_horizon"]["3"]["completion"]["mean"], 1.0)
        self.assertLess(report["by_horizon"]["3"]["error_accumulation"]["mean"], 0.0)
        self.assertEqual(report["by_horizon"]["3"]["time_to_first_error"]["mean"], 3.0)
        self.assertEqual(report["by_horizon"]["3"]["survival_rate"]["mean"], 0.0)
        self.assertEqual(
            report["by_horizon"]["3"]["validity_bases"], ["exact_text_proxy"]
        )

    def test_generator_feeds_model_observation_and_omits_future_think(self):
        class Predictor:
            def __init__(self):
                self.histories = []

            def predict_observation(self, *, goal, history, action, metadata):
                del goal, action, metadata
                self.histories.append(history)
                return f"model observation {len(self.histories)}"

        predictor = Predictor()
        records = generate_oracle_action_rollouts(
            [trajectory()], predictor, horizons=[2], max_rollouts=1
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["steps"][1]["observation_context_source"], "model")
        second_history = predictor.histories[1]
        observation_texts = [
            block["text"] for block in second_history if block["type"] == "Observation"
        ]
        self.assertEqual(observation_texts[-1], "model observation 1")
        self.assertNotIn("An apple is on the table.", observation_texts)
        self.assertFalse(
            any("Future leak" in block["text"] for block in second_history)
        )


if __name__ == "__main__":
    unittest.main()
