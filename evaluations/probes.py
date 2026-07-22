"""Layer-wise linear probes with leakage and baseline controls."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from .schema import RESULT_SCHEMA_VERSION, file_sha256, read_jsonl, write_json


def _classification_metrics(
    expected: Sequence[Any], predicted: Sequence[Any]
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, average="macro", zero_division=0)
        ),
    }


def _redact_label(text: str, label: Any) -> str:
    value = str(label).strip()
    if not value:
        return text
    return text.replace(value, "<TARGET>").replace(value.lower(), "<TARGET>")


def _validate_group_isolation(records: Sequence[Dict[str, Any]]) -> None:
    split_groups: Dict[str, set[str]] = {}
    for record in records:
        group = record.get("group_id")
        if group is None:
            continue
        split_groups.setdefault(str(record["split"]), set()).add(str(group))
    train = split_groups.get("train", set())
    test = split_groups.get("test", set()) | split_groups.get("validation", set())
    overlap = sorted(train & test)
    if overlap:
        raise ValueError(
            f"Probe group leakage across train/evaluation splits: {overlap[:5]}"
        )


def _activation_layers(path: str) -> tuple[List[str], Dict[str, np.ndarray]]:
    archive = np.load(path, allow_pickle=False)
    if "example_ids" not in archive:
        raise ValueError("Activation archive needs example_ids")
    ids = [str(value) for value in archive["example_ids"].tolist()]
    layers = {
        key: np.asarray(archive[key], dtype=np.float32)
        for key in archive.files
        if key != "example_ids"
    }
    if not layers:
        raise ValueError("Activation archive contains no layer matrices")
    for name, matrix in layers.items():
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError(f"Layer {name} must have shape [examples, features]")
    return ids, layers


def evaluate_probes(
    activation_path: str,
    label_records: Iterable[Dict[str, Any]],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    records = list(label_records)
    _validate_group_isolation(records)
    activation_ids, layers = _activation_layers(activation_path)
    activation_index = {
        example_id: index for index, example_id in enumerate(activation_ids)
    }
    record_by_id = {str(record["example_id"]): record for record in records}
    common_ids = [
        example_id for example_id in activation_ids if example_id in record_by_id
    ]
    target_names = sorted(
        {
            target
            for example_id in common_ids
            for target in (record_by_id[example_id].get("labels") or {})
        }
    )
    results: Dict[str, Any] = {}
    for target_name in target_names:
        target_ids = [
            example_id
            for example_id in common_ids
            if target_name in (record_by_id[example_id].get("labels") or {})
        ]
        train_ids = [
            example_id
            for example_id in target_ids
            if record_by_id[example_id].get("split") == "train"
        ]
        test_ids = [
            example_id
            for example_id in target_ids
            if record_by_id[example_id].get("split") in {"validation", "test"}
        ]
        y_train = [record_by_id[item]["labels"][target_name] for item in train_ids]
        y_test = [record_by_id[item]["labels"][target_name] for item in test_ids]
        if not train_ids or not test_ids or len({str(value) for value in y_train}) < 2:
            results[target_name] = {
                "status": "insufficient_data",
                "train_examples": len(train_ids),
                "test_examples": len(test_ids),
            }
            continue
        counts: Dict[Any, int] = {}
        for value in y_train:
            counts[value] = counts.get(value, 0) + 1
        majority = max(counts, key=counts.get)
        controls: Dict[str, Any] = {
            "majority": _classification_metrics(y_test, [majority] * len(y_test))
        }
        train_text = [str(record_by_id[item].get("text", "")) for item in train_ids]
        test_text = [str(record_by_id[item].get("text", "")) for item in test_ids]
        if any(train_text) and any(test_text):
            for name, train_values, test_values in (
                ("bag_of_words", train_text, test_text),
                (
                    "bag_of_words_target_redacted",
                    [
                        _redact_label(text, label)
                        for text, label in zip(train_text, y_train)
                    ],
                    [
                        _redact_label(text, label)
                        for text, label in zip(test_text, y_test)
                    ],
                ),
            ):
                model = make_pipeline(
                    TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=seed
                    ),
                )
                model.fit(train_values, y_train)
                controls[name] = _classification_metrics(
                    y_test, model.predict(test_values)
                )

        layer_results = {}
        train_indices = [activation_index[item] for item in train_ids]
        test_indices = [activation_index[item] for item in test_ids]
        for layer_name, matrix in sorted(layers.items()):
            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            )
            probe.fit(matrix[train_indices], y_train)
            layer_result = _classification_metrics(
                y_test, probe.predict(matrix[test_indices])
            )
            shuffled = list(y_train)
            random.Random(seed).shuffle(shuffled)
            shuffled_probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=seed
                ),
            )
            shuffled_probe.fit(matrix[train_indices], shuffled)
            layer_result["shuffled_label_accuracy"] = float(
                _classification_metrics(
                    y_test, shuffled_probe.predict(matrix[test_indices])
                )["accuracy"]
            )
            layer_results[layer_name] = layer_result
        results[target_name] = {
            "status": "complete",
            "train_examples": len(train_ids),
            "test_examples": len(test_ids),
            "classes": sorted({str(value) for value in y_train + y_test}),
            "controls": controls,
            "layers": layer_results,
        }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "evaluation": "latent_state_linear_probes",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "activation_path": activation_path,
        "activation_sha256": file_sha256(activation_path),
        "activation_examples": len(activation_ids),
        "label_examples": len(records),
        "matched_examples": len(common_ids),
        "seed": seed,
        "targets": results,
    }


def evaluate_probes_to_file(
    activation_path: str, labels_path: str, output_path: str, *, seed: int = 0
) -> Dict[str, Any]:
    report = evaluate_probes(activation_path, read_jsonl(labels_path), seed=seed)
    report["labels_path"] = labels_path
    report["labels_sha256"] = file_sha256(labels_path)
    write_json(output_path, report)
    return report
