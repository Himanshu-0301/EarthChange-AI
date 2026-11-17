"""Dataset integrity checks for the independent NLP experiment."""

from __future__ import annotations

from collections import Counter

from experiments.nlp_evaluation import DATASET_PATH, load_examples, parameter_extraction_metrics


def test_question_dataset_has_disjoint_balanced_splits() -> None:
    splits = load_examples(DATASET_PATH)
    assert {name: len(items) for name, items in splits.items()} == {
        "train": 36,
        "validation": 18,
        "test": 18,
    }
    questions = [item["question"].casefold() for items in splits.values() for item in items]
    assert len(questions) == len(set(questions))
    for items in splits.values():
        assert set(Counter(item["expected"]["intent"] for item in items)) == {
            "GENERAL_CHANGE",
            "CHANGE_LOCALIZATION",
            "CHANGE_QUANTIFICATION",
            "REGIONAL_COMPARISON",
            "MULTI_STEP_ANALYSIS",
            "UNSUPPORTED",
        }


def test_parameter_metrics_are_computed_from_dataset_annotations() -> None:
    test_examples = load_examples(DATASET_PATH)["test"]
    metrics = parameter_extraction_metrics(test_examples)
    assert metrics["examples"] == 18
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
