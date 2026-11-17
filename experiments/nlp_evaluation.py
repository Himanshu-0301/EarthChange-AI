"""Compare the rule-based intent parser with a TF-IDF + logistic baseline.

Run with ``python experiments/nlp_evaluation.py`` after installing the
optional ``nlp-eval`` extra. No satellite images or CV tools are loaded.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from earthchange.nlp.classifier import fit_tfidf_intent_classifier
from earthchange.nlp.parser import parse_question

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "nlp" / "question_dataset.json"
DEFAULT_OUTPUT = ROOT / "experiments" / "outputs" / "nlp" / "evaluation.json"


def load_examples(path: Path = DATASET_PATH) -> dict[str, list[dict[str, Any]]]:
    """Load and validate the manually annotated, grouped split dataset."""
    dataset = json.loads(path.read_text(encoding="utf-8"))
    profiles = dataset["profiles"]
    splits = dataset["splits"]
    if set(splits) != {"train", "validation", "test"}:
        raise ValueError("Dataset must have train, validation, and test splits")
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    result: dict[str, list[dict[str, Any]]] = {}
    for split, examples in splits.items():
        result[split] = []
        for example in examples:
            question = example["question"].strip()
            key = " ".join(question.casefold().split())
            if example["id"] in seen_ids or key in seen_questions:
                raise ValueError(f"Duplicate ID or question in dataset: {example['id']}")
            if example["profile"] not in profiles:
                raise ValueError(f"Unknown annotation profile: {example['profile']}")
            seen_ids.add(example["id"])
            seen_questions.add(key)
            result[split].append({**example, "question": question, "expected": profiles[example["profile"]]})

    classes_by_split = {
        split: {item["expected"]["intent"] for item in examples}
        for split, examples in result.items()
    }
    if not classes_by_split["train"] <= classes_by_split["validation"] or not classes_by_split["train"] <= classes_by_split["test"]:
        raise ValueError("Every intent represented in training must also appear in validation and test")
    if not all(result.values()):
        raise ValueError("Each dataset split must contain at least one example")
    return result


def _classification_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "unsupported_precision": float(precision_score(
            [label == "UNSUPPORTED" for label in y_true],
            [label == "UNSUPPORTED" for label in y_pred],
            zero_division=0,
        )),
        "unsupported_recall": float(recall_score(
            [label == "UNSUPPORTED" for label in y_true],
            [label == "UNSUPPORTED" for label in y_pred],
            zero_division=0,
        )),
        "unsupported_f1": float(f1_score(
            [label == "UNSUPPORTED" for label in y_true],
            [label == "UNSUPPORTED" for label in y_pred],
            zero_division=0,
        )),
    }


def _query_parameter_tokens(query: Any) -> set[str]:
    tokens = {
        f"target:{query.target}",
        f"region:{query.region.value.lower()}",
        f"operation:{query.operation.value.lower()}",
    }
    tokens.update(f"metric:{item.value}" for item in query.requested_metrics)
    tokens.update(f"comparison_target:{item.value.lower()}" for item in query.comparison_targets)
    tokens.update(f"constraint:{item.value}" for item in query.constraints)
    tokens.update(f"task:{item.action.value}" for item in query.decomposition)
    return tokens


def _expected_parameter_tokens(expected: dict[str, Any]) -> set[str]:
    tokens = {
        f"target:{expected['target']}",
        f"region:{expected['region']}",
        f"operation:{expected['operation']}",
    }
    tokens.update(f"metric:{item}" for item in expected["metrics"])
    tokens.update(f"comparison_target:{item}" for item in expected["comparison_targets"])
    tokens.update(f"constraint:{item}" for item in expected["constraints"])
    tokens.update(f"task:{item}" for item in expected["decomposition"])
    return tokens


def parameter_extraction_metrics(examples: list[dict[str, Any]]) -> dict[str, float | int]:
    """Micro precision/recall/F1 across typed slot and decomposition values."""
    true_positive = false_positive = false_negative = 0
    exact_examples = 0
    for example in examples:
        predicted = _query_parameter_tokens(parse_question(example["question"]))
        expected = _expected_parameter_tokens(example["expected"])
        true_positive += len(predicted & expected)
        false_positive += len(predicted - expected)
        false_negative += len(expected - predicted)
        exact_examples += predicted == expected
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_query_parameter_match": exact_examples / len(examples),
        "examples": len(examples),
        "true_positive_slots": true_positive,
        "false_positive_slots": false_positive,
        "false_negative_slots": false_negative,
    }


def evaluate(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    """Tune only on validation, then compare both systems on the held-out test set."""
    splits = load_examples(dataset_path)
    train, validation, test = (splits[key] for key in ("train", "validation", "test"))
    train_x = [item["question"] for item in train]
    train_y = [item["expected"]["intent"] for item in train]
    val_x = [item["question"] for item in validation]
    val_y = [item["expected"]["intent"] for item in validation]
    test_x = [item["question"] for item in test]
    test_y = [item["expected"]["intent"] for item in test]

    best: tuple[float, float, tuple[int, int]] | None = None
    for c_value in (0.25, 1.0, 4.0):
        for ngrams in ((1, 1), (1, 2)):
            candidate = fit_tfidf_intent_classifier(train_x, train_y, c=c_value, ngram_range=ngrams)
            score = _classification_metrics(val_y, list(candidate.predict(val_x)))["macro_f1"]
            if best is None or score > best[0]:
                best = (score, c_value, ngrams)
    assert best is not None
    _, selected_c, selected_ngrams = best

    final_x = train_x + val_x
    final_y = train_y + val_y
    model = fit_tfidf_intent_classifier(final_x, final_y, c=selected_c, ngram_range=selected_ngrams)
    ml_predictions = [str(value) for value in model.predict(test_x)]
    rule_predictions = [parse_question(question).intent.value for question in test_x]
    rule_queries = [parse_question(question) for question in test_x]
    gold_multistep = [item for item in test if item["expected"]["intent"] == "MULTI_STEP_ANALYSIS"]
    correct_decompositions = sum(
        [step.action.value for step in parse_question(item["question"]).decomposition]
        == item["expected"]["decomposition"]
        for item in gold_multistep
    )

    predictions = []
    for example, rule_query, ml_label in zip(test, rule_queries, ml_predictions, strict=True):
        predictions.append({
            "id": example["id"],
            "question": example["question"],
            "expected_intent": example["expected"]["intent"],
            "rule_intent": rule_query.intent.value,
            "tfidf_logistic_intent": ml_label,
            "rule_parameters_match": _query_parameter_tokens(rule_query) == _expected_parameter_tokens(example["expected"]),
        })

    counts = {
        split: dict(sorted(Counter(item["expected"]["intent"] for item in examples).items()))
        for split, examples in splits.items()
    }
    return {
        "dataset": {
            "path": str(dataset_path),
            "version": json.loads(dataset_path.read_text(encoding="utf-8"))["version"],
            "total_examples": sum(map(len, splits.values())),
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "class_counts": counts,
            "note": "Small curated research set; metrics are descriptive and not a generalization claim.",
        },
        "test": {
            "examples": len(test),
            "rule_parser_intent": _classification_metrics(test_y, rule_predictions),
            "tfidf_logistic_intent": _classification_metrics(test_y, ml_predictions),
            "rule_parser_parameter_extraction": parameter_extraction_metrics(test),
            "multi_step_decomposition_exact_match": {
                "correct": correct_decompositions,
                "total": len(gold_multistep),
                "accuracy": correct_decompositions / len(gold_multistep) if gold_multistep else 0.0,
            },
        },
        "classifier_selection": {
            "method": "validation macro F1",
            "selected_c": selected_c,
            "selected_ngram_range": list(selected_ngrams),
            "validation_macro_f1": best[0],
            "trained_on": "train + validation only after model selection",
        },
        "test_predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved NLP evaluation to {args.output}")
    print(json.dumps(result["test"], indent=2))


if __name__ == "__main__":
    main()
