"""Contract tests for intent, slot extraction, decomposition, and planning input."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from earthchange.agent.planner import plan_for_query
from earthchange.nlp import (
    Intent,
    Operation,
    QueryAction,
    QueryConstraint,
    Region,
    RequestedMetric,
    StructuredQuery,
    parse_question,
)


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Is there any change between the images?", Intent.GENERAL_CHANGE),
        ("Where did changes occur?", Intent.CHANGE_LOCALIZATION),
        ("What percentage of the image changed?", Intent.CHANGE_QUANTIFICATION),
        ("Compare the left versus right regions.", Intent.REGIONAL_COMPARISON),
        ("How much changed and where?", Intent.MULTI_STEP_ANALYSIS),
        ("Will it rain tomorrow?", Intent.UNSUPPORTED),
    ],
)
def test_each_intent_family(question: str, intent: Intent) -> None:
    assert parse_question(question).intent == intent


@pytest.mark.parametrize(
    ("phrase", "region"),
    [
        ("upper left", Region.UPPER_LEFT),
        ("top-left", Region.UPPER_LEFT),
        ("northwest", Region.UPPER_LEFT),
        ("upper right", Region.UPPER_RIGHT),
        ("lower-left", Region.LOWER_LEFT),
        ("south east", Region.LOWER_RIGHT),
        ("top", Region.TOP),
        ("bottom", Region.BOTTOM),
        ("west", Region.LEFT),
        ("east", Region.RIGHT),
        ("middle", Region.CENTER),
    ],
)
def test_region_aliases(phrase: str, region: Region) -> None:
    assert parse_question(f"Locate change in the {phrase}.").region == region


def test_quantification_extracts_metric_and_operation() -> None:
    query = parse_question("How many pixels changed?")
    assert query.intent == Intent.CHANGE_QUANTIFICATION
    assert query.operation == Operation.COUNT
    assert query.requested_metrics == [RequestedMetric.CHANGED_PIXEL_COUNT]


def test_area_request_records_resolution_constraint() -> None:
    query = parse_question("Estimate the changed area in square metres.")
    assert query.operation == Operation.AREA
    assert RequestedMetric.CHANGED_AREA in query.requested_metrics
    assert QueryConstraint.PIXEL_RESOLUTION_REQUIRED in query.constraints


def test_region_specific_quantification_requests_region_tool() -> None:
    query = parse_question("What percentage changed in the top half?")
    assert query.region == Region.TOP
    assert RequestedMetric.REGIONAL_CHANGE_PERCENTAGE in query.requested_metrics
    assert QueryAction.ANALYZE_REGIONS in {task.action for task in query.decomposition}


def test_comparison_targets_are_canonical_and_not_expanded() -> None:
    query = parse_question("Compare the upper-left and lower-right corners.")
    assert query.intent == Intent.REGIONAL_COMPARISON
    assert query.comparison_targets == [Region.UPPER_LEFT, Region.LOWER_RIGHT]
    plan = plan_for_query(query)
    region_step = next(step for step in plan.steps if step.tool.value == "region_analysis")
    assert region_step.parameters == {"regions": ["upper_left", "lower_right"]}
    assert plan.structured_query == query


def test_multi_step_rank_and_percentage_is_ordered() -> None:
    query = parse_question("Find the region with the most change and calculate its percentage.")
    actions = [task.action for task in query.decomposition]
    assert query.intent == Intent.MULTI_STEP_ANALYSIS
    assert actions == [
        QueryAction.DETECT_CHANGE,
        QueryAction.ANALYZE_REGIONS,
        QueryAction.COMPARE_REGIONS,
        QueryAction.IDENTIFY_MAX_REGION,
        QueryAction.CALCULATE_METRIC,
    ]
    plan = plan_for_query(query)
    assert "region_analysis" in plan.tool_names()
    assert "area_calculation" in plan.tool_names()


def test_compound_amount_and_location_has_both_metrics() -> None:
    query = parse_question("How much changed, and where?")
    assert query.intent == Intent.MULTI_STEP_ANALYSIS
    assert RequestedMetric.CHANGED_PERCENTAGE in query.requested_metrics
    assert RequestedMetric.REGIONAL_DISTRIBUTION in query.requested_metrics


@pytest.mark.parametrize(
    ("question", "reason", "constraint"),
    [
        ("How much forest was lost?", "semantic_target_not_supported", QueryConstraint.SEMANTIC_CLASS_UNAVAILABLE),
        ("How many individual buildings were destroyed?", "object_count_not_supported", QueryConstraint.OBJECT_COUNT_UNSUPPORTED),
        ("Count the changes.", "count_unit_ambiguous", QueryConstraint.AMBIGUOUS_REQUEST),
        ("Compare them.", "ambiguous_comparison_targets", QueryConstraint.AMBIGUOUS_REQUEST),
        ("What happened there?", "ambiguous_or_unrecognized_request", QueryConstraint.AMBIGUOUS_REQUEST),
    ],
)
def test_unsupported_and_ambiguous_requests(
    question: str, reason: str, constraint: QueryConstraint
) -> None:
    query = parse_question(question)
    assert query.intent == Intent.UNSUPPORTED
    assert query.parameters["reason"] == reason
    assert constraint in query.constraints
    assert query.decomposition == []


def test_empty_invalid_and_overlong_questions() -> None:
    assert parse_question("").parameters["reason"] == "empty_query"
    assert parse_question("  \n ").intent == Intent.UNSUPPORTED
    with pytest.raises(TypeError, match="string"):
        parse_question(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1000"):
        parse_question("x" * 1001)


def test_rule_parser_does_not_emit_unsupported_confidence() -> None:
    assert parse_question("Is there a change?").confidence is None
    assert Intent("general_change") == Intent.GENERAL_CHANGE
    assert Region("upper_left") == Region.UPPER_LEFT
    assert Operation("percentage") == Operation.PERCENTAGE


def test_structured_query_schema_rejects_invalid_enum_values() -> None:
    with pytest.raises(ValidationError):
        StructuredQuery(raw_question="q", intent="not_an_intent")
