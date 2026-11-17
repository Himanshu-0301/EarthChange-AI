"""Interpretable baseline parser for satellite-change questions.

The parser handles a deliberately bounded vocabulary. It identifies intent,
slots, supported metrics, known constraints, and a small set of ordered tasks.
It does not infer semantic land-cover labels or object counts from binary masks.
"""

from __future__ import annotations

import re

from earthchange.nlp.schema import (
    Intent,
    Operation,
    QueryAction,
    QueryConstraint,
    QuerySubtask,
    Region,
    RequestedMetric,
    StructuredQuery,
)

_UNSUPPORTED_PATTERNS = (
    re.compile(r"\b(weather|temperature|forecast|predict|who owns|who built)\b", re.I),
    re.compile(r"\bwhen\s+was\b.*\bbuilt\b", re.I),
)
_TARGETS = (
    "deforestation", "vegetation", "infrastructure", "building", "tree",
    "forest", "water", "river", "lake", "flooding", "flood", "urban",
    "road", "construction", "house", "vehicle", "object", "structure",
)
_OBJECT_WORDS = {"building", "tree", "house", "vehicle", "object", "road", "structure"}

_CORNER_PATTERNS: tuple[tuple[Region, re.Pattern[str]], ...] = (
    (Region.UPPER_LEFT, re.compile(r"\b(?:upper|top)[\s-]+left\b|\bnorth[\s-]+west\b|\bnorthwest\b", re.I)),
    (Region.UPPER_RIGHT, re.compile(r"\b(?:upper|top)[\s-]+right\b|\bnorth[\s-]+east\b|\bnortheast\b", re.I)),
    (Region.LOWER_LEFT, re.compile(r"\b(?:lower|bottom)[\s-]+left\b|\bsouth[\s-]+west\b|\bsouthwest\b", re.I)),
    (Region.LOWER_RIGHT, re.compile(r"\b(?:lower|bottom)[\s-]+right\b|\bsouth[\s-]+east\b|\bsoutheast\b", re.I)),
)
_REGION_PATTERNS: tuple[tuple[Region, re.Pattern[str]], ...] = (
    (Region.TOP, re.compile(r"\b(top|upper|north|northern)\b", re.I)),
    (Region.BOTTOM, re.compile(r"\b(bottom|lower|south|southern)\b", re.I)),
    (Region.LEFT, re.compile(r"\b(left|west|western)\b", re.I)),
    (Region.RIGHT, re.compile(r"\b(right|east|eastern)\b", re.I)),
    (Region.CENTER, re.compile(r"\b(center|centre|middle)\b", re.I)),
)
_AMOUNT = re.compile(
    r"\b(how\s+much|how\s+many|how\s+large|percentage|percent|fraction|proportion|share|quantif|"
    r"total\s+area|changed\s+area|count|number\s+of|size\s+of)\b",
    re.I,
)
_LOCATION = re.compile(r"\b(where|location|which\s+(?:[\w-]+\s+){0,2}(?:areas?|parts?|regions?)|locate|locali[sz])\b", re.I)
_COMPARISON = re.compile(
    r"\b(compare|contrast|versus|vs\.?|differ\s+between|difference\s+between|"
    r"which\s+(?:side|region|area).{0,24}(?:more|less))\b", re.I
)
_RANKING = re.compile(r"\b(most|greatest|highest|largest|biggest|maximum|rank(?:ed)?|more|least)\b", re.I)
_GENERAL_CHANGE = re.compile(r"\b(change|changed|changes|unchanged|differ|different|difference|differences|transform\w*)\b", re.I)
_PERCENTAGE = re.compile(r"\b(percent(?:age)?|fraction|proportion|how\s+much)\b", re.I)
_AREA = re.compile(r"\b(area|hectares?|acres?|square\s+(?:metres|meters|kilomet(?:er|re)s))\b", re.I)
_SPATIAL_AREA = re.compile(r"\b(area|areas)\b(?=\s+(?:of\s+)?(?:the\s+)?(?:image|scene|region|part|landscape))", re.I)
_PIXEL_COUNT = re.compile(r"\b(pixel|pixels)\b", re.I)
_MULTI_MARKERS = re.compile(r"\b(and\s+also|additionally|first\b.*\bthen)\b", re.I)


def _extract_target(question: str) -> str:
    for target in _TARGETS:
        if re.search(rf"\b{re.escape(target)}s?\b", question, re.I):
            return target
    return "change"


def _extract_regions(question: str) -> list[Region]:
    """Extract canonical image regions; corner aliases suppress their components."""
    regions: list[Region] = []
    masked = list(question)
    for region, pattern in _CORNER_PATTERNS:
        for match in pattern.finditer(question):
            if region not in regions:
                regions.append(region)
            for index in range(match.start(), match.end()):
                masked[index] = " "
    remainder = "".join(masked)
    for region, pattern in _REGION_PATTERNS:
        if pattern.search(remainder) and region not in regions:
            regions.append(region)
    return regions


def _unsupported_query(
    raw: str,
    reason: str,
    *,
    target: str = "change",
    region: Region = Region.FULL_IMAGE,
    operation: Operation = Operation.DESCRIBE,
    metrics: list[RequestedMetric] | None = None,
    constraints: list[QueryConstraint] | None = None,
) -> StructuredQuery:
    return StructuredQuery(
        raw_question=raw,
        intent=Intent.UNSUPPORTED,
        target=target,
        region=region,
        operation=operation,
        requested_metrics=metrics or [],
        constraints=constraints or [QueryConstraint.AMBIGUOUS_REQUEST],
        parameters={"reason": reason},
    )


def parse_question(question: str) -> StructuredQuery:
    """Parse a bounded natural-language request into validated structured fields.

    Raises:
        TypeError: ``question`` is not a string.
        ValueError: question exceeds the schema's 1000-character limit.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    raw = question.strip()
    if len(raw) > 1000:
        raise ValueError("question must be no longer than 1000 characters")
    if not raw:
        return _unsupported_query("", "empty_query")

    lowered = raw.lower()
    if any(pattern.search(lowered) for pattern in _UNSUPPORTED_PATTERNS):
        return _unsupported_query(raw, "out_of_scope")

    target = _extract_target(lowered)
    asks_amount = bool(_AMOUNT.search(lowered))
    if _AREA.search(lowered) and re.search(r"\b(?:calculate|estimate|measure|what\s+is|give|report)\b.{0,30}\barea\b|\barea\s+in\s+(?:square|hectares?|acres?)\b", lowered):
        asks_amount = True
    asks_comparison = bool(_COMPARISON.search(lowered))
    regions = _extract_regions(lowered)
    asks_location = bool(
        _LOCATION.search(lowered)
        or (regions and re.search(r"\b(which|what)\s+changes?\b", lowered))
        or (regions and not asks_amount and _GENERAL_CHANGE.search(lowered))
    )
    asks_ranking = bool(
        _RANKING.search(lowered)
        and (re.search(r"\b(region|area|part|zone|side)\b", lowered) or len(regions) > 1)
    )
    explicit_multi = bool(_MULTI_MARKERS.search(lowered))

    if target != "change":
        reason = "object_count_not_supported" if (
            asks_amount and target in _OBJECT_WORDS
        ) else "semantic_target_not_supported"
        constraint = (
            QueryConstraint.OBJECT_COUNT_UNSUPPORTED
            if reason == "object_count_not_supported"
            else QueryConstraint.SEMANTIC_CLASS_UNAVAILABLE
        )
        unsupported_operation = (
            Operation.COUNT if asks_amount and target in _OBJECT_WORDS
            else Operation.AREA if asks_amount and _AREA.search(lowered)
            else Operation.PERCENTAGE if asks_amount
            else Operation.LOCATE if asks_location or regions
            else Operation.DESCRIBE
        )
        requested_metric = (
            RequestedMetric.OBJECT_COUNT if target in _OBJECT_WORDS and asks_amount
            else RequestedMetric.SEMANTIC_AREA if asks_amount and _AREA.search(lowered)
            else RequestedMetric.SEMANTIC_CLASS_CHANGE
        )
        return _unsupported_query(
            raw,
            reason,
            target=target,
            region=regions[0] if regions else Region.FULL_IMAGE,
            operation=unsupported_operation,
            metrics=[requested_metric],
            constraints=[constraint, *([QueryConstraint.IMAGE_REGION_ONLY] if asks_location or regions else [])],
        )

    asks_count = bool(re.search(r"\b(how\s+many|count|number\s+of)\b", lowered))
    if asks_count and not _PIXEL_COUNT.search(lowered):
        return _unsupported_query(
            raw,
            "count_unit_ambiguous",
            operation=Operation.COUNT,
            constraints=[QueryConstraint.AMBIGUOUS_REQUEST, QueryConstraint.PIXEL_LEVEL_ONLY],
        )

    if asks_comparison and len(regions) < 2 and not asks_ranking:
        return _unsupported_query(raw, "ambiguous_comparison_targets", operation=Operation.COMPARE)

    if not any((asks_amount, asks_location, asks_comparison, asks_ranking)):
        if len(regions) > 1:
            intent = Intent.REGIONAL_COMPARISON
        elif regions and _GENERAL_CHANGE.search(lowered):
            intent = Intent.CHANGE_LOCALIZATION
        elif _GENERAL_CHANGE.search(lowered):
            intent = Intent.GENERAL_CHANGE
        else:
            return _unsupported_query(raw, "ambiguous_or_unrecognized_request")
    elif (
        (asks_ranking and asks_amount)
        or (asks_amount and (asks_location or asks_comparison))
        or (explicit_multi and (asks_amount or asks_location or asks_comparison))
    ):
        intent = Intent.MULTI_STEP_ANALYSIS
    elif asks_comparison or asks_ranking or (len(regions) > 1 and not asks_amount):
        intent = Intent.REGIONAL_COMPARISON
    elif len(regions) > 1:
        intent = Intent.REGIONAL_COMPARISON
    elif asks_amount:
        intent = Intent.CHANGE_QUANTIFICATION
    elif asks_location or regions:
        intent = Intent.CHANGE_LOCALIZATION
    else:
        intent = Intent.GENERAL_CHANGE

    comparison_targets = regions if (
        asks_comparison or asks_ranking or (intent == Intent.REGIONAL_COMPARISON and len(regions) > 1)
    ) else []
    region = regions[0] if regions else Region.FULL_IMAGE
    if intent == Intent.REGIONAL_COMPARISON and len(regions) >= 2:
        region = Region.FULL_IMAGE

    if asks_comparison or (asks_ranking and intent == Intent.REGIONAL_COMPARISON) or (len(regions) >= 2 and intent == Intent.REGIONAL_COMPARISON):
        operation = Operation.COMPARE
    elif asks_amount and _AREA.search(lowered) and not _PERCENTAGE.search(lowered):
        operation = Operation.AREA
    elif asks_amount and re.search(r"\bhow\s+many\b|\bcount\b|\bnumber\s+of\b", lowered):
        operation = Operation.COUNT
    elif asks_amount:
        operation = Operation.PERCENTAGE
    elif asks_location or asks_ranking:
        operation = Operation.LOCATE
    else:
        operation = Operation.DESCRIBE

    metrics: list[RequestedMetric] = []
    if asks_ranking:
        metrics.append(RequestedMetric.HIGHEST_CHANGE_REGION)
    if asks_amount:
        if operation == Operation.AREA:
            metrics.append(RequestedMetric.CHANGED_AREA)
        elif operation == Operation.COUNT:
            metrics.append(RequestedMetric.CHANGED_PIXEL_COUNT)
        elif (asks_ranking and intent == Intent.MULTI_STEP_ANALYSIS) or region != Region.FULL_IMAGE:
            metrics.append(RequestedMetric.REGIONAL_CHANGE_PERCENTAGE)
        else:
            metrics.append(RequestedMetric.CHANGED_PERCENTAGE)
    if asks_location:
        metrics.append(RequestedMetric.REGIONAL_DISTRIBUTION)
    if intent == Intent.REGIONAL_COMPARISON and RequestedMetric.REGIONAL_DISTRIBUTION not in metrics:
        metrics.append(RequestedMetric.REGIONAL_DISTRIBUTION)
    if intent == Intent.GENERAL_CHANGE:
        metrics.append(RequestedMetric.CHANGE_PRESENCE)

    tasks = [QuerySubtask(action=QueryAction.DETECT_CHANGE)]
    needs_regions = any(
        metric in (RequestedMetric.REGIONAL_DISTRIBUTION, RequestedMetric.HIGHEST_CHANGE_REGION,
                   RequestedMetric.REGIONAL_CHANGE_PERCENTAGE)
        for metric in metrics
    )
    if region != Region.FULL_IMAGE and asks_amount:
        needs_regions = True
    if intent == Intent.REGIONAL_COMPARISON or asks_comparison:
        needs_regions = True
    if needs_regions:
        tasks.append(QuerySubtask(action=QueryAction.ANALYZE_REGIONS))
    if intent == Intent.REGIONAL_COMPARISON or asks_comparison or asks_ranking:
        tasks.append(QuerySubtask(action=QueryAction.COMPARE_REGIONS))
    if asks_ranking:
        tasks.append(QuerySubtask(action=QueryAction.IDENTIFY_MAX_REGION))
    for metric in metrics:
        if metric not in (RequestedMetric.CHANGE_PRESENCE, RequestedMetric.REGIONAL_DISTRIBUTION,
                          RequestedMetric.HIGHEST_CHANGE_REGION):
            tasks.append(QuerySubtask(action=QueryAction.CALCULATE_METRIC, metric=metric))
    if intent == Intent.GENERAL_CHANGE:
        tasks.append(QuerySubtask(action=QueryAction.SUMMARIZE_CHANGE))

    constraints = [QueryConstraint.PIXEL_LEVEL_ONLY]
    if asks_location or needs_regions:
        constraints.append(QueryConstraint.IMAGE_REGION_ONLY)
    if operation == Operation.AREA:
        constraints.append(QueryConstraint.PIXEL_RESOLUTION_REQUIRED)

    parameters: dict[str, object] = {}
    if comparison_targets:
        parameters["comparison_regions"] = [item.value.lower() for item in comparison_targets]

    return StructuredQuery(
        raw_question=raw,
        intent=intent,
        target=target,
        region=region,
        operation=operation,
        requested_metrics=metrics,
        comparison_targets=comparison_targets,
        constraints=constraints,
        decomposition=tasks,
        parameters=parameters,
    )
