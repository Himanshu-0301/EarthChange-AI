"""
Natural Language Processing module for EarthChange AI.

Responsible for intent classification, parameter extraction, and query structuring.
Must NOT perform visual analysis or generate measurements.
"""

from __future__ import annotations

from earthchange.nlp.parser import parse_question
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

__all__ = [
    "Intent",
    "Region",
    "Operation",
    "RequestedMetric",
    "QueryConstraint",
    "QueryAction",
    "QuerySubtask",
    "StructuredQuery",
    "parse_question",
]
