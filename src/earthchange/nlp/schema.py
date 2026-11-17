"""Validated semantic representation of a satellite-change question."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _CaseInsensitiveEnum(str, Enum):
    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.strip().lower():
                    return member
        return None


class Intent(_CaseInsensitiveEnum):
    GENERAL_CHANGE = "GENERAL_CHANGE"
    CHANGE_LOCALIZATION = "CHANGE_LOCALIZATION"
    CHANGE_QUANTIFICATION = "CHANGE_QUANTIFICATION"
    REGIONAL_COMPARISON = "REGIONAL_COMPARISON"
    MULTI_STEP_ANALYSIS = "MULTI_STEP_ANALYSIS"
    UNSUPPORTED = "UNSUPPORTED"


class Region(_CaseInsensitiveEnum):
    FULL_IMAGE = "FULL_IMAGE"
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UPPER_LEFT = "UPPER_LEFT"
    UPPER_RIGHT = "UPPER_RIGHT"
    LOWER_LEFT = "LOWER_LEFT"
    LOWER_RIGHT = "LOWER_RIGHT"
    CENTER = "CENTER"


class Operation(_CaseInsensitiveEnum):
    DESCRIBE = "DESCRIBE"
    LOCATE = "LOCATE"
    COUNT = "COUNT"
    AREA = "AREA"
    PERCENTAGE = "PERCENTAGE"
    COMPARE = "COMPARE"


class RequestedMetric(_CaseInsensitiveEnum):
    CHANGE_PRESENCE = "change_presence"
    CHANGED_PIXEL_COUNT = "changed_pixel_count"
    CHANGED_PERCENTAGE = "changed_percentage"
    CHANGED_AREA = "changed_area"
    REGIONAL_CHANGE_PERCENTAGE = "regional_change_percentage"
    HIGHEST_CHANGE_REGION = "highest_change_region"
    REGIONAL_DISTRIBUTION = "regional_distribution"
    SEMANTIC_CLASS_CHANGE = "semantic_class_change"
    SEMANTIC_AREA = "semantic_area"
    OBJECT_COUNT = "object_count"


class QueryConstraint(_CaseInsensitiveEnum):
    PIXEL_LEVEL_ONLY = "pixel_level_only"
    SEMANTIC_CLASS_UNAVAILABLE = "semantic_class_unavailable"
    OBJECT_COUNT_UNSUPPORTED = "object_count_unsupported"
    PIXEL_RESOLUTION_REQUIRED = "pixel_resolution_required"
    IMAGE_REGION_ONLY = "image_region_only"
    AMBIGUOUS_REQUEST = "ambiguous_request"


class QueryAction(_CaseInsensitiveEnum):
    DETECT_CHANGE = "detect_change"
    SUMMARIZE_CHANGE = "summarize_change"
    ANALYZE_REGIONS = "analyze_regions"
    COMPARE_REGIONS = "compare_regions"
    IDENTIFY_MAX_REGION = "identify_max_region"
    CALCULATE_METRIC = "calculate_metric"


class QuerySubtask(BaseModel):
    """One ordered, auditable step extracted from a supported request."""

    model_config = ConfigDict(extra="forbid")

    action: QueryAction
    metric: RequestedMetric | None = None


class StructuredQuery(BaseModel):
    """Typed parser output consumed by planning and evaluation.

    ``confidence`` is retained as an optional compatibility field, but the
    current rule parser does not emit a confidence estimate.
    """

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    raw_question: str = Field(max_length=1000)
    intent: Intent
    target: str = "change"
    region: Region = Region.FULL_IMAGE
    operation: Operation = Operation.DESCRIBE
    temporal_mode: str = "before_after"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    requested_metrics: list[RequestedMetric] = Field(default_factory=list)
    comparison_targets: list[Region] = Field(default_factory=list)
    constraints: list[QueryConstraint] = Field(default_factory=list)
    decomposition: list[QuerySubtask] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
