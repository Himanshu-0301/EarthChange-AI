"""Tool registry and execution planner for EarthChange AI.

The registry holds a fixed set of CV tools with explicit input/output contracts.
The planner maps a ``StructuredQuery`` to an ordered execution plan of registered
tools.  It never generates arbitrary code — it only selects and sequences tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from earthchange.nlp.schema import (
    Intent,
    Operation,
    QueryAction,
    Region,
    RequestedMetric,
    StructuredQuery,
)


class PlanningError(ValueError):
    """Raised when a valid execution plan cannot be constructed."""


class ToolName(str, Enum):
    """Identifiers for all registered CV tools."""

    VALIDATE_INPUT = "validate_input"
    CHANGE_DETECTION = "change_detection"
    REGION_ANALYSIS = "region_analysis"
    AREA_CALCULATION = "area_calculation"
    VISUALIZATION = "generate_visual_evidence"


@dataclass(frozen=True)
class ToolSpec:
    """Specification of a registered tool."""

    name: ToolName
    description: str
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class PlannedStep:
    """A single step in an execution plan."""

    tool: ToolName
    parameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated sequence of tool invocations for a query."""

    steps: tuple[PlannedStep, ...]
    intent: str
    region: str
    operation: str
    structured_query: StructuredQuery | None = None

    def tool_names(self) -> list[str]:
        """Return the ordered list of tool names in this plan."""
        return [step.tool.value for step in self.steps]


def validate_execution_plan(plan: ExecutionPlan) -> None:
    """Check tool registration, input availability, and step parameters.

    This keeps the planner/executor boundary explicit: a plan can only call
    registered tools, consume values produced earlier in the plan, and pass
    the small set of parameters each tool accepts.
    """
    if not isinstance(plan, ExecutionPlan) or not isinstance(plan.steps, tuple):
        raise PlanningError("Execution plan must be an ExecutionPlan")
    if not plan.steps:
        raise PlanningError("Execution plan must contain at least one tool")
    available = {"image_a", "image_b"}
    for index, step in enumerate(plan.steps):
        if (
            not isinstance(step, PlannedStep)
            or not isinstance(step.tool, ToolName)
            or step.tool not in TOOL_REGISTRY
        ):
            raise PlanningError(f"Plan step {index} uses an unregistered tool")
        if not isinstance(step.parameters, dict) or not all(
            isinstance(parameter, str) for parameter in step.parameters
        ):
            raise PlanningError(f"Plan step {index} parameters must be a string-keyed mapping")
        spec = TOOL_REGISTRY[step.tool]
        missing = set(spec.required_inputs) - available
        if missing:
            raise PlanningError(
                f"Tool {step.tool.value} is missing required inputs: {', '.join(sorted(missing))}"
            )
        allowed_parameters = {"region", "regions"} if step.tool == ToolName.REGION_ANALYSIS else set()
        unexpected = set(step.parameters) - allowed_parameters
        if unexpected:
            raise PlanningError(
                f"Tool {step.tool.value} does not accept parameters: "
                f"{', '.join(sorted(unexpected))}"
            )
        if step.tool == ToolName.REGION_ANALYSIS:
            region = step.parameters.get("region", "full_image")
            valid_regions = {name.upper() for name in (
                "full_image", "top", "bottom", "left", "right", "upper_left",
                "upper_right", "lower_left", "lower_right", "center"
            )}
            if not isinstance(region, str) or region.upper() not in valid_regions:
                raise PlanningError(f"Unsupported region parameter: {region!r}")
            regions = step.parameters.get("regions")
            if regions is not None and (
                not isinstance(regions, list)
                or not regions
                or any(not isinstance(item, str) or item.upper() not in valid_regions for item in regions)
            ):
                raise PlanningError("regions must be a non-empty list of supported region names")
        available.update(spec.outputs)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[ToolName, ToolSpec] = {
    ToolName.VALIDATE_INPUT: ToolSpec(
        name=ToolName.VALIDATE_INPUT,
        description="Validate that input images are compatible for change detection",
        required_inputs=("image_a", "image_b"),
        outputs=("validated",),
    ),
    ToolName.CHANGE_DETECTION: ToolSpec(
        name=ToolName.CHANGE_DETECTION,
        description="Detect changes between temporal image pair",
        required_inputs=("image_a", "image_b", "validated"),
        outputs=("change_result",),
    ),
    ToolName.REGION_ANALYSIS: ToolSpec(
        name=ToolName.REGION_ANALYSIS,
        description="Analyze change statistics for specific image regions",
        required_inputs=("change_result",),
        outputs=("region_stats",),
    ),
    ToolName.AREA_CALCULATION: ToolSpec(
        name=ToolName.AREA_CALCULATION,
        description="Calculate changed area statistics (pixels, percentage, physical area)",
        required_inputs=("change_result",),
        outputs=("area_stats",),
    ),
    ToolName.VISUALIZATION: ToolSpec(
        name=ToolName.VISUALIZATION,
        description="Generate visual evidence (change maps, overlays)",
        required_inputs=("image_a", "image_b", "change_result"),
        outputs=("figure",),
    ),
}


# ---------------------------------------------------------------------------
# Intent-to-plan mapping
# ---------------------------------------------------------------------------

def plan_for_query(
    query: StructuredQuery | str,
    region: str | None = None,
    operation: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> ExecutionPlan:
    """Plan registered tools from a structured query.

    The ``str`` form remains as a compatibility adapter for existing callers;
    the application orchestrator passes a validated ``StructuredQuery``.
    """
    if isinstance(query, StructuredQuery):
        structured = query
    elif isinstance(query, str):
        try:
            structured = StructuredQuery(
                raw_question="legacy planner call",
                intent=Intent(query),
                region=Region(region or "full_image"),
                operation=Operation(operation or "describe"),
                parameters=parameters or {},
                comparison_targets=[
                    Region(item) for item in (parameters or {}).get("comparison_regions", [])
                ],
            )
        except (TypeError, ValueError) as exc:
            raise PlanningError(f"Invalid structured query: {exc}") from exc
    else:
        raise PlanningError("Planner requires a StructuredQuery")

    intent = structured.intent
    if intent == Intent.UNSUPPORTED:
        raise PlanningError(f"Cannot plan execution for unsupported intent: {intent.value}")

    steps: list[PlannedStep] = [
        PlannedStep(ToolName.VALIDATE_INPUT, description="Validate image compatibility"),
        PlannedStep(ToolName.CHANGE_DETECTION, description="Measure spectral pixel change"),
    ]
    actions = {task.action for task in structured.decomposition}
    metrics = set(structured.requested_metrics)
    needs_regions = bool(actions & {
        QueryAction.ANALYZE_REGIONS,
        QueryAction.COMPARE_REGIONS,
        QueryAction.IDENTIFY_MAX_REGION,
    }) or intent in (Intent.CHANGE_LOCALIZATION, Intent.REGIONAL_COMPARISON, Intent.MULTI_STEP_ANALYSIS)
    needs_area = bool(metrics & {
        RequestedMetric.CHANGED_PIXEL_COUNT,
        RequestedMetric.CHANGED_PERCENTAGE,
        RequestedMetric.CHANGED_AREA,
    }) or intent in (Intent.CHANGE_QUANTIFICATION, Intent.REGIONAL_COMPARISON, Intent.MULTI_STEP_ANALYSIS)

    if needs_regions:
        if len(structured.comparison_targets) > 1:
            region_parameters: dict[str, Any] = {
                "regions": [item.value.lower() for item in structured.comparison_targets]
            }
        else:
            region_parameters = {"region": structured.region.value.lower()}
        steps.append(PlannedStep(
            ToolName.REGION_ANALYSIS,
            parameters=region_parameters,
            description="Analyze requested image regions",
        ))
    if needs_area:
        steps.append(PlannedStep(
            ToolName.AREA_CALCULATION,
            description="Calculate requested pixel and area statistics",
        ))
    steps.append(PlannedStep(ToolName.VISUALIZATION, description="Generate visual evidence"))

    return ExecutionPlan(
        steps=tuple(steps),
        intent=intent.value,
        region=structured.region.value.lower(),
        operation=structured.operation.value.lower(),
        structured_query=structured,
    )
