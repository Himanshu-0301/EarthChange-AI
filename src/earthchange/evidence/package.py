"""Structured evidence container produced by the CV tool pipeline.

An ``EvidencePackage`` carries every measurable fact the pipeline discovered.
The answer-generation layer must base its response exclusively on the contents
of this package — it must never fabricate measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExecution:
    """Record of a single tool invocation."""

    tool_name: str
    parameters: dict[str, Any]
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class ChangeStatistics:
    """Pixel-level change measurements."""

    changed_pixels: int
    total_pixels: int
    changed_percentage: float
    changed_area_m2: float | None = None
    pixel_resolution_m: tuple[float, float] | None = None


@dataclass(frozen=True)
class RegionStatistics:
    """Change measurements for a named spatial region."""

    region_name: str
    changed_pixels: int
    total_pixels: int
    changed_percentage: float


@dataclass
class EvidencePackage:
    """All structured evidence produced by an analysis run.

    Fields default to empty/None so the package can be built incrementally
    as each tool executes.
    """

    # What tools ran and with what parameters
    tools_executed: list[ToolExecution] = field(default_factory=list)

    # Quantitative change results
    change_stats: ChangeStatistics | None = None

    # Per-region breakdowns
    region_stats: dict[str, RegionStatistics] = field(default_factory=dict)

    # Paths to generated visualizations
    visualization_paths: dict[str, str] = field(default_factory=dict)

    # Change-detection configuration used
    detection_config: dict[str, Any] = field(default_factory=dict)

    # Free-form metadata (image shapes, dates, band info, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # The threshold applied for binary change masking
    threshold_used: float | None = None

    def record_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        *,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Append a tool execution record."""
        self.tools_executed.append(
            ToolExecution(
                tool_name=tool_name,
                parameters=parameters or {},
                success=success,
                error_message=error_message,
            )
        )

    def summary_dict(self) -> dict[str, Any]:
        """Flat dictionary for answer generation and serialization."""
        result: dict[str, Any] = {
            "tools_executed": [t.tool_name for t in self.tools_executed if t.success],
            "tools_failed": [t.tool_name for t in self.tools_executed if not t.success],
        }
        if self.change_stats:
            result["changed_pixels"] = self.change_stats.changed_pixels
            result["total_pixels"] = self.change_stats.total_pixels
            result["changed_percentage"] = self.change_stats.changed_percentage
            if self.change_stats.changed_area_m2 is not None:
                result["changed_area_m2"] = self.change_stats.changed_area_m2
            if self.change_stats.pixel_resolution_m is not None:
                result["pixel_resolution_m"] = self.change_stats.pixel_resolution_m
        if self.region_stats:
            result["regions"] = {
                name: {
                    "changed_pixels": rs.changed_pixels,
                    "total_pixels": rs.total_pixels,
                    "changed_percentage": rs.changed_percentage,
                }
                for name, rs in self.region_stats.items()
            }
        if self.threshold_used is not None:
            result["threshold_used"] = self.threshold_used
        result["visualization_paths"] = self.visualization_paths
        result["metadata"] = self.metadata
        return result
