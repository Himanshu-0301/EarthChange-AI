"""Execute a validated plan by dispatching each step to the CV pipeline.

The executor connects the planner's ``ExecutionPlan`` to concrete CV functions
and accumulates results into an ``EvidencePackage``.  It never interprets
questions or generates natural-language answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from earthchange.agent.planner import (
    ExecutionPlan,
    PlanningError,
    ToolName,
    validate_execution_plan,
)
from earthchange.evidence.package import (
    ChangeStatistics,
    EvidencePackage,
    RegionStatistics,
)


class ExecutionError(RuntimeError):
    """Raised when plan execution fails irrecoverably."""

    def __init__(self, message: str, evidence: EvidencePackage | None = None) -> None:
        super().__init__(message)
        self.evidence = evidence


def execute_plan(
    plan: ExecutionPlan,
    image_a: NDArray[np.generic],
    image_b: NDArray[np.generic],
    *,
    pixel_resolution: tuple[float, float] | None = None,
    output_dir: Path | None = None,
    ground_truth: NDArray[np.bool_] | None = None,
) -> EvidencePackage:
    """Run every step in *plan* and return a populated evidence package.

    Parameters
    ----------
    plan:
        Validated execution plan from the planner.
    image_a, image_b:
        Temporal image pair in ``(bands, rows, columns)`` order.
    pixel_resolution:
        ``(x_m, y_m)`` per-pixel ground distance, if known.
    output_dir:
        Directory where visualizations will be saved.  If ``None``,
        visualization figures are created but not saved to disk.
    ground_truth:
        Optional boolean ground-truth mask for visualization overlay.
    """
    # Lazy imports keep the module importable without matplotlib installed
    from earthchange.cv.pipeline import (
        ChangeDetectionConfig,
        ChangeDetectionResult,
        InputValidationError,
        analyze_region,
        area_calculation,
        detect_changes,
        generate_visualization,
        validate_images,
    )

    evidence = EvidencePackage()
    try:
        validate_execution_plan(plan)
    except PlanningError as exc:
        evidence.record_tool("validate_plan", success=False, error_message=str(exc))
        raise ExecutionError(str(exc), evidence) from exc

    if pixel_resolution is not None:
        evidence.metadata["pixel_resolution_unit"] = "metres per pixel"
    evidence.metadata["image_a_shape"] = (
        list(image_a.shape) if isinstance(image_a, np.ndarray) else None
    )
    evidence.metadata["image_b_shape"] = (
        list(image_b.shape) if isinstance(image_b, np.ndarray) else None
    )

    change_result: ChangeDetectionResult | None = None

    for step in plan.steps:
        try:
            if step.tool == ToolName.VALIDATE_INPUT:
                validate_images(image_a, image_b)
                evidence.record_tool("validate_input", step.parameters)

            elif step.tool == ToolName.CHANGE_DETECTION:
                config = ChangeDetectionConfig()
                change_result = detect_changes(image_a, image_b, config)
                evidence.record_tool("change_detection", {
                    "method": config.method,
                    "threshold_method": config.threshold_method,
                })
                evidence.change_stats = ChangeStatistics(
                    changed_pixels=change_result.changed_pixels,
                    total_pixels=change_result.total_pixels,
                    changed_percentage=change_result.changed_percentage,
                )
                evidence.threshold_used = change_result.threshold_used
                evidence.detection_config = {
                    "method": config.method,
                    "threshold_method": config.threshold_method,
                    "threshold_value": change_result.threshold_used,
                }

            elif step.tool == ToolName.AREA_CALCULATION:
                if change_result is None:
                    raise ExecutionError("area_calculation requires prior change_detection")
                area_info = area_calculation(change_result, pixel_resolution)
                evidence.record_tool("area_calculation", {"pixel_resolution": pixel_resolution})
                # Update change stats with area info if resolution available
                evidence.change_stats = ChangeStatistics(
                    changed_pixels=area_info["changed_pixels"],
                    total_pixels=area_info["total_pixels"],
                    changed_percentage=area_info["changed_percentage"],
                    changed_area_m2=area_info.get("changed_area_m2"),
                    pixel_resolution_m=area_info.get("pixel_resolution_m"),
                )

            elif step.tool == ToolName.REGION_ANALYSIS:
                if change_result is None:
                    raise ExecutionError("region_analysis requires prior change_detection")
                region_name = step.parameters.get("region", "full_image").lower()
                requested_regions = step.parameters.get("regions")
                from earthchange.cv.pipeline import REGION_SLICES
                if requested_regions:
                    for rname in requested_regions:
                        rname = rname.lower()
                        rs = analyze_region(change_result.binary_mask, rname)
                        evidence.region_stats[rname] = RegionStatistics(
                            region_name=rs.region_name,
                            changed_pixels=rs.changed_pixels,
                            total_pixels=rs.total_pixels,
                            changed_percentage=rs.changed_percentage,
                        )
                elif region_name == "full_image" or plan.intent in (
                    "CHANGE_LOCALIZATION", "MULTI_STEP_ANALYSIS", "REGIONAL_COMPARISON"
                ):
                    region_names = list(REGION_SLICES) if region_name == "full_image" else [region_name]
                    for rname in region_names:
                        rs = analyze_region(change_result.binary_mask, rname)
                        evidence.region_stats[rname] = RegionStatistics(
                            region_name=rs.region_name,
                            changed_pixels=rs.changed_pixels,
                            total_pixels=rs.total_pixels,
                            changed_percentage=rs.changed_percentage,
                        )
                else:
                    rs = analyze_region(change_result.binary_mask, region_name)
                    evidence.region_stats[region_name] = RegionStatistics(
                        region_name=rs.region_name,
                        changed_pixels=rs.changed_pixels,
                        total_pixels=rs.total_pixels,
                        changed_percentage=rs.changed_percentage,
                    )
                evidence.record_tool("region_analysis", step.parameters or {"region": region_name})

            elif step.tool == ToolName.VISUALIZATION:
                if change_result is None:
                    raise ExecutionError("visualization requires prior change_detection")
                fig = generate_visualization(
                    image_a, image_b, change_result, ground_truth=ground_truth
                )
                if output_dir is not None:
                    viz_path = Path(output_dir) / "change_analysis.png"
                    viz_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(str(viz_path), dpi=150, bbox_inches="tight")
                    evidence.visualization_paths["change_analysis"] = str(viz_path)
                # Always close the figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
                evidence.record_tool("generate_visual_evidence")

        except Exception as exc:
            evidence.record_tool(
                step.tool.value,
                step.parameters,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )
            if isinstance(exc, ExecutionError):
                raise ExecutionError(str(exc), evidence) from exc
            if isinstance(exc, InputValidationError):
                raise ExecutionError(str(exc), evidence) from exc
            raise ExecutionError(f"Tool {step.tool.value} failed: {exc}", evidence) from exc

    return evidence
