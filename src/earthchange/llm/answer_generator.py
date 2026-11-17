"""Generate natural-language answers from structured evidence.

The answer generator must ONLY use facts present in the ``EvidencePackage``.
It must never invent pixel counts, percentages, areas, or other quantitative
measurements.  For the MVP, answers are generated with deterministic templates
rather than an LLM.
"""

from __future__ import annotations

from typing import Any

from earthchange.evidence.package import EvidencePackage
from earthchange.nlp.schema import QueryAction, RequestedMetric, StructuredQuery


def generate_answer(
    evidence: EvidencePackage,
    intent: str,
    region: str,
    *,
    query: StructuredQuery | None = None,
) -> str:
    """Produce a grounded natural-language answer from evidence.

    Parameters
    ----------
    evidence:
        Populated evidence package from the executor.
    intent:
        The classified intent (e.g. ``"CHANGE_QUANTIFICATION"``).
    region:
        The target region (e.g. ``"full_image"``).

    Returns
    -------
    str
        Human-readable answer grounded exclusively in the evidence.
    """
    failed_tools = [t for t in evidence.tools_executed if not t.success]
    if failed_tools:
        failures = ", ".join(t.tool_name for t in failed_tools)
        return (
            f"The analysis could not be completed. "
            f"The following tools encountered errors: {failures}. "
            f"Error: {failed_tools[0].error_message}"
        )

    intent_upper = intent.upper() if isinstance(intent, str) else str(intent).upper()

    if "UNSUPPORTED" in intent_upper:
        if evidence.metadata.get("unsupported_reason") == "semantic_target_not_supported":
            return (
                "EarthChange AI currently measures pixel-level spectral differences; "
                "it does not identify land-cover classes such as forest, water, or buildings. "
                "Ask about overall change, its image regions, or its pixel/area amount instead."
            )
        return (
            "This question is outside the scope of EarthChange AI. "
            "I can answer questions about visible changes between satellite "
            "image pairs, such as 'How much of the image changed?' or "
            "'Where did changes occur?'"
        )

    if evidence.change_stats is None:
        return "No change detection results are available."

    stats = evidence.change_stats

    if "QUANTIFICATION" in intent_upper:
        return _quantification_answer(stats, region, evidence)

    if "LOCALIZATION" in intent_upper:
        return _localization_answer(stats, evidence)

    if "COMPARISON" in intent_upper:
        return _comparison_answer(evidence)

    if "MULTI_STEP" in intent_upper:
        if query and RequestedMetric.HIGHEST_CHANGE_REGION in query.requested_metrics:
            ranked = sorted(
                evidence.region_stats.values(),
                key=lambda item: item.changed_percentage,
                reverse=True,
            )
            if not ranked:
                return "The requested region ranking is unavailable because no regional measurements were produced."
            highest = ranked[0]
            if QueryAction.IDENTIFY_MAX_REGION in {task.action for task in query.decomposition}:
                return (
                    f"The region with the most detected pixel change is "
                    f"{highest.region_name.replace('_', ' ')}, where "
                    f"{highest.changed_percentage:.2f}% of pixels were flagged."
                )
        # Combine quantification and localization
        parts = [
            _quantification_answer(stats, region, evidence),
            "",
            _localization_answer(stats, evidence),
        ]
        return "\n".join(parts)

    # GENERAL_CHANGE — default
    return _general_answer(stats, evidence)


def _general_answer(stats: Any, evidence: EvidencePackage) -> str:
    """Answer for GENERAL_CHANGE intent."""
    if stats.changed_pixels == 0:
        return "No pixels were flagged as changed by the current threshold."

    answer = (
        f"Changes were detected between the two images. "
        f"Approximately {stats.changed_percentage:.1f}% of the image area "
        f"shows change ({stats.changed_pixels:,} out of "
        f"{stats.total_pixels:,} pixels)."
    )
    if stats.changed_area_m2 is not None:
        answer += f" This corresponds to approximately {stats.changed_area_m2:,.0f} m²."

    tools = [t.tool_name for t in evidence.tools_executed if t.success]
    answer += f"\n\nAnalysis tools used: {', '.join(tools)}."
    return answer


def _quantification_answer(stats: Any, region: str, evidence: EvidencePackage) -> str:
    """Answer for CHANGE_QUANTIFICATION intent."""
    region_label = region.replace("_", " ")

    if stats.changed_pixels == 0:
        return f"No pixels were flagged as changed in the {region_label}."

    answer = (
        f"In the {region_label}, {stats.changed_percentage:.2f}% of pixels "
        f"show change ({stats.changed_pixels:,} changed out of "
        f"{stats.total_pixels:,} total pixels)."
    )

    if stats.changed_area_m2 is not None and stats.pixel_resolution_m is not None:
        res = stats.pixel_resolution_m
        answer += (
            f"\n\nAt {res[0]:.1f}m × {res[1]:.1f}m per pixel, "
            f"the changed area is approximately {stats.changed_area_m2:,.0f} m²."
        )
    else:
        answer += (
            "\n\nPhysical area could not be calculated because reliable "
            "spatial resolution metadata is not available."
        )

    if evidence.threshold_used is not None:
        answer += (
            f"\n\nChange detection threshold: {evidence.threshold_used:.4f} "
            f"(Otsu automatic thresholding)."
        )

    return answer


def _localization_answer(stats: Any, evidence: EvidencePackage) -> str:
    """Answer for CHANGE_LOCALIZATION intent."""
    if stats.changed_pixels == 0:
        return "No pixels were flagged as changed, so there is no detected region to summarize."

    if not evidence.region_stats:
        return (
            f"Changes were detected ({stats.changed_percentage:.1f}% of the image). "
            f"Regional breakdown is not available for this analysis."
        )

    # Sort regions by change percentage, descending
    sorted_regions = sorted(
        evidence.region_stats.values(),
        key=lambda r: r.changed_percentage,
        reverse=True,
    )

    # Report top changed regions
    top_regions = [r for r in sorted_regions if r.changed_percentage > 0]
    if not top_regions:
        return "No pixels were flagged as changed in any analyzed region."

    lines = ["Changes are concentrated in the following areas:\n"]
    for r in top_regions[:5]:  # Top 5
        label = r.region_name.replace("_", " ")
        lines.append(
            f"- **{label}**: {r.changed_percentage:.1f}% changed "
            f"({r.changed_pixels:,} pixels)"
        )

    no_change = [r for r in sorted_regions if r.changed_percentage == 0.0]
    if no_change:
        names = ", ".join(r.region_name.replace("_", " ") for r in no_change)
        lines.append(f"\nNo changes detected in: {names}.")

    return "\n".join(lines)


def _comparison_answer(evidence: EvidencePackage) -> str:
    """Answer for REGIONAL_COMPARISON intent."""
    if not evidence.region_stats:
        return "Regional comparison data is not available."

    sorted_regions = sorted(
        evidence.region_stats.values(),
        key=lambda r: r.changed_percentage,
        reverse=True,
    )

    lines = ["Regional comparison of change intensity:\n"]
    for r in sorted_regions:
        label = r.region_name.replace("_", " ")
        lines.append(
            f"- **{label}**: {r.changed_percentage:.1f}% "
            f"({r.changed_pixels:,}/{r.total_pixels:,} pixels)"
        )

    most = sorted_regions[0]
    least = sorted_regions[-1]
    lines.append(
        f"\nThe **{most.region_name.replace('_', ' ')}** region shows the most "
        f"change ({most.changed_percentage:.1f}%), while the "
        f"**{least.region_name.replace('_', ' ')}** region shows the least "
        f"({least.changed_percentage:.1f}%)."
    )

    return "\n".join(lines)
