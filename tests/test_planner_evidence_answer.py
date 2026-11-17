"""Tests for the tool registry, planner, and evidence package."""

import numpy as np
import pytest

from earthchange.agent.planner import (
    ExecutionPlan,
    PlanningError,
    ToolName,
    plan_for_query,
)
from earthchange.evidence.package import (
    ChangeStatistics,
    EvidencePackage,
    RegionStatistics,
    ToolExecution,
)
from earthchange.llm.answer_generator import generate_answer


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_general_change_includes_validation_detection_visualization(self) -> None:
        plan = plan_for_query("GENERAL_CHANGE", "full_image", "describe")
        names = plan.tool_names()
        assert "validate_input" in names
        assert "change_detection" in names
        assert "generate_visual_evidence" in names

    def test_quantification_includes_area_calculation(self) -> None:
        plan = plan_for_query("CHANGE_QUANTIFICATION", "full_image", "percentage")
        names = plan.tool_names()
        assert "area_calculation" in names

    def test_localization_includes_region_analysis(self) -> None:
        plan = plan_for_query("CHANGE_LOCALIZATION", "full_image", "locate")
        names = plan.tool_names()
        assert "region_analysis" in names

    def test_comparison_includes_region_and_area(self) -> None:
        plan = plan_for_query("REGIONAL_COMPARISON", "full_image", "compare")
        names = plan.tool_names()
        assert "region_analysis" in names
        assert "area_calculation" in names

    def test_multi_step_includes_all_tools(self) -> None:
        plan = plan_for_query("MULTI_STEP_ANALYSIS", "full_image", "describe")
        names = plan.tool_names()
        assert "area_calculation" in names
        assert "region_analysis" in names

    def test_unsupported_raises_planning_error(self) -> None:
        with pytest.raises(PlanningError, match="unsupported"):
            plan_for_query("UNSUPPORTED", "full_image", "describe")

    def test_plan_has_correct_metadata(self) -> None:
        plan = plan_for_query("GENERAL_CHANGE", "center", "describe")
        assert plan.intent == "GENERAL_CHANGE"
        assert plan.region == "center"
        assert plan.operation == "describe"


# ---------------------------------------------------------------------------
# Evidence package tests
# ---------------------------------------------------------------------------

class TestEvidencePackage:
    def test_record_tool_appends_execution(self) -> None:
        ep = EvidencePackage()
        ep.record_tool("change_detection", {"method": "cva"})
        assert len(ep.tools_executed) == 1
        assert ep.tools_executed[0].tool_name == "change_detection"
        assert ep.tools_executed[0].success is True

    def test_record_failed_tool(self) -> None:
        ep = EvidencePackage()
        ep.record_tool("validate_input", success=False, error_message="bad shape")
        assert ep.tools_executed[0].success is False
        assert "bad shape" in ep.tools_executed[0].error_message

    def test_summary_dict_includes_stats(self) -> None:
        ep = EvidencePackage()
        ep.change_stats = ChangeStatistics(
            changed_pixels=100,
            total_pixels=1000,
            changed_percentage=10.0,
        )
        ep.record_tool("change_detection")
        summary = ep.summary_dict()
        assert summary["changed_pixels"] == 100
        assert summary["changed_percentage"] == 10.0
        assert "change_detection" in summary["tools_executed"]

    def test_summary_dict_includes_area_when_available(self) -> None:
        ep = EvidencePackage()
        ep.change_stats = ChangeStatistics(
            changed_pixels=100,
            total_pixels=1000,
            changed_percentage=10.0,
            changed_area_m2=10000.0,
            pixel_resolution_m=(10.0, 10.0),
        )
        summary = ep.summary_dict()
        assert summary["changed_area_m2"] == 10000.0

    def test_summary_dict_with_regions(self) -> None:
        ep = EvidencePackage()
        ep.change_stats = ChangeStatistics(changed_pixels=50, total_pixels=500, changed_percentage=10.0)
        ep.region_stats["top"] = RegionStatistics(
            region_name="top", changed_pixels=30, total_pixels=250, changed_percentage=12.0
        )
        summary = ep.summary_dict()
        assert "top" in summary["regions"]
        assert summary["regions"]["top"]["changed_percentage"] == 12.0


# ---------------------------------------------------------------------------
# Answer generator tests
# ---------------------------------------------------------------------------

class TestAnswerGenerator:
    def _make_evidence(self, changed: int = 100, total: int = 1000) -> EvidencePackage:
        ep = EvidencePackage()
        ep.change_stats = ChangeStatistics(
            changed_pixels=changed,
            total_pixels=total,
            changed_percentage=round(changed / total * 100, 2),
        )
        ep.record_tool("change_detection")
        ep.threshold_used = 42.5
        return ep

    def test_general_change_mentions_percentage(self) -> None:
        answer = generate_answer(self._make_evidence(), "GENERAL_CHANGE", "full_image")
        assert "10.0%" in answer
        assert "100" in answer

    def test_quantification_reports_numbers(self) -> None:
        answer = generate_answer(self._make_evidence(), "CHANGE_QUANTIFICATION", "full_image")
        assert "10.00%" in answer
        assert "1,000" in answer

    def test_no_change_detected(self) -> None:
        answer = generate_answer(self._make_evidence(0, 1000), "GENERAL_CHANGE", "full_image")
        assert "no pixels were flagged" in answer.lower()

    def test_unsupported_gives_scope_message(self) -> None:
        ep = EvidencePackage()
        answer = generate_answer(ep, "UNSUPPORTED", "full_image")
        assert "outside the scope" in answer.lower()

    def test_failed_tools_reported(self) -> None:
        ep = EvidencePackage()
        ep.record_tool("validate_input", success=False, error_message="images differ in shape")
        answer = generate_answer(ep, "GENERAL_CHANGE", "full_image")
        assert "error" in answer.lower()
        assert "validate_input" in answer

    def test_localization_with_regions(self) -> None:
        ep = self._make_evidence(100, 1000)
        ep.region_stats["top"] = RegionStatistics("top", 80, 500, 16.0)
        ep.region_stats["bottom"] = RegionStatistics("bottom", 20, 500, 4.0)
        answer = generate_answer(ep, "CHANGE_LOCALIZATION", "full_image")
        assert "top" in answer.lower()
        assert "16.0%" in answer

    def test_comparison_ranks_regions(self) -> None:
        ep = self._make_evidence(100, 1000)
        ep.region_stats["left"] = RegionStatistics("left", 80, 500, 16.0)
        ep.region_stats["right"] = RegionStatistics("right", 20, 500, 4.0)
        answer = generate_answer(ep, "REGIONAL_COMPARISON", "full_image")
        assert "most" in answer.lower()
        assert "least" in answer.lower()

    def test_quantification_without_resolution_notes_unavailable(self) -> None:
        answer = generate_answer(self._make_evidence(), "CHANGE_QUANTIFICATION", "full_image")
        assert "not available" in answer.lower() or "could not" in answer.lower()

    def test_quantification_with_resolution_reports_area(self) -> None:
        ep = EvidencePackage()
        ep.change_stats = ChangeStatistics(
            changed_pixels=100,
            total_pixels=1000,
            changed_percentage=10.0,
            changed_area_m2=10000.0,
            pixel_resolution_m=(10.0, 10.0),
        )
        ep.record_tool("change_detection")
        ep.threshold_used = 42.5
        answer = generate_answer(ep, "CHANGE_QUANTIFICATION", "full_image")
        assert "10,000" in answer
        assert "10.0m" in answer
