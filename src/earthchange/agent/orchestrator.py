"""End-to-end EarthChange AI analysis pipeline.

This is the single entry-point that connects:
  Question → NLP parsing → Planning → CV execution → Evidence → Answer

Usage::

    from earthchange.agent.orchestrator import analyze
    result = analyze(image_a, image_b, "How much of the image changed?")
    print(result.answer)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from earthchange.agent.executor import execute_plan
from earthchange.agent.planner import PlanningError, plan_for_query
from earthchange.evidence.package import EvidencePackage
from earthchange.llm.answer_generator import generate_answer


@dataclass(frozen=True)
class AnalysisResult:
    """Complete result of an EarthChange AI analysis."""

    answer: str
    evidence: EvidencePackage
    intent: str
    region: str
    operation: str
    target: str
    parameters: dict
    plan_steps: list[str]
    success: bool
    error: str | None = None


def analyze(
    image_a: NDArray[np.generic],
    image_b: NDArray[np.generic],
    question: str,
    *,
    pixel_resolution: tuple[float, float] | None = None,
    output_dir: Path | str | None = None,
    ground_truth: NDArray[np.bool_] | None = None,
) -> AnalysisResult:
    """Run the full EarthChange AI pipeline on an image pair and question.

    Parameters
    ----------
    image_a, image_b:
        Temporal image pair, ``(bands, rows, columns)`` order.
    question:
        Natural-language question about the images.
    pixel_resolution:
        Optional ``(x_m, y_m)`` ground sampling distance.
    output_dir:
        Directory for saving visualizations.
    ground_truth:
        Optional boolean change mask for evaluation overlays.

    Returns
    -------
    AnalysisResult
        Contains the grounded answer, full evidence, and metadata.
    """
    # Lazy import so the module is importable before NLP deps are installed
    from earthchange.nlp.parser import parse_question
    from earthchange.nlp.schema import Intent

    # Step 1 — Parse the question
    query = parse_question(question)

    # Step 2 — Handle unsupported questions without running CV
    if query.intent == Intent.UNSUPPORTED:
        evidence = EvidencePackage()
        evidence.metadata["unsupported_reason"] = query.parameters.get("reason")
        answer = generate_answer(evidence, query.intent.value, query.region.value)
        return AnalysisResult(
            answer=answer,
            evidence=evidence,
            intent=query.intent.value,
            region=query.region.value,
            operation=query.operation.value,
            target=query.target,
            parameters=query.parameters,
            plan_steps=[],
            success=True,
        )

    # Step 3 — Plan
    try:
        plan = plan_for_query(
            query,
        )
    except PlanningError as exc:
        evidence = EvidencePackage()
        return AnalysisResult(
            answer=str(exc),
            evidence=evidence,
            intent=query.intent.value,
            region=query.region.value,
            operation=query.operation.value,
            target=query.target,
            parameters=query.parameters,
            plan_steps=[],
            success=False,
            error=str(exc),
        )

    # Step 4 — Execute
    out_path = Path(output_dir) if output_dir else None
    try:
        evidence = execute_plan(
            plan,
            image_a,
            image_b,
            pixel_resolution=pixel_resolution,
            output_dir=out_path,
            ground_truth=ground_truth,
        )
    except Exception as exc:  # noqa: BLE001 - convert pipeline failures into a structured result
        evidence = getattr(exc, "evidence", None) or EvidencePackage()
        evidence.metadata["analysis_error"] = str(exc)
        return AnalysisResult(
            answer=f"Analysis could not be completed: {exc}",
            evidence=evidence,
            intent=query.intent.value,
            region=query.region.value,
            operation=query.operation.value,
            target=query.target,
            parameters=query.parameters,
            plan_steps=plan.tool_names(),
            success=False,
            error=str(exc),
        )

    # Step 5 — Generate grounded answer
    answer = generate_answer(evidence, query.intent.value, query.region.value, query=query)

    return AnalysisResult(
        answer=answer,
        evidence=evidence,
        intent=query.intent.value,
        region=query.region.value,
        operation=query.operation.value,
        target=query.target,
        parameters=query.parameters,
        plan_steps=plan.tool_names(),
        success=True,
    )
