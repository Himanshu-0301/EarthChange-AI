# AGENTS.md — EarthChange AI Project Rules

This file defines the rules, constraints, and design principles that all development
agents (human or AI) must follow when working on this project.

---

## Core Principle

> Vision detects. Language understands and reasons. The agent decides what evidence is needed.

## Research Question

> Can tool-augmented agentic reasoning improve grounded question answering for
> multi-temporal satellite imagery compared with a pure VLM and a fixed CV pipeline?

This is a **hypothesis to test**, not an assumed conclusion.

---

## Non-Negotiable Design Principles

1. **Do not redesign the project** unless there is a genuine technical problem.

2. **Do not introduce technologies merely because they sound advanced.**

3. **Separate maturity levels clearly:**
   - Proposed — designed but not coded
   - Implemented — coded but not tested
   - Validated — tested with unit/integration tests
   - Experimentally validated — supported by quantitative experiments

4. **CV performs visual measurement.** Computer vision tools detect, segment, and
   measure. They do not interpret user questions or generate natural language.

5. **NLP interprets natural language.** NLP components classify intent, extract
   parameters, and structure queries. They do not perform visual analysis.

6. **The agent selects and sequences tools.** The orchestration layer plans which
   tools to call and in what order. It does not perform CV or NLP tasks directly.

7. **The language model must not invent quantitative visual measurements.** All
   quantitative claims in answers must originate from CV tool outputs.

8. **LLM confidence/token probabilities must never be presented as actual evidence
   confidence.** Confidence scores must come from the CV or evaluation pipeline.

9. **Prefer simple, interpretable, reproducible implementations.**

10. **No unnecessary complexity:**
    - No microservices unless genuinely needed
    - No vector databases unless genuinely needed
    - No agent frameworks (LangChain, LangGraph, CrewAI, AutoGen) unless a later
      architectural decision explicitly requires one
    - No RAG unless genuinely needed

11. **Do not train large foundation models.** Use pretrained models and fine-tune
    only if necessary and feasible within time constraints.

12. **All important research claims must eventually be supported by experiments.**

13. **Time budget:** The project must be feasible within approximately 6–7 hours/week
    over 4–6 weeks total (~24–42 hours).

14. **Optional features must never delay the core MVP.**

---

## Responsibility Boundaries

| Module | Responsibility | Must NOT Do |
|--------|---------------|-------------|
| `earthchange.cv` | Visual measurement (change detection, masking, area) | Interpret questions, generate text |
| `earthchange.nlp` | Intent classification, parameter extraction, query structuring | Perform visual analysis |
| `earthchange.agent` | Tool selection, sequencing, plan execution | Perform CV or NLP directly |
| `earthchange.evidence` | Package and structure tool outputs | Interpret or generate answers |
| `earthchange.llm` | Answer generation from structured evidence | Invent quantitative facts |
| `earthchange.evaluation` | Metrics, benchmarking, experiment tracking | — |

---

## Tool Architecture Rules

- Tools must have **explicit input/output contracts** (typed signatures, documented).
- The agent must select from **registered tools** — it must not generate arbitrary
  code for execution.
- Each tool should do **one focused thing** well.

### Planned Tool Registry

| Tool | Purpose |
|------|---------|
| `validate_input` | Validate image inputs |
| `preprocess_images` | Normalize, align, prepare images |
| `change_detection` | Detect changes between image pairs |
| `clean_change_mask` | Morphological cleanup of change masks |
| `region_analysis` | Analyze specific image regions |
| `area_calculation` | Calculate changed area statistics |
| `generate_visual_evidence` | Produce visualizations |

---

## NLP Architecture Rules

- Natural-language questions must be parsed into a **StructuredQuery**.
- The pipeline is: `Question → Intent → Parameters → StructuredQuery`

### Supported Intents

- `GENERAL_CHANGE` — Is there change?
- `CHANGE_LOCALIZATION` — Where is the change?
- `CHANGE_QUANTIFICATION` — How much change?
- `REGIONAL_COMPARISON` — Compare regions
- `MULTI_STEP_ANALYSIS` — Compound questions
- `UNSUPPORTED` — Out of scope

### Region Vocabulary

`full_image`, `top`, `bottom`, `left`, `right`, `upper_left`, `upper_right`,
`lower_left`, `lower_right`, `center`

### Operations

`describe`, `locate`, `count`, `area`, `percentage`, `compare`

---

## Evidence Architecture Rules

- The pipeline must produce an **EvidencePackage** containing:
  - Tools executed and their parameters
  - Change masks
  - Changed-pixel counts
  - Area measurements
  - Regional statistics
  - Quantitative results
  - Visualization paths
  - Relevant metadata

- The answer-generation layer must use this evidence rather than independently
  calculating or hallucinating visual facts.

---

## Dataset Rules

- **Primary benchmark:** OSCD (Onera Satellite Change Detection Dataset)
- Use OSCD for binary change-detection evaluation.
- Do **not** assume OSCD supports semantic questions (e.g., "forest loss") unless
  appropriate semantic labels are actually available.
- **Potential later extension:** DynamicEarthNet (do not implement now).

---

## CV Direction

- **MVP candidate:** CVA (Change Vector Analysis) + thresholding
- This decision is **provisional** — it requires a feasibility experiment.
- Do **not** implement a deep model until the simpler approach is evaluated.
- OSCD image pairs are already registered; do not build a registration system
  unless evidence shows a real need.

---

## Development Process Rules

1. **Inspect before modifying.** Always understand the current state before making
   changes.

2. **Plan before implementing.** Create a brief plan and identify ambiguities.

3. **Test after implementing.** Run relevant tests and verify correctness.

4. **Report honestly.** State exactly what was done, what assumptions were made,
   and what remains.

5. **Do not claim implementation beyond what exists.** Separate "created" from
   "validated" from "experimentally validated."

6. **Do not auto-advance phases.** Stop after completing a scoped task and report.

7. **Multi-agent development must remain controlled and scoped.** Do not create
   all development agents at once.

---

## What This Project Is NOT

- Not a production SaaS platform
- Not a foundation-model training project
- Not a microservices architecture
- Not a RAG system
- Not dependent on any specific agent framework

It is a **research prototype** demonstrating that tool-augmented agentic reasoning
can produce better grounded answers than alternatives, evaluated on a real
remote-sensing benchmark.
