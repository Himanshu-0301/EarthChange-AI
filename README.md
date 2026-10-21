# EarthChange-AI

**An Agentic Vision-Language Assistant for Multi-Temporal Satellite Change Detection and Analysis**

---

## Research Question

> *Can tool-augmented agentic reasoning improve grounded question answering for multi-temporal satellite imagery compared with a pure VLM and a fixed CV pipeline?*

This is a hypothesis to test, **not** an assumed conclusion.

## Project Purpose

EarthChange AI is a language-driven, tool-augmented system for answering natural-language questions about one or two satellite images. It bridges computer vision and natural language processing through an agentic architecture where:

- **Vision detects.** Computer vision tools perform visual measurement.
- **Language understands and reasons.** NLP components interpret natural language.
- **The agent decides what evidence is needed.** An orchestration layer selects and sequences tools.

The language model must never invent quantitative visual measurements. All quantitative claims are grounded in evidence produced by CV tools.

## High-Level Architecture

```
User Question
  → NLP Understanding (intent classification, parameter extraction)
  → Structured Query
  → Agentic Planning (tool selection and sequencing)
  → Selected CV Tools (change detection, region analysis, etc.)
  → Structured Evidence (EvidencePackage)
  → Grounded Answer Generation
```

### Separation of Responsibilities

| Layer | Responsibility | Does NOT do |
|-------|---------------|-------------|
| **CV** (`earthchange.cv`) | Visual measurement: change detection, masking, area calculation | Interpret user questions or generate text |
| **NLP** (`earthchange.nlp`) | Intent classification, parameter extraction, query structuring | Perform visual analysis or generate measurements |
| **Agent** (`earthchange.agent`) | Tool selection, sequencing, plan execution | Perform CV or NLP tasks directly |
| **Evidence** (`earthchange.evidence`) | Package and structure tool outputs | Interpret or generate answers |
| **LLM** (`earthchange.llm`) | Answer generation from structured evidence | Invent quantitative facts not in evidence |
| **Evaluation** (`earthchange.evaluation`) | Metrics, benchmarking, experiment tracking | — |

## Current MVP Scope

### Capabilities (Planned)

1. Before/after satellite image input
2. Image preprocessing
3. Change detection (CVA + thresholding — provisional)
4. Change-mask visualization
5. Quantitative change calculation
6. Natural-language questions
7. Intent classification
8. Parameter extraction
9. Tool routing
10. Evidence-grounded answer generation
11. Clean interactive UI

### Supported Intent Families

- `GENERAL_CHANGE` — Is there change?
- `CHANGE_LOCALIZATION` — Where is the change?
- `CHANGE_QUANTIFICATION` — How much change?
- `REGIONAL_COMPARISON` — Compare regions
- `MULTI_STEP_ANALYSIS` — Compound questions
- `UNSUPPORTED` — Out of scope

### Region Vocabulary

`full_image`, `top`, `bottom`, `left`, `right`, `upper_left`, `upper_right`, `lower_left`, `lower_right`, `center`

### Operations

`describe`, `locate`, `count`, `area`, `percentage`, `compare`

## Dataset

**Primary benchmark:** [OSCD — Onera Satellite Change Detection Dataset](https://rcdaudt.github.io/oscd/)

Used for binary change-detection evaluation. Semantic extensions (e.g., DynamicEarthNet) may be explored later.

## Current Implementation Status

| Component | Status |
|-----------|--------|
| Repository structure | ✅ Created |
| Project configuration | ✅ Created |
| Test suite | ✅ Passing |
| CVA feasibility experiment | 🟡 Implemented; OSCD evaluation pending imagery |
| CV pipeline | ⬜ Not started |
| NLP pipeline | ⬜ Not started |
| Agent orchestration | ⬜ Not started |
| Evidence packaging | ⬜ Not started |
| LLM integration | ⬜ Not started |
| Evaluation framework | ⬜ Not started |
| UI | ⬜ Not started |
| OSCD data loading | ✅ Implemented and validated with synthetic fixtures |

### CVA Feasibility Experiment

The experimental baseline computes per-pixel spectral change magnitude, applies
an Otsu threshold, and compares the resulting mask with OSCD labels. It reports
precision, recall, F1, and IoU for explicitly selected pairs. It does not perform
registration, normalization, or morphology.

```bash
python experiments/cv_feasibility.py --root data/raw --pairs beihai rennes
```

The experiment writes results to `experiments/outputs/cv_feasibility.csv` by
default. Real OSCD evaluation remains pending until the dataset is placed under
`data/raw`.

## Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

### Project Layout

```
src/earthchange/       # Main package
  cv/                  # Computer vision tools
  nlp/                 # Natural language processing
  agent/               # Agent orchestration
  evidence/            # Evidence packaging
  evaluation/          # Metrics and benchmarking
  llm/                 # LLM interface
tests/                 # Test suite
notebooks/             # Jupyter notebooks
data/                  # Datasets (raw, processed, samples)
experiments/           # Experiment configurations and results
docs/                  # Documentation
app/                   # Application UI (future)
```

## License

MIT
