# EarthChange-AI

**Language-Guided Satellite Change Analysis**

---

## Research Question

> *Can tool-augmented agentic reasoning improve grounded question answering for multi-temporal satellite imagery compared with a pure VLM and a fixed CV pipeline?*

This is a hypothesis to test, **not** an assumed conclusion.

## Project Purpose

EarthChange AI is a research prototype that turns natural-language questions about a pair of satellite images into a structured, tool-backed analysis request. Its current language component is a deterministic local parser with answer templates; it does not use an LLM or a vision-language model. The pipeline separates:

- **Language extracts.** NLP classifies the request and extracts its target, operation, and image region.
- **The planner routes.** The agent selects a fixed sequence of registered tools from the structured request.
- **Vision measures.** Computer vision produces the pixel-level change evidence.

The language model must never invent quantitative visual measurements. All quantitative claims are grounded in evidence produced by CV tools.

## High-Level Architecture

```
User Question
  → NLP Parsing (intent, target, operation, region)
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

### Implemented Capabilities

1. Before/after image input through the local dashboard
2. CVA change detection with Otsu thresholding
3. Change-mask and magnitude visualization
4. Pixel-level quantification and optional physical area from a supplied pixel size or projected GeoTIFF metadata
5. Structured intent, target, operation, region, metric, and constraint extraction
6. Bounded multi-step decomposition for supported compound questions
7. A rule-based baseline and an independently evaluated TF-IDF + Logistic Regression alternative
8. Evidence-grounded answers and per-tool execution records
9. Regional analysis and comparison for image quadrants (not map coordinates)
10. JavaScript dashboard served locally by the Python backend

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
| Test suite | 88 passed; 6 non-blocking rasterio deprecation warnings |
| CVA feasibility experiment | Implemented; real OSCD evaluation pending imagery |
| CV pipeline | Implemented; CVA remains provisional pending benchmark results |
| NLP rule parser | Implemented; evaluated against a held-out curated question set |
| TF-IDF intent classifier | Optional experimental baseline; lower held-out score, not used for routing |
| Agent orchestration | Implemented with a fixed registered-tool plan |
| Evidence packaging | Implemented, including completed and failed tool records |
| Answer generation | Implemented with deterministic evidence-grounded templates |
| Evaluation framework | Implemented; benchmark results depend on local OSCD data |
| UI | Implemented as a local JavaScript dashboard |
| OSCD data loading | Implemented; real dataset evaluation is pending |

Execution plans are checked against the registered tool inputs and parameters.
The language component is a bounded research parser, not a general-purpose LLM.
See [NLP design and evaluation](docs/nlp.md) for supported requests, the curated
dataset, and reproducible model comparisons. The dashboard exposes tool-by-tool outcomes and regional statistics. GeoTIFF
pairs must have matching dimensions, CRS, and affine alignment; nodata pixels
are not supported yet. Physical area is shown only when pixel dimensions in
metres are known. The CVA output measures spectral pixel differences, not
semantic classes such as forest or water. It is a research prototype, not a
validated land-cover product.

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

### Dashboard

Launch the local JavaScript dashboard on Windows by double-clicking
`run_dashboard.bat`, or from PowerShell:

```powershell
cd "D:\Earth ChangeAi"
& ".\.venv\Scripts\python.exe" ".\app\server.py"
```

Then open `http://127.0.0.1:8765`. The dashboard accepts matching PNG, JPEG,
and GeoTIFF pairs and sends them to the existing EarthChange analysis pipeline.
Choose **Load OSCD sample** to use a bundled Sentinel-2 pair (2015 and 2017).
Its RGB preview files and CC BY-NC-SA 4.0 attribution are in
`data/samples/oscd100/README.md`.
Stop the local server with Ctrl+C in its terminal window.

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
