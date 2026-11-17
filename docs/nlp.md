# NLP query understanding

EarthChange AI converts a bounded set of satellite-change questions into a
validated `StructuredQuery`. The NLP module does not inspect pixels. Its output
is consumed by the registered-tool planner; the CV pipeline remains the source
of image measurements.

## Current pipeline

1. `parse_question()` applies an interpretable rule-based intent baseline.
2. The parser extracts the target, operation, image region(s), requested
   metrics, and supported constraints.
3. Supported compound requests receive an ordered `QuerySubtask` decomposition.
4. `plan_for_query(query)` translates the structured query into a validated
   sequence of registered tools. The original query is retained on the plan.
5. The answer layer reports only measurements in the evidence package.

`StructuredQuery.confidence` is optional and left unset by the parser. Keyword
matches are not calibrated probabilities.

## Supported and refused requests

Supported outputs describe pixel-level spectral change, count changed pixels,
calculate percentages or area when resolution is known, locate change in
predefined image regions, compare named regions, and handle the bounded
multi-step patterns listed in the dataset. Regions are image quadrants, not
geographic coordinates.

Semantic land-cover requests (for example, forest loss) are marked unsupported
because the current CV output is a binary change mask. Individual object counts
(for example, destroyed buildings) and counts without an explicit pixel unit
are also refused. Changed pixels are never presented as object counts.

## Evaluation dataset

`data/nlp/question_dataset.json` contains 72 manually authored questions with
expected structured outputs. The 6 intent classes are balanced across the
train (36), validation (18), and test (18) splits. Each intent appears in every
split. Questions were assigned to splits as distinct phrasings; the loader
rejects exact duplicates across all splits. The test set is intentionally
small, so scores are descriptive of this curated set only.

Run the independent NLP comparison without loading satellite imagery or CV:

```powershell
pip install -e ".[nlp-eval]"
python experiments/nlp_evaluation.py
```

The evaluation tunes TF-IDF n-gram range and Logistic Regression regularization
only on validation data, refits on train plus validation, and evaluates once on
the held-out test set. It compares intent accuracy, macro F1, and unsupported
precision/recall/F1 against the rule parser. It also reports rule-parser slot
precision/recall/F1 and exact multi-step decomposition accuracy. Results and
per-question predictions are saved to
`experiments/outputs/nlp/evaluation.json`. No classifier probabilities are
reported.

The optional classifier is an experiment, not the default request router. It
should only replace the rules after the measured held-out comparison supports
that change and the slot-extraction behavior remains safe.

## Current held-out result

On the current curated 18-question test split, the rule parser scored 100% intent
accuracy and macro F1, 100% precision/recall/F1 for unsupported intent, 100%
micro slot precision/recall/F1, and exact structured output on all 18 questions.
Its ordered decomposition matched all 3 multi-step examples. The tuned TF-IDF +
Logistic Regression baseline scored 77.8% intent accuracy and 78.7% macro F1;
unsupported recall was 66.7% (precision 100%). The rules therefore remain the
request router.

These are descriptive results on a small, manually authored dataset with
bounded vocabulary. The same project team authored the parser and annotations,
so the perfect rule score is especially vulnerable to shared assumptions and
does not establish performance on user-written questions. Expand the dataset
with independently authored, paraphrased examples and review disagreements
before making generalization claims.
