"""Optional TF-IDF + logistic-regression intent classifier for comparison studies."""

from __future__ import annotations

from typing import Any, Sequence


def fit_tfidf_intent_classifier(
    questions: Sequence[str],
    labels: Sequence[str],
    *,
    c: float = 1.0,
    ngram_range: tuple[int, int] = (1, 2),
) -> Any:
    """Fit a small, reproducible intent model.

    Scikit-learn is an optional ``nlp-eval`` dependency and is imported only
    when this comparison model is requested. No probabilities are exposed.
    """
    if len(questions) != len(labels) or not questions:
        raise ValueError("questions and labels must be non-empty sequences of equal length")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional NLP evaluation dependencies with `pip install -e .[nlp-eval]`."
        ) from exc

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=ngram_range,
                sublinear_tf=True,
            )),
            ("classifier", LogisticRegression(
                C=c,
                class_weight="balanced",
                max_iter=2000,
                random_state=17,
            )),
        ]
    )
    return model.fit(list(questions), list(labels))
