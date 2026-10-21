"""Smoke tests — verify that the package structure is importable."""


def test_earthchange_imports():
    """The earthchange package can be imported and has a version."""
    import earthchange

    assert hasattr(earthchange, "__version__")
    assert isinstance(earthchange.__version__, str)
    assert earthchange.__version__ == "0.1.0"


def test_submodule_imports():
    """All submodules are importable."""
    import earthchange.cv
    import earthchange.nlp
    import earthchange.agent
    import earthchange.evidence
    import earthchange.evaluation
    import earthchange.llm

    # Each submodule should have a docstring (set in __init__.py)
    assert earthchange.cv.__doc__ is not None
    assert earthchange.nlp.__doc__ is not None
    assert earthchange.agent.__doc__ is not None
    assert earthchange.evidence.__doc__ is not None
    assert earthchange.evaluation.__doc__ is not None
    assert earthchange.llm.__doc__ is not None
