"""Import-surface contract: the root package must stay lean.

The kernelization audit (PHASE 1) established that ``import neurophase``
must not eagerly load heavy scientific dependencies. This suite fixes
that contract as a test so the surface cannot silently regrow.

Contract:

* ``import neurophase`` loads without pulling ``mne``, ``pandas``,
  ``pywt``, ``neurodsp``, ``networkx``, ``sklearn``.
* ``import neurophase.api`` loads the blessed runtime façade without
  those same heavy modules either.
* Backward-compat symbols (``KLRConfig``) remain reachable through
  :pep:`562` lazy attribute access on first reference.
* ``neurophase.experiments`` itself is safe to import; only its concrete
  submodules may pull research dependencies, and those do so at call
  time.

Implementation note
-------------------
The assertions run in a fresh Python subprocess so the parent pytest
process (which has already loaded the full scientific stack for other
tests) cannot contaminate the observation. Each subprocess reports
``"OK"`` followed by the set of loaded top-level modules, or raises.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

HEAVY_MODULES = (
    "mne",
    "pandas",
    "pywt",
    "neurodsp",
    "networkx",
    "sklearn",
)


def _run(script: str) -> str:
    """Run ``script`` in a fresh interpreter and return its stdout.

    A non-zero exit is promoted to a test failure with the captured
    stderr in the message.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"subprocess failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout


def test_root_import_has_version() -> None:
    out = _run(
        """
        import neurophase
        assert isinstance(neurophase.__version__, str) and neurophase.__version__
        print(neurophase.__version__)
        """
    )
    assert out.strip()


@pytest.mark.parametrize("heavy", HEAVY_MODULES)
def test_root_import_does_not_pull_heavy_dep(heavy: str) -> None:
    """``import neurophase`` must not load any heavy scientific module."""
    out = _run(
        f"""
        import sys
        import neurophase
        loaded = {heavy!r} in sys.modules
        print("LOADED" if loaded else "CLEAN")
        """
    )
    assert out.strip() == "CLEAN", (
        f"import neurophase leaked {heavy!r} into sys.modules; "
        f"something in the __init__ chain pulled it eagerly."
    )


@pytest.mark.parametrize("heavy", HEAVY_MODULES)
def test_api_facade_does_not_pull_heavy_dep(heavy: str) -> None:
    """``import neurophase.api`` must stay clean of research deps too."""
    out = _run(
        f"""
        import sys
        import neurophase.api  # noqa: F401
        print("LOADED" if {heavy!r} in sys.modules else "CLEAN")
        """
    )
    assert out.strip() == "CLEAN", (
        f"import neurophase.api leaked {heavy!r}; the blessed façade pulled a research dependency."
    )


def test_experiments_package_is_safe_to_import() -> None:
    """``import neurophase.experiments`` must not trigger mne/pandas load."""
    out = _run(
        """
        import sys
        import neurophase.experiments  # noqa: F401
        leaks = [m for m in ("mne", "pandas") if m in sys.modules]
        print("LEAKS=" + ",".join(leaks) if leaks else "CLEAN")
        """
    )
    assert out.strip() == "CLEAN", (
        f"neurophase.experiments package init leaked a research dep: {out.strip()}"
    )


def test_klr_config_backward_compat_is_lazy() -> None:
    """``from neurophase import KLRConfig`` works via PEP 562 ``__getattr__``."""
    out = _run(
        """
        import sys
        import neurophase
        assert "neurophase.reset" not in sys.modules, (
            "neurophase.reset loaded eagerly, but should be lazy."
        )
        cfg = neurophase.KLRConfig  # triggers __getattr__
        assert cfg is not None
        assert "neurophase.reset" in sys.modules
        print("OK")
        """
    )
    assert out.strip() == "OK"


def test_unknown_attribute_raises_attribute_error() -> None:
    """``__getattr__`` must not silently swallow typos."""
    import neurophase

    with pytest.raises(AttributeError):
        _ = neurophase.does_not_exist  # type: ignore[attr-defined]


def test_dir_includes_public_symbols() -> None:
    import neurophase

    d = dir(neurophase)
    assert "__version__" in d
    assert "KLRConfig" in d


def test_core_phase_importable_without_pywt() -> None:
    """``from neurophase.core.phase import compute_phase`` must work without pywt.

    ``pywt`` is imported lazily inside :func:`preprocess_signal`; the
    module itself must stay importable so consumers that only want
    ``compute_phase(..., denoise=False)`` or ``adaptive_threshold`` do
    not carry the wavelet dependency.
    """
    out = _run(
        """
        import sys
        from neurophase.core.phase import compute_phase, adaptive_threshold  # noqa: F401
        print("LOADED" if "pywt" in sys.modules else "CLEAN")
        """
    )
    assert out.strip() == "CLEAN", "core.phase eagerly loaded pywt"
