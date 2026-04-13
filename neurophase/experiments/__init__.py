"""Experiments — reproducible validation runs with JSON output.

Research-only surface. Each entry point pulls heavy scientific dependencies
(``mne``, ``pandas``, ``scipy.stats``) **locally**, not at package load time,
so that ``import neurophase.experiments`` is safe even when those
dependencies are absent. A missing dependency will raise the usual
``ModuleNotFoundError`` at call time, pointing to the actual import that
fails.

Available entry points::

    from neurophase.experiments.ds003458_analysis import run_analysis
    from neurophase.experiments.ds003458_csd_analysis import run_csd_analysis
    from neurophase.experiments.ds003458_delta_analysis import run_delta_analysis
    from neurophase.experiments.ds003458_delta_q import run_delta_q_analysis
    from neurophase.experiments.ds003458_rpe_analysis import run_rpe_analysis
    from neurophase.experiments.ds003458_scp_analysis import run_scp_analysis
    from neurophase.experiments.ds003458_trial_lme import run_trial_lme_analysis
    from neurophase.experiments.synthetic_plv_validation import run_sweep
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# The ``TYPE_CHECKING`` block is not executed at runtime but is visible to
# both ``ast.parse`` (used by ``governance.completeness.PUBLIC_MODULE_REACHABLE``)
# and to static type checkers. It therefore fulfils the kernel completeness
# contract without forcing the concrete experiment modules to load their
# heavy scientific dependencies on package import.
if TYPE_CHECKING:
    from neurophase.experiments.ds003458_analysis import run_analysis  # noqa: F401
    from neurophase.experiments.ds003458_csd_analysis import run_csd_analysis  # noqa: F401
    from neurophase.experiments.ds003458_delta_analysis import run_delta_analysis  # noqa: F401
    from neurophase.experiments.ds003458_delta_q import run_delta_q_analysis  # noqa: F401
    from neurophase.experiments.ds003458_rpe_analysis import run_rpe_analysis  # noqa: F401
    from neurophase.experiments.ds003458_scp_analysis import run_scp_analysis  # noqa: F401
    from neurophase.experiments.ds003458_trial_lme import run_trial_lme_analysis  # noqa: F401
    from neurophase.experiments.synthetic_plv_validation import run_sweep  # noqa: F401

__all__: list[str] = []
