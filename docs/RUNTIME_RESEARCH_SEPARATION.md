# neurophase — Runtime / Research Separation

**Status.** Load-bearing boundary as of kernelization v1.

The package has two installation modes and two dependency closures.
The distinction is not cosmetic; it is enforced at the Python import
level (``tests/test_import_surface.py``) and at the package-metadata
level (``pyproject.toml``).

## Two install modes

| Mode | Command | What you get | Closure |
|---|---|---|---|
| **Kernel-only** | `pip install neurophase` | Runtime gate, orchestrator, audit ledger, canonical frame, governance loaders. | `numpy`, `scipy`, `PyYAML` |
| **Research-full** | `pip install neurophase[research]` | Kernel + BIDS loader, experiments, benchmarks, calibration, Ricci / SCP / δ-power metrics, synthetic simulators. | kernel + `mne`, `pandas`, `neurodsp`, `PyWavelets`, `scikit-learn`, `networkx` |
| **Dev** | `pip install neurophase[dev]` | Research-full + `pytest`, `ruff`, `mypy`, `hypothesis`. | research + dev tools |

Kernel-only is the real surface for production consumers. Research-full
is for reproducing the null-result archive (``results/``) and for
authoring new experiments.

## Truth table — what belongs where

### Runtime kernel (stays on import path of `neurophase.api`)

| Module | Role |
|---|---|
| `neurophase/__init__.py` | thin root, lazy backward-compat |
| `neurophase/api.py` | blessed public façade — 32 symbols |
| `neurophase/contracts/frame.py` | canonical frame schema |
| `neurophase/core/phase.py` | phase extraction (pywt lazy) |
| `neurophase/core/order_parameter.py` | R(t) |
| `neurophase/core/kuramoto.py` | reference Kuramoto |
| `neurophase/data/temporal_validator.py` | B1 |
| `neurophase/data/stream_detector.py` | B2 + B6 |
| `neurophase/gate/execution_gate.py` | I₁–I₄ |
| `neurophase/gate/stillness_detector.py` | I₄ |
| `neurophase/gate/direction_index.py` | direction |
| `neurophase/gate/emergent_phase.py` | emergence criteria |
| `neurophase/analysis/regime.py` | G1 classifier |
| `neurophase/analysis/regime_transitions.py` | transition tracker |
| `neurophase/analysis/prediction_error.py` | cognitive state (pandas lazy) |
| `neurophase/policy/action.py` | I1 policy / action intent |
| `neurophase/explain.py` | decision explanation |
| `neurophase/runtime/pipeline.py` | StreamingPipeline |
| `neurophase/runtime/orchestrator.py` | RuntimeOrchestrator |
| `neurophase/runtime/memory_audit.py` | bounded memory audit |
| `neurophase/audit/decision_ledger.py` | F1 ledger |
| `neurophase/audit/replay.py` | deterministic replay |
| `neurophase/audit/session_manifest.py` | K1 manifest |
| `neurophase/governance/*.py` | INVARIANTS / CLAIMS / STATE_MACHINE / doctor |
| `neurophase/metrics/plv.py` | PLV, HeldOutSplit, PPC |
| `neurophase/metrics/rayleigh.py` | Rayleigh uniformity test |
| `neurophase/metrics/asymmetry.py` | kurtosis, skewness |
| `neurophase/metrics/effect_size.py` | Cohen's d / Hedges' g |
| `neurophase/metrics/hurst.py` | Hurst DFA / R/S |
| `neurophase/metrics/entropy.py` | Shannon / Rényi / Tsallis |
| `neurophase/metrics/plv_verdict.py` | 3-gate PLVVerdict |
| `neurophase/metrics/ism.py` | ISM + topological energy |
| `neurophase/validation/null_model.py` | NullModelHarness |
| `neurophase/validation/surrogates.py` | cyclic_shift, time_reversal |

### Adjunct subsystems (import path is stable, heavy deps are local)

| Module | Subsystem | Heavy deps |
|---|---|---|
| `neurophase/reset/` | KLR / ketamine-like reset controller (20 modules) | none at load; numpy only |
| `neurophase/state/` | executive monitor | numpy |
| `neurophase/oscillators/` | market + neural-sensor protocols | numpy |
| `neurophase/sensors/` | sensor registry + synthetic sensor | numpy |

These are reachable and importable with **kernel-only** deps. They are
architecturally adjacent to the canonical runtime path but do not flow
into `OrchestratedFrame` as first-class fields (see
`docs/RUNTIME_CANONICAL_FRAME.md` §5). The `klr_*` fields on
`DecisionFrame` are the *only* way KLR affects the canonical envelope,
and they are advisory.

### Research-only (require `neurophase[research]`)

| Module | Deps required to load |
|---|---|
| `neurophase/data/ds003458_loader.py` | `mne`, `pandas` |
| `neurophase/data/eeg_preprocessor.py` | `mne`, `scipy` |
| `neurophase/experiments/ds003458_*.py` | `mne`, `pandas`, `scipy` |
| `neurophase/experiments/synthetic_plv_validation.py` | `scipy` |
| `neurophase/benchmarks/neural_phase_generator.py` | `scipy` |
| `neurophase/benchmarks/phase_coupling.py` | `scipy` |
| `neurophase/benchmarks/stochastic_market_sim.py` | `scipy` |
| `neurophase/benchmarks/ppc_analytical.py` | `scipy` |
| `neurophase/benchmarks/parameter_sweep.py` | `scipy` |
| `neurophase/calibration/threshold.py` | `scipy` |
| `neurophase/calibration/stillness.py` | `scipy` |
| `neurophase/metrics/ricci.py` | `networkx`, `scipy` |
| `neurophase/metrics/scp.py` | `scipy`, `neurodsp` |
| `neurophase/metrics/delta_power.py` | `scipy`, `neurodsp` |
| `neurophase/metrics/delta_price_xcorr.py` | `scipy` |
| `neurophase/metrics/iplv.py` | `scipy` |
| `neurophase/metrics/trial_theta_lme.py` | `pandas`, `scipy` |
| `neurophase/risk/evt.py` | `scipy` |
| `neurophase/risk/mfdfa.py` | `scipy` |
| `neurophase/risk/sizer.py` | numpy (thin — listed because it is downstream of the gate and not on the canonical path) |
| `neurophase/sync/coupled_brain_market.py` | `scipy` |
| `neurophase/sync/market_phase.py` | `scipy` |
| `neurophase/intel/btc_field_order.py` | stdlib + numpy (domain-specific; kept in research class because it is a BTC-specific ingress validator, not generic runtime) |
| `neurophase/indicators/{fmn,qilm}.py` | numpy + scipy (crypto-specific scalar indicators; adjacent to BTC domain, not runtime) |
| `neurophase/agents/pi_agent.py` | numpy (legacy agent experiment) |

Note the distinction between "heavy at load" (the module top imports
e.g. `pandas`) and "heavy at call" (the module defers the import to a
method body). Only the latter is safe for the kernel. Several
research-class modules above load `scipy` at import; `scipy` is in the
runtime-core extras, so kernel-only installs still have it. `scipy` is
the single unavoidable scientific dep of this codebase because the
kernel performs Hilbert transforms and surrogate stats.

## Boundary enforcement

* **`tests/test_import_surface.py`** — 18 assertions in subprocess
  isolation: neither `import neurophase` nor `import neurophase.api`
  pulls `mne`, `pandas`, `pywt`, `neurodsp`, `networkx`, or `sklearn`
  into `sys.modules`. `neurophase.experiments` package init does not
  leak either.
* **`tests/test_pyproject_boundary.py`** — asserts that the research
  deps listed above are present in `pyproject.toml`'s
  `[project.optional-dependencies].research` but **not** in
  `[project].dependencies`.
* **`governance.completeness.API_FACADE_SURFACE`** (Doctor check #6) —
  every symbol on `neurophase.api.__all__` resolves.

If any of these break, the kernel contract has regressed.
