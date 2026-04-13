# `tests/` — taxonomy map

This directory holds ~120 test files covering every domain the repo
claims to defend. The layout is flat for legacy reasons; this file is
the **navigation index**. When you want to know where a given contract
is tested, start here.

No tests were moved to write this index. Every file listed below
exists exactly where it exists; only the categorisation is new.

## How tests are gated in CI

Markers registered in `pyproject.toml` (`[tool.pytest.ini_options]`):

| marker | meaning | default run? |
|---|---|---|
| (none) | any unit / integration / contract test | **yes** |
| `hardware` | requires Polar H10 + BT adapter / other device | no, opt-in |
| `acceptance` | full fresh-clone chain (slow) | no, opt-in |

Default `pytest` command excludes `hardware` and `acceptance` via
`addopts`. The CI workflow runs the default suite + an explicit
`-m acceptance` step. The `hardware` suite is never run in CI; it is
the Layer-C/D formalisation for operator-side execution.

## Domain map

Each row: **domain** → **what the test proves** → **canonical files**.
Filenames are matched verbatim against `tests/` at commit time.

### Runtime & gate kernel

| Domain | Proves | Canonical files |
|---|---|---|
| Execution gate (5-state) | I₁..I₄ + fail-closed invariant | `test_execution_gate.py`, `test_state_machine_spec.py`, `test_state_transitions.py`, `test_gate_first_execution.py`, `test_direction_index.py` |
| Stillness detector | I₄ advisory, regime classification | `test_stillness_detector.py`, `test_stillness_pipeline.py` |
| Streaming pipeline | B1/B2/B6 temporal precondition + composition | `test_runtime_pipeline.py`, `test_runtime_orchestrator.py`, `test_pipeline_frame.py`, `test_batch_pipeline.py`, `test_closure_path.py` |
| Canonical frame schema | on-wire frame contract | `test_canonical_frame.py`, `test_pipeline_frame_contract.py` |
| Determinism | bit-identical replay across seeds / runs | `test_determinism_certification.py`, `test_determinism_lock.py`, `test_replay_engine.py` |
| Memory bounded | RUNTIME_MEMORY_BOUNDED axis | `test_memory_audit.py` |

### Decision layer

| Domain | Proves | Canonical files |
|---|---|---|
| Action policy | sizing + direction composition | `test_action_policy.py` |
| Direction index | skew + curvature + bias → LONG/SHORT/FLAT | `test_direction_index.py` |
| Sizer | CVaR + synchronisation + multifractal deflator | `test_sizer.py` |
| Market-side decision chain | end-to-end DI + Sizer + Gate + Ledger | `test_market_decision_chain.py` |
| BTC field order | external signal-scan request builder | `test_btc_field_order.py` |

### Physio stack (shared core)

| Domain | Proves | Canonical files |
|---|---|---|
| Replay CSV ingest | HRS / RR validation + provenance marker | `test_physio_replay.py` |
| HRV features | rolling window + fail-closed feature nulls | `test_physio_features.py` |
| Physio gate | 4-state vocabulary + invariants | `test_physio_gate.py` |
| Replay pipeline | replay-driven `PhysioSession` composition | `test_physio_pipeline.py` |
| Replay demo | one-command CLI smoke | `test_physio_demo.py` |
| Live LSL consumer | subprocess integration + handshake + round-trip ledger | `test_physio_live.py` |
| Adversarial faults | 7 fault classes, no false EXECUTE_ALLOWED | `test_physio_faults.py` |
| Session ledger | immutable JSONL + replay parity | `test_physio_ledger.py` |
| Session parity matrix | decision-parity vs full-parity + tolerance + partial-line | `test_session_parity_matrix.py` |
| Calibration | profile schema + calibrator floors + gate modes | `test_physio_calibration.py` |
| Canonical contract | ONE place asserting the LSL + sentinel + exit-code contract | `test_physio_contract.py` |
| Fail-closed mutation proof | each guard enumerated with kill-test | `test_fail_closed_mutation.py` |
| Hardware (Polar H10) | Layer-C/D formalisation; marker `hardware`; skipped in CI | `test_hardware_polar.py` |

### Audit & governance

| Domain | Proves | Canonical files |
|---|---|---|
| Decision ledger | SHA-256 chain + tamper-evidence | `test_decision_ledger.py`, `test_market_decision_chain.py` |
| Session manifest | session-scope audit binding | `test_session_manifest.py` |
| Claim registry | CLAIMS.yaml schema + promotion rules | `test_claim_registry.py`, `test_bibliography_contract.py` |
| Invariant registry | INVARIANTS.yaml + binding to tests | `test_invariant_registry.py` |
| State-machine contract | STATE_MACHINE.yaml exhaustiveness | `test_state_machine_spec.py` |
| Aesthetics / honest naming | HN* naming invariants | `test_aesthetics_contract.py`, `test_honest_naming.py` |
| Doctor (11 axes) | 8th + 9th + 10th axis composition | `test_eighth_axis_doctor.py`, `test_ninth_axis_completeness.py`, `test_tenth_axis_reproducibility.py`, `test_seventh_axis.py` |
| Public API surface | import + completeness reachability | `test_import_surface.py`, `test_public_api.py`, `test_pyproject_boundary.py` |

### Physics + metrics

| Domain | Proves | Canonical files |
|---|---|---|
| PLV / iPLV | held-out + surrogate significance | `test_plv.py`, `test_iplv.py`, `test_held_out.py` |
| Phase extraction | Hilbert, Daubechies, Morlet | `test_phase_extraction.py`, `test_morlet.py`, `test_qilm.py` |
| Kuramoto simulator | coupled brain-market order parameter | `test_coupled_brain_market.py`, `test_kuramoto.py` |
| Market oscillators | GBM + neural EMA tracking | `test_stochastic_market.py`, `test_market_phase.py` |
| Rayleigh + surrogates | directional + null significance | `test_rayleigh.py`, `test_surrogates.py`, `test_surrogate_seed_contract.py` |
| Entropy, Hurst, Ricci | scalar metrics | `test_entropy.py`, `test_hurst.py`, `test_ricci.py` |
| SCP / delta / trial-LME | low-frequency + trial-theta | `test_scp.py`, `test_delta_power.py`, `test_delta_price_xcorr.py`, `test_trial_theta_lme.py` |
| Synthetic PLV bridge | methodology validation at known c | `test_synthetic_plv.py` |
| 7-axis truth | analytical PPC + Rayleigh + effect sizes | `test_seventh_axis.py`, `test_seventh_axis_reproducibility.py` |

### Research experiments (ds003458)

| Domain | Proves | Canonical files |
|---|---|---|
| Loader + preprocessor | BIDS ingest + channel picking | `test_ds003458.py` |
| PLV analysis (null) | 0/17 FMθ × market PLV | `test_ds003458.py::TestVerdictIntegration` |
| Delta / SCP / trial-LME | all NULL on this dataset | same |
| CSD + Morlet existence | 9/17 covariate, mixed direction | (results JSON only; no kernel test) |
| ΔQ_gate utility benchmark | 0/17 across 3 metrics, NULL | (results JSON only) |

### Bridge / contracts / adapters

| Domain | Proves | Canonical files |
|---|---|---|
| Bridge layer | ingress + egress contracts | `test_bridge_layer.py`, `test_observatory_export.py` |
| Sensor adapters | NullNeuralExtractor + sensor registry | `test_sensor_adapter_layer.py` |
| Reset / KLR | ketamine-like reset subsystem | `test_reset_support_modules.py`, `test_klr_*.py` (pattern) |

### Acceptance (marker `acceptance`, not in default)

| Domain | Proves | Canonical files |
|---|---|---|
| Fresh-clone chain | demo + live loopback + ledger + strict replay, in a `git clone` | `test_acceptance_chain.py` |

### Hardware (marker `hardware`, never in CI)

| Domain | Proves | Canonical files |
|---|---|---|
| Polar H10 Layer-C/D | discover → connect → notify → RR parse → LSL emit → sentinel → clean shutdown | `test_hardware_polar.py` |

## Conventions

- **One invariant → one test** (CLAUDE.md `testing_policy`). If a
  failing test name is not a clear statement of the invariant being
  defended, rename.
- **Parser math → fixture tests with exact expected values** — no
  `pytest.approx` for bit-exact parsers (HRS, ledger hash).
- **Fail-closed state → explicit negative-path test**.
- **New public module → completeness-reachability entry** added in
  `neurophase/<pkg>/__init__.py`'s `TYPE_CHECKING` block.
- **New fail-closed guard → entry in `test_fail_closed_mutation.py`**
  so a regression that weakens the guard is caught by name.

## How to run

```bash
# Default gate (fast):
pytest tests/ -q

# Acceptance chain (slow, runs fresh-clone sub-flow):
pytest tests/ -q -m acceptance

# Hardware suite (requires real Polar H10 + BT adapter):
pytest tests/ -q -m hardware

# A specific domain, e.g. physio:
pytest tests/test_physio_*.py -q
```
