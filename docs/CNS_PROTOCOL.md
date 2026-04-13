# CNS protocol — 4 operator-controlled modes

This document specifies a *protocol*, not a proof. It defines four
modes the operator can deliberately enter, the *objective markers*
that distinguish them (from data this pipeline can actually see), the
expected gate behaviour under each mode, and the transition criteria.
Nothing here is claimed as established until an operator fills in
`artifacts/live_runs/.../RUN_REPORT.md` with real data.

Scope: this is a **closed-loop self-regulation** protocol against a
signal-quality gate, not a neural readout. The inputs are RR intervals
over LSL; the outputs are the four-state physio vocabulary
(`EXECUTE_ALLOWED`, `EXECUTE_REDUCED`, `ABSTAIN`, `SENSOR_DEGRADED`).

## Modes

### 1. `stillness`

**Intent.** Deliberate low-stimulation state — seated, eyes closed or
soft gaze, slow nasal breathing, no task.

**Objective markers** (available from this pipeline):
- `continuity_fraction ≈ 1.0` (sensor contact solid; no gaps).
- `stability` in the **upper quartile** of the user's profile band.
- `rmssd_ms` moderate-to-high; specifically within
  `profile.rmssd_ms.p50 .. p95`.
- `confidence` at or above `threshold_allow`.

**Expected gate behaviour.**
- `EXECUTE_ALLOWED` stable for most of the session;
- rare `EXECUTE_REDUCED` excursions only.

**Transition-in criteria.** 60 s of uninterrupted `EXECUTE_ALLOWED`
after the operator enters the posture.

**Transition-out criteria.** Any deliberate task engagement, or a
sustained drop below `threshold_allow` for > 30 s.

---

### 2. `focused_work`

**Intent.** Sustained attention on one cognitive task (coding block,
deep reading, writing). Breath somewhat regular but not deliberate;
posture working-adapted; moderate task-induced tension.

**Objective markers.**
- `continuity_fraction ≈ 1.0`.
- `stability` in the **middle** of the profile band
  (`p05 .. p50`) — some task-induced HRV reduction.
- `rmssd_ms` dropping relative to stillness, but still plausible.
- `confidence` oscillating between `threshold_abstain` and
  `threshold_allow`.

**Expected gate behaviour.**
- Mixed `EXECUTE_REDUCED` and `EXECUTE_ALLOWED` with `EXECUTE_REDUCED`
  dominant in the middle third of the block.

**Transition-in criteria.** ≥ 3 min since the last `stillness`
baseline and an observable shift in `rmssd_ms` (downward) within the
user's profile's `p05..p50` band.

**Transition-out criteria.** Task end, or prolonged `ABSTAIN`
episodes (> 90 s consecutive) that indicate overload.

---

### 3. `overload`

**Intent.** Cognitive overload — more than one demanding task, time
pressure, interrupted focus, frustration, or late-night fatigue.

**Objective markers.**
- `continuity_fraction ≈ 1.0` (we differentiate from sensor-degraded).
- `stability` near the **lower quartile** of the profile band.
- `rmssd_ms` at the low end of profile (`< p05`) persistently.
- `confidence` near or below `threshold_abstain`.

**Expected gate behaviour.**
- Sustained `ABSTAIN` is the dominant state; `EXECUTE_REDUCED`
  appears only briefly;
- `EXECUTE_ALLOWED` is rare and short-lived (< 30 s).

**Transition-in criteria.** Operator self-declares (subjective), AND
`ABSTAIN` becomes the modal state for > 60 s.

**Transition-out criteria.** Deliberate cool-down move (step away,
breath reset) or session end.

---

### 4. `recovery`

**Intent.** Post-load cool-down. Target is an observable return to the
baseline distribution.

**Objective markers.**
- `continuity_fraction ≈ 1.0`.
- `rmssd_ms` climbing back toward `profile.rmssd_ms.p50` within the
  cool-down window.
- `stability` recovering toward the profile upper half.
- `confidence` rising through `threshold_abstain` and into
  `threshold_allow`.

**Expected gate behaviour.**
- Transition: `ABSTAIN → EXECUTE_REDUCED → EXECUTE_ALLOWED` within
  3-5 min of cool-down posture.

**Transition-in criteria.** Operator stops the load task and enters a
deliberate cool-down posture.

**Transition-out criteria.** The gate stabilises at `EXECUTE_ALLOWED`
for > 60 s, OR 10 min pass without recovery (in which case the
session ends and is flagged in the run report).

---

## What is NOT claimed

- **No claim that gate states distinguish these modes automatically.**
  The protocol says the *operator* enters a labeled mode; the gate's
  state sequence is recorded; the run report compares the two.
  Distinguishing "overload" from "focused_work" from "recovery"
  purely from gate states is a hypothesis for the protocol to test,
  not a built-in capability.
- **No clinical claim.** RMSSD bands are calibrated per user as
  signal-quality indicators; they are not medical diagnoses.
- **No universal thresholds.** Every mode description above uses the
  user's **own profile** bands. A generic user running the default
  gate will see different boundaries.

## Run protocol

For a single CNS-protocol day:

1. Record a `baseline-calm` session per `docs/LIVE_SESSION_PROTOCOL.md`.
2. Enter `focused_work` for 15 min. Record.
3. Enter `overload` for 10 min. Record.
4. Enter `recovery` for 5-10 min. Record.
5. Write `RUN_REPORT.md` in that day's `artifacts/live_runs/<DATE>/`
   folder with:
   - the operator's self-declared mode per session;
   - the dominant gate state per session;
   - whether the expected mode-to-gate mapping held;
   - any surprising episode worth analysing offline via
     `python -m neurophase.physio.session_replay`.

After N days of data: stability of the mapping is the question.
Three-of-four modes being reliably distinguishable across N days is
the explicit bar set in the v1.2 brief.
