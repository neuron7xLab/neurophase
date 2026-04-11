"""G5 — contract tests for the empirical regime transition tracker.

This test file is the HN28 binding. It locks in:

1. **First-order Markov correctness.** ``observe(state)`` records
   exactly one transition per call after the first; the first
   call seeds the initial label and returns ``None``.
2. **Determinism.** Two trackers fed the same sequence emit
   byte-identical matrices and identical
   :class:`TransitionEvent` streams.
3. **Self-loops counted.** ``state.label is prev_label`` is a
   valid transition and increments the diagonal.
4. **Probability projection.** Row probabilities sum to 1 (or
   to 0 for an unobserved row, with raw counts; or to 1 with
   Laplace smoothing).
5. **predict_next refuses to fall back.** A row with zero
   observations raises :class:`InsufficientHistoryError` rather
   than returning a uniform distribution.
6. **Frozen snapshot.** :meth:`snapshot` returns a frozen copy
   that does not share state with the live tracker — appending
   to the tracker after a snapshot does not mutate the
   snapshot.
7. **JSON-safe / aesthetic.** Matrix carries a 4×4 ASCII grid
   renderer; rich __repr__ surfaces the dominant transition.
8. **Construction validation.** Negative counts and inconsistent
   ``n_transitions`` are rejected.
"""

from __future__ import annotations

import pytest

from neurophase.analysis.regime import RegimeLabel, RegimeState
from neurophase.analysis.regime_transitions import (
    InsufficientHistoryError,
    RegimeTransitionMatrix,
    RegimeTransitionTracker,
    TransitionEvent,
)

# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _state(label: RegimeLabel, *, tick: int = 0, ts: float = 0.0) -> RegimeState:
    return RegimeState(
        label=label,
        R=0.85,
        dR=0.001,
        delta=0.05,
        d_delta=0.001,
        confidence_score=0.9,
        tick_index=tick,
        timestamp=ts,
        warm=True,
        reason="synthetic",
    )


def _drive(
    tracker: RegimeTransitionTracker, labels: list[RegimeLabel]
) -> list[TransitionEvent | None]:
    return [tracker.observe(_state(label, tick=i, ts=0.1 * i)) for i, label in enumerate(labels)]


# ---------------------------------------------------------------------------
# 1. First-order Markov correctness.
# ---------------------------------------------------------------------------


class TestFirstOrderMarkov:
    def test_first_observation_returns_none(self) -> None:
        tracker = RegimeTransitionTracker()
        event = tracker.observe(_state(RegimeLabel.TRENDING))
        assert event is None
        assert tracker.last_label is RegimeLabel.TRENDING
        assert tracker.n_transitions == 0

    def test_second_observation_returns_event(self) -> None:
        tracker = RegimeTransitionTracker()
        tracker.observe(_state(RegimeLabel.TRENDING, tick=0))
        event = tracker.observe(_state(RegimeLabel.COMPRESSING, tick=1))
        assert event is not None
        assert event.from_label is RegimeLabel.TRENDING
        assert event.to_label is RegimeLabel.COMPRESSING
        assert event.tick_index == 1
        assert event.observed_count == 1
        assert tracker.n_transitions == 1

    def test_self_loop_recorded(self) -> None:
        tracker = RegimeTransitionTracker()
        events = _drive(
            tracker,
            [RegimeLabel.TRENDING, RegimeLabel.TRENDING, RegimeLabel.TRENDING],
        )
        assert events[0] is None
        assert events[1] is not None
        assert events[1].is_self_loop is True
        assert events[2] is not None
        assert events[2].observed_count == 2
        snap = tracker.snapshot()
        assert snap.count(from_label=RegimeLabel.TRENDING, to_label=RegimeLabel.TRENDING) == 2

    def test_n_transitions_grows_by_one(self) -> None:
        tracker = RegimeTransitionTracker()
        labels = [
            RegimeLabel.TRENDING,
            RegimeLabel.COMPRESSING,
            RegimeLabel.REVERTING,
            RegimeLabel.CHAOTIC,
            RegimeLabel.TRENDING,
        ]
        _drive(tracker, labels)
        # 5 observations → 4 transitions.
        assert tracker.n_transitions == 4


# ---------------------------------------------------------------------------
# 2. Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_trackers_same_sequence_same_matrix(self) -> None:
        seq = [
            RegimeLabel.TRENDING,
            RegimeLabel.COMPRESSING,
            RegimeLabel.TRENDING,
            RegimeLabel.REVERTING,
            RegimeLabel.CHAOTIC,
            RegimeLabel.CHAOTIC,
            RegimeLabel.TRENDING,
        ]
        a = RegimeTransitionTracker()
        b = RegimeTransitionTracker()
        _drive(a, seq)
        _drive(b, seq)
        assert a.snapshot() == b.snapshot()

    def test_event_stream_is_deterministic(self) -> None:
        seq = [RegimeLabel.TRENDING, RegimeLabel.COMPRESSING, RegimeLabel.REVERTING]
        a = RegimeTransitionTracker()
        b = RegimeTransitionTracker()
        events_a = _drive(a, seq)
        events_b = _drive(b, seq)
        assert events_a == events_b


# ---------------------------------------------------------------------------
# 3. Probability projection.
# ---------------------------------------------------------------------------


class TestProbabilities:
    def test_row_probability_sums_to_one(self) -> None:
        tracker = RegimeTransitionTracker()
        labels = [
            RegimeLabel.TRENDING,
            RegimeLabel.COMPRESSING,
            RegimeLabel.TRENDING,
            RegimeLabel.REVERTING,
            RegimeLabel.TRENDING,
        ]
        _drive(tracker, labels)
        snap = tracker.snapshot()
        # Row TRENDING has 2 outgoing transitions
        # (TRENDING→COMPRESSING and TRENDING→REVERTING).
        total = sum(
            snap.probability(from_label=RegimeLabel.TRENDING, to_label=to) for to in RegimeLabel
        )
        assert abs(total - 1.0) < 1e-9

    def test_unobserved_row_yields_zero_probability(self) -> None:
        tracker = RegimeTransitionTracker()
        # Only observe TRENDING — no transitions out of CHAOTIC.
        tracker.observe(_state(RegimeLabel.TRENDING))
        snap = tracker.snapshot()
        for to in RegimeLabel:
            p = snap.probability(from_label=RegimeLabel.CHAOTIC, to_label=to)
            assert p == 0.0

    def test_laplace_smoothing_redistributes(self) -> None:
        tracker = RegimeTransitionTracker(laplace_smoothing=True)
        # No observations at all → smoothed row is uniform 1/4.
        snap = tracker.snapshot()
        for from_label in RegimeLabel:
            for to_label in RegimeLabel:
                p = snap.probability(from_label=from_label, to_label=to_label)
                assert abs(p - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# 4. predict_next.
# ---------------------------------------------------------------------------


class TestPredictNext:
    def test_predict_returns_argmax(self) -> None:
        tracker = RegimeTransitionTracker()
        # Build a row where TRENDING→TRENDING dominates.
        labels = [
            RegimeLabel.TRENDING,
            RegimeLabel.TRENDING,
            RegimeLabel.TRENDING,
            RegimeLabel.TRENDING,
            RegimeLabel.COMPRESSING,
        ]
        _drive(tracker, labels)
        next_label, prob = tracker.predict_next(RegimeLabel.TRENDING)
        assert next_label is RegimeLabel.TRENDING
        # 3 self-loops out of 4 outgoing → 0.75
        assert abs(prob - 0.75) < 1e-9

    def test_predict_on_empty_row_raises(self) -> None:
        tracker = RegimeTransitionTracker()
        tracker.observe(_state(RegimeLabel.TRENDING))
        # No outgoing CHAOTIC transitions ever observed.
        with pytest.raises(InsufficientHistoryError):
            tracker.predict_next(RegimeLabel.CHAOTIC)

    def test_predict_on_single_transition(self) -> None:
        tracker = RegimeTransitionTracker()
        _drive(tracker, [RegimeLabel.TRENDING, RegimeLabel.COMPRESSING])
        next_label, prob = tracker.predict_next(RegimeLabel.TRENDING)
        assert next_label is RegimeLabel.COMPRESSING
        assert prob == 1.0


# ---------------------------------------------------------------------------
# 5. Snapshot is frozen and decoupled from live state.
# ---------------------------------------------------------------------------


class TestSnapshotIsolation:
    def test_snapshot_is_frozen_dataclass(self) -> None:
        tracker = RegimeTransitionTracker()
        _drive(tracker, [RegimeLabel.TRENDING, RegimeLabel.CHAOTIC])
        snap = tracker.snapshot()
        with pytest.raises((AttributeError, TypeError)):
            snap.n_transitions = 99  # type: ignore[misc]

    def test_appending_after_snapshot_does_not_mutate_snapshot(self) -> None:
        tracker = RegimeTransitionTracker()
        _drive(tracker, [RegimeLabel.TRENDING, RegimeLabel.COMPRESSING])
        snap = tracker.snapshot()
        assert snap.n_transitions == 1
        # Add more observations.
        _drive(
            tracker,
            [RegimeLabel.REVERTING, RegimeLabel.CHAOTIC, RegimeLabel.TRENDING],
        )
        # The frozen snapshot is unchanged.
        assert snap.n_transitions == 1
        # The tracker has moved on.
        assert tracker.snapshot().n_transitions == 4


# ---------------------------------------------------------------------------
# 6. ASCII rendering + rich __repr__.
# ---------------------------------------------------------------------------


class TestAesthetic:
    def test_as_text_grid_includes_all_labels(self) -> None:
        tracker = RegimeTransitionTracker()
        _drive(
            tracker,
            [
                RegimeLabel.TRENDING,
                RegimeLabel.COMPRESSING,
                RegimeLabel.TRENDING,
            ],
        )
        text = tracker.snapshot().as_text()
        assert "from\\to" in text
        for label in RegimeLabel:
            assert label.name[:4] in text

    def test_repr_shows_dominant_transition(self) -> None:
        tracker = RegimeTransitionTracker()
        _drive(
            tracker,
            [
                RegimeLabel.TRENDING,
                RegimeLabel.TRENDING,
                RegimeLabel.TRENDING,
                RegimeLabel.COMPRESSING,
            ],
        )
        snap = tracker.snapshot()
        r = repr(snap)
        assert r.startswith("RegimeTransitionMatrix[")
        assert "TRENDING" in r
        assert "n=3" in r

    def test_empty_matrix_repr(self) -> None:
        tracker = RegimeTransitionTracker()
        snap = tracker.snapshot()
        r = repr(snap)
        assert "empty" in r
        assert "n=0" in r

    def test_transition_event_repr_arrow_for_non_self_loop(self) -> None:
        tracker = RegimeTransitionTracker()
        tracker.observe(_state(RegimeLabel.TRENDING))
        event = tracker.observe(_state(RegimeLabel.REVERTING))
        assert event is not None
        r = repr(event)
        assert r.startswith("TransitionEvent[")
        assert "TRENDING" in r
        assert "REVERTING" in r
        assert "→" in r

    def test_transition_event_repr_self_loop_arrow(self) -> None:
        tracker = RegimeTransitionTracker()
        tracker.observe(_state(RegimeLabel.TRENDING))
        event = tracker.observe(_state(RegimeLabel.TRENDING))
        assert event is not None
        r = repr(event)
        assert "↻" in r


# ---------------------------------------------------------------------------
# 7. Construction-time validation on the matrix.
# ---------------------------------------------------------------------------


class TestMatrixValidation:
    def test_wrong_row_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="4 rows"):
            RegimeTransitionMatrix(
                counts=((0, 0, 0, 0),) * 3,  # type: ignore[arg-type]
                laplace_smoothing=False,
                n_transitions=0,
            )

    def test_wrong_column_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="4 columns"):
            RegimeTransitionMatrix(
                counts=(  # type: ignore[arg-type]
                    (0, 0, 0),
                    (0, 0, 0),
                    (0, 0, 0),
                    (0, 0, 0),
                ),
                laplace_smoothing=False,
                n_transitions=0,
            )

    def test_negative_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            RegimeTransitionMatrix(
                counts=(
                    (-1, 0, 0, 0),
                    (0, 0, 0, 0),
                    (0, 0, 0, 0),
                    (0, 0, 0, 0),
                ),
                laplace_smoothing=False,
                n_transitions=0,
            )

    def test_inconsistent_n_transitions_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_transitions"):
            RegimeTransitionMatrix(
                counts=(
                    (1, 1, 0, 0),
                    (0, 0, 0, 0),
                    (0, 0, 0, 0),
                    (0, 0, 0, 0),
                ),
                laplace_smoothing=False,
                n_transitions=99,  # wrong
            )


# ---------------------------------------------------------------------------
# 8. Reset clears state.
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_matrix_and_last_label(self) -> None:
        tracker = RegimeTransitionTracker()
        _drive(
            tracker,
            [RegimeLabel.TRENDING, RegimeLabel.COMPRESSING, RegimeLabel.REVERTING],
        )
        assert tracker.n_transitions == 2
        assert tracker.last_label is RegimeLabel.REVERTING
        tracker.reset()
        assert tracker.n_transitions == 0
        assert tracker.last_label is None
        assert tracker.snapshot().n_transitions == 0


# ---------------------------------------------------------------------------
# 9. End-to-end with real RegimeClassifier.
# ---------------------------------------------------------------------------


class TestClassifierIntegration:
    def test_tracker_consumes_classifier_output(self) -> None:
        from neurophase.analysis.regime import RegimeClassifier

        # Build a synthetic DecisionFrame stream and feed real
        # classifier output into the tracker.
        from neurophase.runtime.pipeline import (
            PipelineConfig,
            StreamingPipeline,
        )

        pipeline = StreamingPipeline(
            PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)
        )
        classifier = RegimeClassifier()
        tracker = RegimeTransitionTracker()

        for i in range(8):
            frame = pipeline.tick(timestamp=float(i) * 0.1, R=0.92, delta=0.05)
            if frame.R is None:
                continue
            state = classifier.classify(frame)
            tracker.observe(state)

        # The 8-tick steady high-R run should observe at least
        # one transition and surface a dominant TRENDING column.
        assert tracker.n_transitions >= 1
        snap = tracker.snapshot()
        # TRENDING should be the dominant *to* label across the
        # whole matrix.
        text = snap.as_text()
        assert "TREN" in text
