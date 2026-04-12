"""G5 — empirical regime transition model.

The G1 :class:`~neurophase.analysis.regime.RegimeClassifier` answers
*"what regime is the current frame in?"*. G5 answers the obvious
follow-up: *"how does the current regime tend to move?"*. It is
the simplest possible empirical transition model — a 4×4 first-
order Markov **count** matrix over observed regime sequences,
with row-normalised probabilities derived on demand.

Design constraints (formal review axes)
----------------------------------------

* **Elegance** — one stateful object, three methods (``observe``,
  ``predict_next``, ``snapshot``). No knobs, no priors except an
  optional Laplace (add-one) smoothing flag.
* **Aesthetics** — :class:`RegimeTransitionMatrix` carries a
  4×4 ASCII grid renderer (``as_text``) and a one-line rich
  repr that surfaces the dominant transition.
* **Beauty** — symmetric 4×4 over the canonical
  :class:`~neurophase.analysis.regime.RegimeLabel` enum. The row
  index is the *from* label, the column index is the *to*. No
  off-by-one ambiguity.
* **Simplicity** — pure integer counting + division. No EWMA, no
  forgetting factor, no kernel smoothing. The user gets the
  exact empirical history.
* **Precision** — typed ``RegimeLabel`` keys throughout; the
  matrix is frozen; every transition must come from
  :func:`observe(RegimeState)`. Predictions on a row with zero
  observations raise :class:`InsufficientHistoryError` rather
  than returning a silent uniform fallback.
* **Adaptability** — composes with the existing G1 classifier
  output without glue: drop the emitted :class:`RegimeState`
  straight into ``observe()``.

What G5 is NOT
--------------

* It is **not** a *predictive* model. The "prediction" returned
  by :meth:`RegimeTransitionTracker.predict_next` is the
  *empirical mode of the conditional distribution observed so
  far*. It does not extrapolate, and it makes no claim about
  the future beyond "this is what has happened in similar
  past situations."
* It is **not** a learned model. There are no weights, no
  gradient descent, no online optimisation. Every cell of the
  matrix is an integer count plus an optional Laplace prior.
* It is **not** higher-order. G5 is a first-order Markov model:
  the predicted next regime depends only on the current label.
  Higher-order context (G6+) is a deliberate future task.

HN28 contract
-------------

* The matrix is non-negative, integer-valued, and exactly
  4×4 — the four :class:`RegimeLabel` values, in declaration
  order.
* ``observe(state)`` is a pure function of the previous
  observed label and the new state's label. The first call
  records the initial label and increments no transition.
* ``snapshot()`` returns a frozen copy of the current matrix.
* ``predict_next(label)`` returns the argmax of row ``label``
  with the corresponding probability, or raises
  :class:`InsufficientHistoryError` if the row is empty.
* Two trackers fed the same regime sequence emit byte-identical
  matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neurophase.analysis.regime import RegimeState

from neurophase.analysis.regime import RegimeLabel

__all__ = [
    "InsufficientHistoryError",
    "RegimeTransitionMatrix",
    "RegimeTransitionTracker",
    "TransitionEvent",
]

#: Canonical row/column ordering — the declaration order of
#: :class:`RegimeLabel`. Locked here so a future enum reorder
#: would break the matrix layout in one obvious place.
_LABEL_ORDER: tuple[RegimeLabel, ...] = (
    RegimeLabel.TRENDING,
    RegimeLabel.COMPRESSING,
    RegimeLabel.REVERTING,
    RegimeLabel.CHAOTIC,
)


class InsufficientHistoryError(ValueError):
    """Raised when :meth:`predict_next` is asked about an unobserved row.

    The tracker refuses to fall back on a uniform / zero
    distribution because that would silently coerce a "no data"
    answer into a real prediction. Callers must explicitly
    handle the case where they have not yet observed the
    requested ``from`` label.
    """


@dataclass(frozen=True, repr=False)
class TransitionEvent:
    """Frozen, JSON-safe envelope for a single observed transition.

    Emitted by :meth:`RegimeTransitionTracker.observe` whenever
    the input :class:`RegimeState` differs from the previous
    label *or* matches it (self-transitions are recorded). The
    very first call to ``observe`` returns ``None`` because no
    transition has yet been observed; subsequent calls always
    return a :class:`TransitionEvent`.

    Attributes
    ----------
    from_label
        The previous regime label.
    to_label
        The current regime label.
    tick_index
        ``RegimeState.tick_index`` of the new state.
    timestamp
        ``RegimeState.timestamp`` of the new state.
    is_self_loop
        ``True`` iff ``from_label is to_label``. Self-loops are
        valid transitions and are counted in the matrix.
    observed_count
        Total number of times this exact ``(from, to)`` pair
        has been observed *including* this event. Always ≥ 1.
    """

    from_label: RegimeLabel
    to_label: RegimeLabel
    tick_index: int
    timestamp: float
    is_self_loop: bool
    observed_count: int

    def __post_init__(self) -> None:
        if self.observed_count < 1:
            raise ValueError(f"observed_count must be ≥ 1, got {self.observed_count}")
        expected_self = self.from_label is self.to_label
        if expected_self != self.is_self_loop:
            raise ValueError(
                f"is_self_loop={self.is_self_loop} disagrees with "
                f"from_label={self.from_label.name}, "
                f"to_label={self.to_label.name}"
            )

    def __repr__(self) -> str:  # aesthetic rich repr (HN28)
        arrow = "↻" if self.is_self_loop else "→"
        return (
            f"TransitionEvent[{self.from_label.name} {arrow} "
            f"{self.to_label.name} · "
            f"tick={self.tick_index} · n={self.observed_count}]"
        )


@dataclass(frozen=True, repr=False)
class RegimeTransitionMatrix:
    """Frozen 4×4 transition count matrix over the canonical regime labels.

    Constructed via :meth:`RegimeTransitionTracker.snapshot`.
    Direct construction is supported but the helper sets the
    invariants automatically.

    Attributes
    ----------
    counts
        4-tuple of 4-tuples of non-negative integers. Row index
        ``i`` is :data:`_LABEL_ORDER[i]` (the *from* label);
        column index ``j`` is the *to* label.
    laplace_smoothing
        Whether the row-normalised probability projection
        applies add-one smoothing. Smoothing only affects the
        :meth:`probability` view; the underlying counts are
        preserved.
    n_transitions
        Total number of recorded transitions across the whole
        matrix. Equals the number of :func:`observe` calls
        after the first.
    """

    counts: tuple[tuple[int, int, int, int], ...]
    laplace_smoothing: bool
    n_transitions: int

    def __post_init__(self) -> None:
        if len(self.counts) != 4:
            raise ValueError(f"counts must be 4 rows, got {len(self.counts)}")
        for i, row in enumerate(self.counts):
            if len(row) != 4:
                raise ValueError(f"row {i} must have 4 columns, got {len(row)}")
            for j, c in enumerate(row):
                if c < 0:
                    raise ValueError(f"counts[{i}][{j}] = {c} must be non-negative")
        total = sum(sum(row) for row in self.counts)
        if total != self.n_transitions:
            raise ValueError(
                f"n_transitions ({self.n_transitions}) does not match sum of counts ({total})"
            )

    def __repr__(self) -> str:  # aesthetic rich repr (HN28)
        if self.n_transitions == 0:
            return "RegimeTransitionMatrix[empty · n=0 · 4x4]"
        # Surface the dominant transition.
        best_count = -1
        best_pair: tuple[RegimeLabel, RegimeLabel] = (
            RegimeLabel.TRENDING,
            RegimeLabel.TRENDING,
        )
        for i, row in enumerate(self.counts):
            for j, c in enumerate(row):
                if c > best_count:
                    best_count = c
                    best_pair = (_LABEL_ORDER[i], _LABEL_ORDER[j])
        smooth_flag = "+laplace" if self.laplace_smoothing else "raw"
        return (
            f"RegimeTransitionMatrix[n={self.n_transitions} · "
            f"top={best_pair[0].name}→{best_pair[1].name}({best_count}) · "
            f"{smooth_flag}]"
        )

    def count(self, *, from_label: RegimeLabel, to_label: RegimeLabel) -> int:
        """Return the integer count of ``(from_label, to_label)`` transitions."""
        i = _LABEL_ORDER.index(from_label)
        j = _LABEL_ORDER.index(to_label)
        return self.counts[i][j]

    def row_total(self, label: RegimeLabel) -> int:
        """Return the sum of all transitions starting from ``label``."""
        i = _LABEL_ORDER.index(label)
        return sum(self.counts[i])

    def probability(self, *, from_label: RegimeLabel, to_label: RegimeLabel) -> float:
        """Row-normalised probability of ``(from → to)``.

        With ``laplace_smoothing=False`` (the default) and an
        unobserved ``from_label`` row, returns ``0.0`` (the row
        sum is zero so every cell is zero). The caller can use
        :meth:`row_total` to disambiguate "never observed" from
        "observed but never to this label".

        With ``laplace_smoothing=True``, every cell receives an
        add-one prior so unobserved transitions get probability
        ``1 / (row_total + 4)``.
        """
        i = _LABEL_ORDER.index(from_label)
        j = _LABEL_ORDER.index(to_label)
        row_sum = sum(self.counts[i])
        if self.laplace_smoothing:
            return (self.counts[i][j] + 1) / (row_sum + 4)
        if row_sum == 0:
            return 0.0
        return self.counts[i][j] / row_sum

    def as_text(self) -> str:
        """Render the matrix as a 4×4 ASCII grid for human review.

        Rows are labelled with the *from* label, columns with the
        *to* label. The grid uses the same canonical ordering as
        the underlying ``counts`` tuple.
        """
        labels = [label.name[:4] for label in _LABEL_ORDER]
        header = "from\\to  " + " ".join(f"{name:>5}" for name in labels)
        lines = [header, "-" * len(header)]
        for i, row in enumerate(self.counts):
            cells = " ".join(f"{c:>5}" for c in row)
            lines.append(f"{labels[i]:<8} {cells}")
        return "\n".join(lines)


class RegimeTransitionTracker:
    """Stateful first-order Markov tracker over regime label sequences.

    Parameters
    ----------
    laplace_smoothing
        When ``True``, the snapshot's :meth:`probability` view
        applies add-one smoothing. Default ``False`` — the raw
        empirical distribution.

    Notes
    -----
    Two trackers constructed with the same flag and fed the
    same :class:`RegimeState` sequence emit byte-identical
    matrices. There is no clock, no RNG, and no learned state.
    """

    __slots__ = ("_counts", "_last_label", "_n_transitions", "laplace_smoothing")

    def __init__(self, *, laplace_smoothing: bool = False) -> None:
        self.laplace_smoothing: bool = laplace_smoothing
        self._counts: list[list[int]] = [[0, 0, 0, 0] for _ in range(4)]
        self._last_label: RegimeLabel | None = None
        self._n_transitions: int = 0

    def reset(self) -> None:
        """Clear the matrix and the last-label memory."""
        self._counts = [[0, 0, 0, 0] for _ in range(4)]
        self._last_label = None
        self._n_transitions = 0

    @property
    def n_transitions(self) -> int:
        return self._n_transitions

    @property
    def last_label(self) -> RegimeLabel | None:
        return self._last_label

    def observe(self, state: RegimeState) -> TransitionEvent | None:
        """Record one regime observation.

        Parameters
        ----------
        state
            A :class:`~neurophase.analysis.regime.RegimeState`
            emitted by the G1 classifier (or any synthetic
            equivalent — the tracker only reads
            ``state.label``, ``state.tick_index``, and
            ``state.timestamp``).

        Returns
        -------
        TransitionEvent | None
            ``None`` on the very first call (no transition is
            observed yet — only the initial label is recorded).
            On every subsequent call, a frozen
            :class:`TransitionEvent` describing the recorded
            ``(from, to)`` pair.
        """
        new_label = state.label
        if self._last_label is None:
            self._last_label = new_label
            return None

        i = _LABEL_ORDER.index(self._last_label)
        j = _LABEL_ORDER.index(new_label)
        self._counts[i][j] += 1
        self._n_transitions += 1

        event = TransitionEvent(
            from_label=self._last_label,
            to_label=new_label,
            tick_index=state.tick_index,
            timestamp=state.timestamp,
            is_self_loop=self._last_label is new_label,
            observed_count=self._counts[i][j],
        )
        self._last_label = new_label
        return event

    def snapshot(self) -> RegimeTransitionMatrix:
        """Return a frozen copy of the current matrix."""
        counts_tuple: tuple[tuple[int, int, int, int], ...] = tuple(
            (row[0], row[1], row[2], row[3]) for row in self._counts
        )
        return RegimeTransitionMatrix(
            counts=counts_tuple,
            laplace_smoothing=self.laplace_smoothing,
            n_transitions=self._n_transitions,
        )

    def predict_next(self, label: RegimeLabel) -> tuple[RegimeLabel, float]:
        """Return the empirical mode of ``P(next | label)`` and its probability.

        The mode is the column with the highest count in the
        ``label`` row of the snapshot. Ties are broken by
        canonical declaration order (TRENDING > COMPRESSING >
        REVERTING > CHAOTIC).

        Raises
        ------
        InsufficientHistoryError
            If the row for ``label`` has zero total observations.
            The tracker refuses to fall back on a uniform
            distribution silently — see the class docstring.
        """
        i = _LABEL_ORDER.index(label)
        row = self._counts[i]
        row_total = sum(row)
        if row_total == 0:
            raise InsufficientHistoryError(f"no transitions observed from {label.name} yet")

        best_idx = 0
        best_count = row[0]
        for j in range(1, 4):
            if row[j] > best_count:
                best_count = row[j]
                best_idx = j

        prob = best_count / row_total
        return _LABEL_ORDER[best_idx], prob
