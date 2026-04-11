"""F2 — replay engine for decision trace ledgers.

Given an append-only ledger produced by
:class:`~neurophase.audit.decision_ledger.DecisionTraceLedger` and
the original input stream, this module replays the pipeline against
the same inputs and verifies that the replay reproduces the ledger
**byte-for-byte** on disk. This is the complement to F1 (ledger
persistence) and F3 (same-input → same-decision certification): F2
turns those two guarantees into an **incident-postmortem tool**.

Why this matters
----------------

F1 proves that the ledger cannot be tampered without ``verify_ledger``
noticing. F3 proves that the same seeds + same inputs produce
byte-identical ledgers across two independent runs of the same
configuration. F2 stitches the two together: given a real ledger file
and the original input trace, you can reconstruct the pipeline's
decisions and **prove that the stored ledger is the output the
current code produces today**. That is the minimal precondition for
any postmortem — without it, "we replayed the failure" is just
hand-waving.

Contract
--------

The replay engine is deliberately **declarative**, not
interpretive. The caller supplies:

1. The original input stream as an iterable of
   :class:`~neurophase.runtime.pipeline.ReplayInput` tuples.
2. The pipeline configuration the original run used (the same
   ``PipelineConfig`` — if the config changed since the ledger
   was written, the replay will not byte-match).
3. The path to the ledger file to replay against.

The engine:

1. Builds a fresh :class:`~neurophase.runtime.pipeline.StreamingPipeline`
   from the supplied config, writing to a scratch file (never the
   original ledger — side-effects never leak onto disk).
2. Feeds every ``ReplayInput`` through ``tick``.
3. Compares the scratch ledger file to the original, byte-for-byte.
4. Returns a frozen :class:`ReplayResult` with:

   * ``ok`` — ``True`` iff the bytes match exactly.
   * ``original_tip_hash`` / ``replayed_tip_hash`` — the last
     record hashes from each file. When ``ok=True`` these are
     identical; when ``ok=False`` the first divergence is
     surfaced.
   * ``first_divergent_index`` — position of the first divergent
     record, or ``None`` when the replay matched.
   * ``n_records`` — total records in the replayed ledger.
   * ``scratch_path`` — absolute path to the scratch replay file
     (caller may inspect or delete).

What F2 does NOT do
-------------------

* It does **not** replay against a *different* configuration and
  expect agreement. A different config produces a different
  ``parameter_fingerprint`` and therefore different
  ``record_hash`` values by construction (HN6). The caller is
  responsible for supplying the same config.
* It does **not** re-verify the ledger hash chain — that is F1's
  job (``verify_ledger``) and F2 would duplicate the work.
* It does **not** write to the original ledger file. Replay is
  strictly non-destructive.
* It does **not** interpret the ledger contents — replay is a
  byte-level operation. If the production code changes and the
  replay diverges, F2 reports the divergence but does not try to
  explain why.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from neurophase.audit.decision_ledger import verify_ledger
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)


@dataclass(frozen=True)
class ReplayInput:
    """One tick input for the replay engine.

    Mirrors the :meth:`StreamingPipeline.tick` signature exactly —
    the replay engine threads these tuples through ``tick`` in
    order.
    """

    timestamp: float
    R: float | None
    delta: float | None = None
    reference_now: float | None = None


@dataclass(frozen=True, repr=False)
class ReplayResult:
    """Immutable outcome of a replay run."""

    ok: bool
    n_records: int
    original_tip_hash: str | None
    replayed_tip_hash: str | None
    first_divergent_index: int | None
    scratch_path: Path
    reason: str

    def __repr__(self) -> str:  # aesthetic rich repr (HN22)
        flag = "✓ ok" if self.ok else "✗ diverged"
        parts = [flag, f"n={self.n_records}"]
        if self.original_tip_hash is not None:
            parts.append(f"tip={self.original_tip_hash[:8]}…")
        if not self.ok and self.first_divergent_index is not None:
            parts.append(f"diverge@{self.first_divergent_index}")
        return "ReplayResult[" + " · ".join(parts) + "]"


def replay_ledger(
    *,
    original_path: Path | str,
    config: PipelineConfig,
    inputs: Iterable[ReplayInput],
    scratch_path: Path | str,
) -> ReplayResult:
    """Replay a decision trace ledger and verify byte-identical output.

    Parameters
    ----------
    original_path
        Path to the original :class:`DecisionTraceLedger` file.
    config
        The exact :class:`PipelineConfig` the original run used.
        Must have ``ledger_path`` pointing at ``scratch_path`` — if
        the caller supplies a config whose ``ledger_path`` points
        at ``original_path``, the replay would overwrite the
        original, which is forbidden. This contract is enforced
        below.
    inputs
        Iterable of :class:`ReplayInput` tuples. The replay engine
        consumes the iterable once.
    scratch_path
        Destination for the scratch replay ledger. Must not collide
        with ``original_path``. Parent directory must exist.

    Returns
    -------
    ReplayResult
        Frozen outcome. ``ok=True`` iff the scratch file is
        byte-identical to the original.

    Raises
    ------
    ValueError
        If ``original_path`` does not exist, if ``scratch_path``
        collides with ``original_path``, or if the config's
        ``ledger_path`` is not the scratch path.
    """
    original = Path(original_path).resolve()
    scratch = Path(scratch_path).resolve()

    if not original.is_file():
        raise ValueError(f"original ledger not found at {original}")
    if original == scratch:
        raise ValueError(
            "scratch_path must differ from original_path; replay is non-destructive by contract"
        )
    if config.ledger_path is None:
        raise ValueError("PipelineConfig.ledger_path must be set for replay")
    if Path(config.ledger_path).resolve() != scratch:
        raise ValueError(
            f"config.ledger_path {config.ledger_path!r} must equal scratch_path {scratch!r}"
        )

    # Verify the original ledger is intact before we replay against
    # it — a corrupted original cannot yield a meaningful comparison.
    original_verification = verify_ledger(original)
    if not original_verification.ok:
        return ReplayResult(
            ok=False,
            n_records=0,
            original_tip_hash=None,
            replayed_tip_hash=None,
            first_divergent_index=original_verification.first_broken_index,
            scratch_path=scratch,
            reason=(
                f"original ledger failed verification before replay: {original_verification.reason}"
            ),
        )

    # Ensure the scratch file does not exist before we start so the
    # replay ledger chain starts at GENESIS_HASH, not a resumed tip.
    if scratch.exists():
        scratch.unlink()

    pipeline = StreamingPipeline(config)
    frames: list[DecisionFrame] = []
    for inp in inputs:
        frames.append(
            pipeline.tick(
                timestamp=inp.timestamp,
                R=inp.R,
                delta=inp.delta,
                reference_now=inp.reference_now,
            )
        )

    original_bytes = original.read_bytes()
    replayed_bytes = scratch.read_bytes()

    if original_bytes == replayed_bytes:
        replay_verification = verify_ledger(scratch)
        tip = (
            frames[-1].ledger_record.record_hash
            if frames and frames[-1].ledger_record is not None
            else None
        )
        return ReplayResult(
            ok=True,
            n_records=replay_verification.n_records,
            original_tip_hash=tip,
            replayed_tip_hash=tip,
            first_divergent_index=None,
            scratch_path=scratch,
            reason=(
                f"replay matches original byte-for-byte ({replay_verification.n_records} records)"
            ),
        )

    # Bytes diverge — find the first record that differs and report it.
    first_divergent = _first_divergent_record_index(original_bytes, replayed_bytes)
    original_ok = verify_ledger(original)
    replay_ok = verify_ledger(scratch)
    return ReplayResult(
        ok=False,
        n_records=replay_ok.n_records,
        original_tip_hash=_tip_hash_of(original_bytes),
        replayed_tip_hash=_tip_hash_of(replayed_bytes),
        first_divergent_index=first_divergent,
        scratch_path=scratch,
        reason=(
            f"replay diverges from original at record index {first_divergent}; "
            f"original verification: {original_ok.reason}; "
            f"replay verification: {replay_ok.reason}"
        ),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _first_divergent_record_index(original: bytes, replayed: bytes) -> int | None:
    """Return the index of the first record that differs, or ``None``."""
    orig_lines = original.splitlines()
    repl_lines = replayed.splitlines()
    for i, (o, r) in enumerate(zip(orig_lines, repl_lines, strict=False)):
        if o != r:
            return i
    if len(orig_lines) != len(repl_lines):
        return min(len(orig_lines), len(repl_lines))
    return None


def _tip_hash_of(payload: bytes) -> str | None:
    """Extract the ``record_hash`` of the last non-empty record."""
    import json

    lines = [line for line in payload.splitlines() if line.strip()]
    if not lines:
        return None
    last = json.loads(lines[-1])
    value = last.get("record_hash")
    if value is None:
        return None
    return str(value)
