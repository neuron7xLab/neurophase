"""Decision trace ledger — append-only, SHA256-chained audit log.

Stores one :class:`DecisionTraceRecord` per line in a JSONL file.
Each record carries its own SHA256 hash over the previous record's
hash and its own canonical JSON encoding. This yields a tamper-
evident chain: any post-hoc modification of any record breaks the
hash of every subsequent record, and :func:`verify_ledger` will
surface the first broken index.

Design notes
------------

* **JSONL** (one JSON object per line) — append-only, human-readable,
  streamable. No binary formats, no database coupling.
* **Canonical JSON** — the hash is computed over
  ``json.dumps(payload, sort_keys=True, separators=(",", ":"),
  ensure_ascii=False, default=str)``. This makes the hash
  deterministic and platform-independent.
* **Stateless verification** — :func:`verify_ledger` opens the file
  and walks it without any shared state with the producer. Same
  bytes → same verification result.
* **Parameter fingerprint** — each record carries the SHA256 of the
  parameter dict the gate was configured with. Two decisions made
  under different parameters are distinguishable in the ledger even
  if every other field is identical.
* **No SciPy, no pickle, no yaml.** Only ``hashlib`` + ``json``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

#: All-zero SHA256 — the canonical predecessor for the first record.
GENESIS_HASH: Final[str] = "0" * 64


class LedgerError(ValueError):
    """Raised on ledger schema mismatch or tamper detection."""


@dataclass(frozen=True)
class DecisionTraceRecord:
    """One append-only row in the decision ledger.

    All fields are JSON-serializable. Python ``None`` is preserved,
    and complex objects are stringified by the default encoder.

    Attributes
    ----------
    index
        Monotonic zero-based position of the record in the ledger.
    timestamp
        Caller-supplied timestamp (seconds since an arbitrary epoch).
    gate_state
        String name of the ``GateState`` enum member (e.g. ``"READY"``).
    execution_allowed
        Boolean flag from the gate decision.
    R
        The order parameter value at decision time. ``None`` when the
        gate was in ``SENSOR_ABSENT`` / ``DEGRADED``.
    threshold
        The threshold value used by the gate.
    reason
        Human-readable reason string copied verbatim from the decision.
    parameter_fingerprint
        SHA256 of the canonical-JSON encoding of the parameter dict.
        Two records produced under different parameters have
        different fingerprints.
    extras
        Optional dict of extra provenance (stillness state, delta,
        temporal quality, ...). Must be JSON-serializable.
    prev_hash
        Hex-encoded SHA256 of the previous record; ``GENESIS_HASH``
        for the first.
    record_hash
        Hex-encoded SHA256 over ``prev_hash + canonical(payload)``.
    """

    index: int
    timestamp: float
    gate_state: str
    execution_allowed: bool
    R: float | None
    threshold: float
    reason: str
    parameter_fingerprint: str
    extras: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    record_hash: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        """Return the record as a plain dict suitable for json.dumps."""
        return asdict(self)


@dataclass(frozen=True)
class LedgerVerification:
    """Outcome of :func:`verify_ledger`.

    Attributes
    ----------
    ok
        ``True`` iff every record in the file verifies against its
        predecessor.
    n_records
        Total number of records read.
    first_broken_index
        Index of the first tampered record, or ``None`` if ``ok``.
    reason
        Short explanation for the result.
    """

    ok: bool
    n_records: int
    first_broken_index: int | None
    reason: str


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class DecisionTraceLedger:
    """Append-only, SHA256-chained decision ledger.

    Parameters
    ----------
    path
        Filesystem path for the JSONL ledger. Parent directories must
        already exist (the ledger will not create them). The file is
        opened in append-mode by :meth:`append`.
    parameter_fingerprint
        Fingerprint string embedded in every appended record. Compute
        it externally via :func:`fingerprint_parameters` to pin the
        gate configuration.
    """

    __slots__ = ("_last_hash", "_n_appended", "parameter_fingerprint", "path")

    def __init__(self, path: Path | str, parameter_fingerprint: str) -> None:
        self.path: Path = Path(path)
        self.parameter_fingerprint: str = parameter_fingerprint
        self._last_hash: str = GENESIS_HASH
        self._n_appended: int = 0
        # If the ledger already exists, walk it to restore _last_hash and
        # _n_appended so appends continue the chain instead of breaking it.
        if self.path.is_file():
            verification = verify_ledger(self.path)
            if not verification.ok:
                raise LedgerError(
                    f"existing ledger at {self.path} is broken at index "
                    f"{verification.first_broken_index}: {verification.reason}"
                )
            # Walk the file one more time to recover the last hash and
            # the next index. Cheap because verify_ledger already read it.
            with self.path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    self._last_hash = payload["record_hash"]
                    self._n_appended = payload["index"] + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        timestamp: float,
        gate_state: str,
        execution_allowed: bool,
        R: float | None,
        threshold: float,
        reason: str,
        extras: dict[str, Any] | None = None,
    ) -> DecisionTraceRecord:
        """Append a new record to the ledger and return it.

        The caller supplies the raw fields; the ledger computes the
        hash, updates ``_last_hash``, and writes the canonical JSON
        line atomically to the end of the file.
        """
        extras_dict: dict[str, Any] = dict(extras) if extras else {}
        R_val: float | None = None if R is None else float(R)
        payload: dict[str, Any] = {
            "index": self._n_appended,
            "timestamp": float(timestamp),
            "gate_state": str(gate_state),
            "execution_allowed": bool(execution_allowed),
            "R": R_val,
            "threshold": float(threshold),
            "reason": str(reason),
            "parameter_fingerprint": self.parameter_fingerprint,
            "extras": extras_dict,
            "prev_hash": self._last_hash,
        }
        record_hash = _compute_record_hash(payload)
        payload["record_hash"] = record_hash

        record = DecisionTraceRecord(
            index=self._n_appended,
            timestamp=float(timestamp),
            gate_state=str(gate_state),
            execution_allowed=bool(execution_allowed),
            R=R_val,
            threshold=float(threshold),
            reason=str(reason),
            parameter_fingerprint=self.parameter_fingerprint,
            extras=extras_dict,
            prev_hash=self._last_hash,
            record_hash=record_hash,
        )

        line = _canonical_json(payload) + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)

        self._last_hash = record_hash
        self._n_appended += 1
        return record

    @property
    def n_appended(self) -> int:
        """Total number of records written since construction."""
        return self._n_appended

    @property
    def last_hash(self) -> str:
        """Current tip of the hash chain."""
        return self._last_hash


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_ledger(path: Path | str) -> LedgerVerification:
    """Verify a ledger file's hash chain.

    Returns an immutable :class:`LedgerVerification` describing the
    outcome. On the first tampered record, walks stops and records the
    index in ``first_broken_index``.
    """
    resolved = Path(path)
    if not resolved.is_file():
        return LedgerVerification(
            ok=False,
            n_records=0,
            first_broken_index=None,
            reason=f"ledger file not found at {resolved}",
        )

    prev_hash = GENESIS_HASH
    n = 0
    with resolved.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                return LedgerVerification(
                    ok=False,
                    n_records=n,
                    first_broken_index=n,
                    reason=f"line {lineno} is not valid JSON: {exc}",
                )

            required_fields = (
                "index",
                "timestamp",
                "gate_state",
                "execution_allowed",
                "R",
                "threshold",
                "reason",
                "parameter_fingerprint",
                "extras",
                "prev_hash",
                "record_hash",
            )
            for key in required_fields:
                if key not in payload:
                    return LedgerVerification(
                        ok=False,
                        n_records=n,
                        first_broken_index=n,
                        reason=f"record {n} missing field {key!r}",
                    )

            if payload["index"] != n:
                return LedgerVerification(
                    ok=False,
                    n_records=n,
                    first_broken_index=n,
                    reason=(f"record {n} has wrong index {payload['index']}"),
                )

            if payload["prev_hash"] != prev_hash:
                return LedgerVerification(
                    ok=False,
                    n_records=n,
                    first_broken_index=n,
                    reason=(
                        f"record {n} prev_hash mismatch: expected {prev_hash[:12]}…, "
                        f"got {str(payload['prev_hash'])[:12]}…"
                    ),
                )

            recomputed = _compute_record_hash(
                {k: v for k, v in payload.items() if k != "record_hash"}
            )
            if recomputed != payload["record_hash"]:
                return LedgerVerification(
                    ok=False,
                    n_records=n,
                    first_broken_index=n,
                    reason=(
                        f"record {n} hash mismatch: expected {recomputed[:12]}…, "
                        f"got {str(payload['record_hash'])[:12]}…"
                    ),
                )

            prev_hash = payload["record_hash"]
            n += 1

    return LedgerVerification(
        ok=True,
        n_records=n,
        first_broken_index=None,
        reason=f"verified {n} record(s)",
    )


# ---------------------------------------------------------------------------
# Public helper: parameter fingerprinting
# ---------------------------------------------------------------------------


def fingerprint_parameters(params: dict[str, Any]) -> str:
    """Return a deterministic SHA256 fingerprint of a parameter dict.

    The fingerprint is the hex SHA256 of the canonical-JSON encoding
    of ``params``. Two dicts that differ by key order or by value
    representation (``True`` vs ``true``, ``1`` vs ``1.0`` only if
    the serializer distinguishes them) produce different fingerprints.
    """
    return hashlib.sha256(_canonical_json(params).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON serialization suitable for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _compute_record_hash(payload: dict[str, Any]) -> str:
    """Compute the SHA256 hash of a record without the ``record_hash`` field.

    The input must contain every required field **except** ``record_hash``.
    """
    canonical = _canonical_json(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
