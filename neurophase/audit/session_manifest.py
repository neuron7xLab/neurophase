"""K1 — session manifest schema (dataset & run provenance lock).

A :class:`SessionManifest` is the **outermost provenance envelope**
of a single ``neurophase`` session. It binds together:

* the dataset that was fed in (via a caller-supplied SHA256
  fingerprint of the input bytes),
* the parameter set the pipeline ran with (via the existing
  :func:`~neurophase.audit.decision_ledger.fingerprint_parameters`
  helper, the same fingerprint baked into every ledger record),
* the append-only decision ledger that the session produced (via
  the path *and* the SHA256 of the ledger tip — proof of "what
  we audited" at session-end time),
* the optional code commit + host identifier so a future reviewer
  can replay the run on the exact same source state,
* a stable, deterministic ``run_id`` derived from the above so
  two manifests built from the same inputs are byte-identical.

What the manifest is NOT
------------------------

* It is **not** a substitute for the decision ledger. The ledger
  is the load-bearing per-tick audit; the manifest is the
  outermost dataset/run binding.
* It is **not** a substitute for the parameter fingerprint baked
  into ledger records. The manifest stores the same fingerprint
  redundantly so a reviewer who reads the manifest alone can
  still answer "what config was this run on?" without opening
  the ledger.
* It is **not** mutable. Every field is frozen. There is no
  ``update_end_ts(...)`` method — close the manifest by
  constructing a new one.
* It does **not** auto-discover anything. The caller must supply
  the dataset fingerprint, the start/end timestamps, and the
  ledger path. K1's job is to bind them, not to invent them.

Contract
--------

* **Deterministic.** Same inputs → bit-identical manifest →
  bit-identical ``manifest_hash`` (the SHA256 of the manifest's
  canonical JSON projection).
* **JSON-safe.** :meth:`to_json_dict` is a flat projection — no
  nested dataclass objects. :meth:`write` / :meth:`load`
  round-trip through JSON byte-perfectly.
* **Tamper-evident.** The manifest carries its own
  ``manifest_hash``, computed over the canonical JSON of every
  other field. :meth:`load` recomputes the hash and raises
  :class:`ManifestError` on any mismatch — even a single bit
  flip in the file is detected at read time.
* **Ledger-aware.** :meth:`verify_against_ledger` re-runs
  :func:`~neurophase.audit.decision_ledger.verify_ledger` on
  the bound ledger path and checks that the live tip hash
  matches the ``ledger_tip_hash`` recorded in the manifest. A
  ledger that has been tampered with after the session closed
  surfaces here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from neurophase.audit.decision_ledger import (
    LedgerVerification,
    fingerprint_parameters,
    verify_ledger,
)

#: Schema version. Bump on breaking changes to the JSON layout.
SESSION_MANIFEST_SCHEMA_VERSION: Final[int] = 1

#: Sentinel for an empty / never-written ledger tip. Distinct
#: from ``DecisionTraceLedger.GENESIS_HASH`` so a reviewer can
#: tell "session produced zero records" apart from "session
#: produced one record whose prev_hash is the genesis".
EMPTY_LEDGER_TIP: Final[str] = "0" * 64

__all__ = [
    "EMPTY_LEDGER_TIP",
    "SESSION_MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "SessionManifest",
    "build_session_manifest",
    "compute_dataset_fingerprint",
]


class ManifestError(ValueError):
    """Raised on manifest schema mismatch, hash mismatch, or load failure."""


@dataclass(frozen=True, repr=False)
class SessionManifest:
    """Frozen, JSON-safe outermost provenance envelope of one session.

    Construct via :func:`build_session_manifest`. Direct
    construction is supported but the helper sets the schema
    version and recomputes the manifest hash automatically.

    Attributes
    ----------
    schema_version
        Integer schema version of this manifest. Pinned to
        :data:`SESSION_MANIFEST_SCHEMA_VERSION` at write time.
    run_id
        Stable, deterministic identifier of this session. Equal to
        the first 16 hex chars of the SHA256 of
        ``parameter_fingerprint || dataset_fingerprint ||
        ledger_path || start_ts``. Two manifests built from the
        same inputs share a ``run_id``.
    start_ts
        Caller-supplied wall-clock start timestamp (seconds since
        epoch).
    end_ts
        Caller-supplied wall-clock end timestamp (seconds since
        epoch). Must be ≥ ``start_ts``.
    parameter_fingerprint
        SHA256 of the canonical JSON of the parameter dict the
        pipeline ran with. Equal to
        ``StreamingPipeline.parameter_fingerprint``.
    dataset_fingerprint
        SHA256 of the canonical bytes of the input dataset.
        Compute via :func:`compute_dataset_fingerprint` or
        supply directly.
    ledger_path
        String form of the absolute path to the decision ledger
        file. We store the path as a string because :class:`Path`
        does not survive a JSON round-trip.
    ledger_tip_hash
        Hex-encoded SHA256 of the last :class:`DecisionTraceRecord`
        in the bound ledger at session-close time, or
        :data:`EMPTY_LEDGER_TIP` for an empty session.
    n_ticks
        Number of decisions emitted during the session.
    code_commit
        Optional git commit SHA the session ran on. Empty string
        when the caller could not determine it.
    host
        Optional non-PII host identifier (hostname, container id,
        or any free-form string). Empty string when not supplied.
    notes
        Optional free-form notes the caller wants attached.
    manifest_hash
        SHA256 of the canonical JSON projection of every field
        above. Self-validating: :meth:`load` recomputes and
        raises on mismatch.
    """

    schema_version: int
    run_id: str
    start_ts: float
    end_ts: float
    parameter_fingerprint: str
    dataset_fingerprint: str
    ledger_path: str
    ledger_tip_hash: str
    n_ticks: int
    code_commit: str = ""
    host: str = ""
    notes: str = ""
    manifest_hash: str = field(default="")

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.schema_version != SESSION_MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"schema_version mismatch: expected "
                f"{SESSION_MANIFEST_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.end_ts < self.start_ts:
            raise ManifestError(f"end_ts ({self.end_ts}) must be >= start_ts ({self.start_ts})")
        if self.n_ticks < 0:
            raise ManifestError(f"n_ticks must be >= 0, got {self.n_ticks}")
        if not _looks_like_sha256_hex(self.parameter_fingerprint):
            raise ManifestError(
                f"parameter_fingerprint is not a 64-char hex SHA256: {self.parameter_fingerprint!r}"
            )
        if not _looks_like_sha256_hex(self.dataset_fingerprint):
            raise ManifestError(
                f"dataset_fingerprint is not a 64-char hex SHA256: {self.dataset_fingerprint!r}"
            )
        if not _looks_like_sha256_hex(self.ledger_tip_hash):
            raise ManifestError(
                f"ledger_tip_hash is not a 64-char hex SHA256: {self.ledger_tip_hash!r}"
            )
        if not self.run_id:
            raise ManifestError("run_id must be non-empty")

    def __repr__(self) -> str:  # aesthetic rich repr (HN25)
        commit = (self.code_commit[:8] if self.code_commit else "—") or "—"
        return (
            f"SessionManifest[run_id={self.run_id} · "
            f"n={self.n_ticks} · "
            f"Δt={self.end_ts - self.start_ts:.2f}s · "
            f"commit={commit} · "
            f"tip={self.ledger_tip_hash[:8]}…]"
        )

    # ------------------------------------------------------------------
    # JSON projection / round-trip
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe projection — every value is a primitive."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "parameter_fingerprint": self.parameter_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "ledger_path": self.ledger_path,
            "ledger_tip_hash": self.ledger_tip_hash,
            "n_ticks": self.n_ticks,
            "code_commit": self.code_commit,
            "host": self.host,
            "notes": self.notes,
            "manifest_hash": self.manifest_hash,
        }

    def write(self, path: Path | str) -> Path:
        """Write the manifest to ``path`` as canonical JSON.

        Returns the resolved path. The output is byte-identical
        across runs for the same manifest — no insertion-order
        whims.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_canonical_json(self.to_json_dict()) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str) -> SessionManifest:
        """Load a manifest from ``path``, verifying the manifest hash.

        Raises
        ------
        ManifestError
            If the file is missing, malformed, schema-mismatched,
            or the recomputed manifest hash does not match the
            stored value (single-bit corruption is detected here).
        """
        p = Path(path)
        if not p.is_file():
            raise ManifestError(f"manifest file does not exist: {p}")
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ManifestError(f"manifest payload must be a dict, got {type(payload)}")
        try:
            manifest = cls(
                schema_version=int(payload["schema_version"]),
                run_id=str(payload["run_id"]),
                start_ts=float(payload["start_ts"]),
                end_ts=float(payload["end_ts"]),
                parameter_fingerprint=str(payload["parameter_fingerprint"]),
                dataset_fingerprint=str(payload["dataset_fingerprint"]),
                ledger_path=str(payload["ledger_path"]),
                ledger_tip_hash=str(payload["ledger_tip_hash"]),
                n_ticks=int(payload["n_ticks"]),
                code_commit=str(payload.get("code_commit", "")),
                host=str(payload.get("host", "")),
                notes=str(payload.get("notes", "")),
                manifest_hash=str(payload["manifest_hash"]),
            )
        except KeyError as exc:
            raise ManifestError(f"manifest missing required field: {exc}") from exc
        # Recompute and verify.
        recomputed = _compute_manifest_hash(manifest)
        if recomputed != manifest.manifest_hash:
            raise ManifestError(
                f"manifest_hash mismatch: stored={manifest.manifest_hash}, "
                f"recomputed={recomputed} (file has been modified)"
            )
        return manifest

    def verify_against_ledger(self) -> LedgerVerification:
        """Re-walk the bound ledger and assert its tip matches the manifest.

        Returns the :class:`LedgerVerification` from the underlying
        :func:`verify_ledger` call. Raises :class:`ManifestError`
        if the bound ledger's live tip does not match
        :attr:`ledger_tip_hash` — that is the load-bearing
        cross-file integrity check.
        """
        path = Path(self.ledger_path)
        if not path.is_file():
            raise ManifestError(f"ledger bound to manifest does not exist: {path}")
        verification = verify_ledger(path)
        if not verification.ok:
            raise ManifestError(f"bound ledger failed verification: {verification.reason}")
        live_tip = _read_live_ledger_tip(path)
        if live_tip != self.ledger_tip_hash:
            raise ManifestError(
                f"bound ledger tip drifted: manifest expects "
                f"{self.ledger_tip_hash[:16]}…, "
                f"ledger has {live_tip[:16]}…"
            )
        return verification


# ---------------------------------------------------------------------------
# Builders + helpers
# ---------------------------------------------------------------------------


def build_session_manifest(
    *,
    start_ts: float,
    end_ts: float,
    parameter_fingerprint: str,
    dataset_fingerprint: str,
    ledger_path: Path | str,
    ledger_tip_hash: str,
    n_ticks: int,
    code_commit: str = "",
    host: str = "",
    notes: str = "",
) -> SessionManifest:
    """Construct a fully-populated :class:`SessionManifest`.

    The helper sets ``schema_version`` to the current constant,
    derives ``run_id`` deterministically from the inputs, and
    computes the self-validating ``manifest_hash``. Two calls with
    the same inputs return byte-identical manifests.
    """
    ledger_path_str = (
        str(Path(ledger_path).resolve()) if Path(ledger_path).exists() else str(ledger_path)
    )
    run_id = _derive_run_id(
        parameter_fingerprint=parameter_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        ledger_path=ledger_path_str,
        start_ts=start_ts,
    )
    draft = SessionManifest(
        schema_version=SESSION_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        start_ts=start_ts,
        end_ts=end_ts,
        parameter_fingerprint=parameter_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        ledger_path=ledger_path_str,
        ledger_tip_hash=ledger_tip_hash,
        n_ticks=n_ticks,
        code_commit=code_commit,
        host=host,
        notes=notes,
        manifest_hash=_PLACEHOLDER_MANIFEST_HASH,
    )
    real_hash = _compute_manifest_hash(draft)
    # SessionManifest is frozen, so we cannot mutate `draft`. Build
    # a fresh manifest with the real hash. Use object.__setattr__
    # via dataclasses.replace.
    from dataclasses import replace

    return replace(draft, manifest_hash=real_hash)


def compute_dataset_fingerprint(data: bytes) -> str:
    """Return the hex SHA256 of an opaque dataset byte buffer.

    Use this when the caller has the raw input bytes available.
    For DataFrame-shaped inputs, the caller is responsible for
    serialising to a canonical byte form (e.g. ``df.to_parquet``,
    ``df.to_csv(...).encode()``) before calling.
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Re-export the helpers the manifest depends on so callers can
# build a manifest without importing from two places.
# ---------------------------------------------------------------------------

__all__.append("fingerprint_parameters")  # re-export from decision_ledger


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

#: Sentinel placeholder used while computing the real manifest
#: hash. Never persisted.
_PLACEHOLDER_MANIFEST_HASH: Final[str] = "f" * 64


def _looks_like_sha256_hex(value: str) -> bool:
    """Cheap shape check for a 64-char hex string."""
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _derive_run_id(
    *,
    parameter_fingerprint: str,
    dataset_fingerprint: str,
    ledger_path: str,
    start_ts: float,
) -> str:
    """Deterministic run id — first 16 hex of a salted SHA256."""
    payload = f"{parameter_fingerprint}|{dataset_fingerprint}|{ledger_path}|{start_ts!r}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _compute_manifest_hash(manifest: SessionManifest) -> str:
    """Hash every field except ``manifest_hash`` itself."""
    payload = manifest.to_json_dict()
    payload.pop("manifest_hash", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _read_live_ledger_tip(path: Path) -> str:
    """Return the ``record_hash`` of the last record in the ledger,
    or :data:`EMPTY_LEDGER_TIP` for an empty file.

    Used by :meth:`SessionManifest.verify_against_ledger` to
    cross-check the manifest's snapshot of the tip against the
    current state of the file.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return EMPTY_LEDGER_TIP
    last_line = text.strip().splitlines()[-1]
    try:
        record = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"ledger tail is not valid JSON: {exc}") from exc
    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str):
        raise ManifestError(f"ledger tail record missing record_hash field: {record}")
    return record_hash


# Re-export for convenience
__all__ = [
    "EMPTY_LEDGER_TIP",
    "SESSION_MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "SessionManifest",
    "build_session_manifest",
    "compute_dataset_fingerprint",
    "fingerprint_parameters",
]
