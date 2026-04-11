"""K1 — contract tests for the SessionManifest provenance lock.

This test file is the HN25 binding. It locks in:

1. **Construction is deterministic.** Same inputs →
   byte-identical manifest including identical ``manifest_hash``.
2. **Self-validating hash.** Single-bit corruption of the
   manifest file is detected at :meth:`SessionManifest.load` time
   via the recomputed manifest hash.
3. **Schema-version pinning.** A manifest written under the
   current schema and tampered to claim a different version
   raises :class:`ManifestError` on load.
4. **Cross-file ledger integrity.**
   :meth:`SessionManifest.verify_against_ledger` re-walks the
   bound ledger and rejects a tampered ledger or a ledger whose
   tip has drifted from the recorded snapshot.
5. **JSON round-trip is exact.** ``SessionManifest.write`` then
   ``SessionManifest.load`` reconstructs every field bit-perfect.
6. **Construction-time validation** rejects malformed
   fingerprints, negative tick counts, end_ts < start_ts.
7. **Frozen dataclass.** Reassigning a field raises.
8. **Aesthetic** rich __repr__ in canonical HN22/HN25 design
   language.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.audit.decision_ledger import (
    GENESIS_HASH,
    DecisionTraceLedger,
    fingerprint_parameters,
)
from neurophase.audit.session_manifest import (
    EMPTY_LEDGER_TIP,
    SESSION_MANIFEST_SCHEMA_VERSION,
    ManifestError,
    SessionManifest,
    build_session_manifest,
    compute_dataset_fingerprint,
)

# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _make_ledger_with_records(path: Path, n: int) -> tuple[str, str]:
    """Create a ledger at ``path`` with ``n`` records.

    Returns ``(parameter_fingerprint, last_record_hash)``.
    """
    fp = fingerprint_parameters({"threshold": 0.65, "n": n})
    ledger = DecisionTraceLedger(path, fp)
    last_hash = GENESIS_HASH
    for i in range(n):
        record = ledger.append(
            timestamp=float(i) * 0.1,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason=f"synthetic tick {i}",
            extras={"tick_index": i},
        )
        last_hash = record.record_hash
    return fp, last_hash


def _build(
    *,
    tmp_path: Path,
    n_records: int = 3,
    start_ts: float = 0.0,
    end_ts: float = 5.0,
    code_commit: str = "",
    host: str = "",
    notes: str = "",
) -> tuple[SessionManifest, Path, str]:
    """Build a manifest backed by a real ledger. Returns
    ``(manifest, ledger_path, parameter_fingerprint)``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger_path = tmp_path / "ledger.jsonl"
    fp, tip = _make_ledger_with_records(ledger_path, n_records)
    dataset_fp = compute_dataset_fingerprint(b"synthetic dataset bytes")
    manifest = build_session_manifest(
        start_ts=start_ts,
        end_ts=end_ts,
        parameter_fingerprint=fp,
        dataset_fingerprint=dataset_fp,
        ledger_path=ledger_path,
        ledger_tip_hash=tip,
        n_ticks=n_records,
        code_commit=code_commit,
        host=host,
        notes=notes,
    )
    return manifest, ledger_path, fp


# ---------------------------------------------------------------------------
# 1. Determinism — same inputs → byte-identical manifest.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_inputs_yield_identical_manifest(self, tmp_path: Path) -> None:
        m1, _, _ = _build(tmp_path=tmp_path / "a")
        m2, _, _ = _build(tmp_path=tmp_path / "b")
        # Different ledger paths → different manifest_hash by design.
        # Build twice in the same dir to get a stable comparison.
        m3, _, _ = _build(tmp_path=tmp_path / "c")
        # The manifest_hash should be derivable from inputs alone.
        # Build the *same* manifest twice via direct construction
        # with identical fields and check byte-identity.
        from neurophase.audit.session_manifest import build_session_manifest

        same1 = build_session_manifest(
            start_ts=m3.start_ts,
            end_ts=m3.end_ts,
            parameter_fingerprint=m3.parameter_fingerprint,
            dataset_fingerprint=m3.dataset_fingerprint,
            ledger_path=m3.ledger_path,
            ledger_tip_hash=m3.ledger_tip_hash,
            n_ticks=m3.n_ticks,
        )
        same2 = build_session_manifest(
            start_ts=m3.start_ts,
            end_ts=m3.end_ts,
            parameter_fingerprint=m3.parameter_fingerprint,
            dataset_fingerprint=m3.dataset_fingerprint,
            ledger_path=m3.ledger_path,
            ledger_tip_hash=m3.ledger_tip_hash,
            n_ticks=m3.n_ticks,
        )
        assert same1 == same2
        assert same1.manifest_hash == same2.manifest_hash
        assert same1.run_id == same2.run_id
        # m1 and m2 differ only in ledger_path → different run_id.
        assert m1.run_id != m2.run_id

    def test_run_id_is_16_hex_chars(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        assert len(manifest.run_id) == 16
        int(manifest.run_id, 16)  # parses as hex


# ---------------------------------------------------------------------------
# 2. Self-validating hash — corruption is detected.
# ---------------------------------------------------------------------------


class TestSelfValidatingHash:
    def test_unmodified_file_loads_cleanly(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        manifest_path = tmp_path / "session.json"
        manifest.write(manifest_path)
        loaded = SessionManifest.load(manifest_path)
        assert loaded == manifest

    def test_single_bit_flip_detected(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        manifest_path = tmp_path / "session.json"
        manifest.write(manifest_path)
        text = manifest_path.read_text()
        # Flip the n_ticks field from 3 to 4 by direct text edit.
        tampered = text.replace('"n_ticks":3', '"n_ticks":4')
        assert tampered != text, "tamper failed to land"
        manifest_path.write_text(tampered)
        with pytest.raises(ManifestError, match="manifest_hash mismatch"):
            SessionManifest.load(manifest_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="does not exist"):
            SessionManifest.load(tmp_path / "missing.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not valid json")
        with pytest.raises(ManifestError, match="not valid JSON"):
            SessionManifest.load(bad)


# ---------------------------------------------------------------------------
# 3. Schema version pinning.
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_wrong_schema_version_rejected_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="schema_version"):
            SessionManifest(
                schema_version=999,
                run_id="abcdef0123456789",
                start_ts=0.0,
                end_ts=1.0,
                parameter_fingerprint="a" * 64,
                dataset_fingerprint="b" * 64,
                ledger_path=str(tmp_path / "x.jsonl"),
                ledger_tip_hash="c" * 64,
                n_ticks=0,
                manifest_hash="d" * 64,
            )

    def test_current_schema_version_constant(self) -> None:
        assert SESSION_MANIFEST_SCHEMA_VERSION == 1


# ---------------------------------------------------------------------------
# 4. Cross-file ledger integrity.
# ---------------------------------------------------------------------------


class TestLedgerIntegrity:
    def test_clean_ledger_verifies(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        verification = manifest.verify_against_ledger()
        assert verification.ok

    def test_tampered_ledger_detected(self, tmp_path: Path) -> None:
        manifest, ledger_path, _ = _build(tmp_path=tmp_path)
        # Tamper with the first record's reason — this propagates
        # through every subsequent hash, so verify_ledger fails.
        text = ledger_path.read_text()
        tampered = text.replace("synthetic tick 0", "TAMPERED tick 0")
        ledger_path.write_text(tampered)
        with pytest.raises(ManifestError, match="failed verification"):
            manifest.verify_against_ledger()

    def test_drifted_tip_detected(self, tmp_path: Path) -> None:
        """If new records are appended to the ledger after manifest
        write, the live tip diverges from the snapshot."""
        manifest, ledger_path, fp = _build(tmp_path=tmp_path, n_records=3)
        # Append one more record using the same parameter fingerprint.
        ledger = DecisionTraceLedger(ledger_path, fp)
        ledger.append(
            timestamp=10.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.95,
            threshold=0.65,
            reason="post-manifest tick",
            extras={"tick_index": 99},
        )
        with pytest.raises(ManifestError, match="tip drifted"):
            manifest.verify_against_ledger()

    def test_missing_ledger_path_detected(self, tmp_path: Path) -> None:
        manifest, ledger_path, _ = _build(tmp_path=tmp_path)
        ledger_path.unlink()
        with pytest.raises(ManifestError, match="does not exist"):
            manifest.verify_against_ledger()


# ---------------------------------------------------------------------------
# 5. JSON round-trip exactness.
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_write_then_load_is_identity(self, tmp_path: Path) -> None:
        original, _, _ = _build(
            tmp_path=tmp_path,
            code_commit="deadbeefcafebabe",
            host="ci-runner-42",
            notes="K1 test",
        )
        manifest_path = tmp_path / "session.json"
        original.write(manifest_path)
        loaded = SessionManifest.load(manifest_path)
        assert loaded.to_json_dict() == original.to_json_dict()

    def test_to_json_dict_is_flat(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        d = manifest.to_json_dict()
        for key, value in d.items():
            assert isinstance(value, (str, int, float, bool, type(None))), (
                f"key {key!r} has non-primitive value {value!r}"
            )


# ---------------------------------------------------------------------------
# 6. Construction-time validation rejects malformed inputs.
# ---------------------------------------------------------------------------


class TestConstructionValidation:
    def test_negative_n_ticks_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="n_ticks"):
            SessionManifest(
                schema_version=SESSION_MANIFEST_SCHEMA_VERSION,
                run_id="abcdef0123456789",
                start_ts=0.0,
                end_ts=1.0,
                parameter_fingerprint="a" * 64,
                dataset_fingerprint="b" * 64,
                ledger_path=str(tmp_path / "x.jsonl"),
                ledger_tip_hash="c" * 64,
                n_ticks=-1,
                manifest_hash="d" * 64,
            )

    def test_end_before_start_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="end_ts"):
            SessionManifest(
                schema_version=SESSION_MANIFEST_SCHEMA_VERSION,
                run_id="abcdef0123456789",
                start_ts=10.0,
                end_ts=5.0,
                parameter_fingerprint="a" * 64,
                dataset_fingerprint="b" * 64,
                ledger_path=str(tmp_path / "x.jsonl"),
                ledger_tip_hash="c" * 64,
                n_ticks=0,
                manifest_hash="d" * 64,
            )

    def test_malformed_parameter_fingerprint_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="parameter_fingerprint"):
            SessionManifest(
                schema_version=SESSION_MANIFEST_SCHEMA_VERSION,
                run_id="abcdef0123456789",
                start_ts=0.0,
                end_ts=1.0,
                parameter_fingerprint="not-hex",
                dataset_fingerprint="b" * 64,
                ledger_path=str(tmp_path / "x.jsonl"),
                ledger_tip_hash="c" * 64,
                n_ticks=0,
                manifest_hash="d" * 64,
            )

    def test_empty_run_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="run_id"):
            SessionManifest(
                schema_version=SESSION_MANIFEST_SCHEMA_VERSION,
                run_id="",
                start_ts=0.0,
                end_ts=1.0,
                parameter_fingerprint="a" * 64,
                dataset_fingerprint="b" * 64,
                ledger_path=str(tmp_path / "x.jsonl"),
                ledger_tip_hash="c" * 64,
                n_ticks=0,
                manifest_hash="d" * 64,
            )


# ---------------------------------------------------------------------------
# 7. Frozen dataclass.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_manifest_is_frozen(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            manifest.n_ticks = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. Empty session — zero ticks must round-trip.
# ---------------------------------------------------------------------------


class TestEmptySession:
    def test_zero_records_uses_empty_ledger_tip(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "empty.jsonl"
        ledger_path.write_text("")
        fp = fingerprint_parameters({"threshold": 0.65, "n": 0})
        manifest = build_session_manifest(
            start_ts=0.0,
            end_ts=0.0,
            parameter_fingerprint=fp,
            dataset_fingerprint=compute_dataset_fingerprint(b""),
            ledger_path=ledger_path,
            ledger_tip_hash=EMPTY_LEDGER_TIP,
            n_ticks=0,
        )
        assert manifest.n_ticks == 0
        assert manifest.ledger_tip_hash == EMPTY_LEDGER_TIP
        # Round-trip works.
        path = tmp_path / "empty_session.json"
        manifest.write(path)
        loaded = SessionManifest.load(path)
        assert loaded == manifest


# ---------------------------------------------------------------------------
# 9. Aesthetic rich __repr__ — HN22 / HN25 design language.
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_format(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path, code_commit="abcdef1234567890")
        r = repr(manifest)
        assert r.startswith("SessionManifest[")
        assert "run_id=" in r
        assert "n=3" in r
        assert "Δt=" in r
        assert "abcdef12" in r  # code commit prefix
        assert "tip=" in r

    def test_repr_with_no_commit(self, tmp_path: Path) -> None:
        manifest, _, _ = _build(tmp_path=tmp_path)
        r = repr(manifest)
        assert "commit=—" in r


# ---------------------------------------------------------------------------
# 10. Dataset fingerprint is deterministic.
# ---------------------------------------------------------------------------


class TestDatasetFingerprint:
    def test_same_bytes_same_fingerprint(self) -> None:
        a = compute_dataset_fingerprint(b"hello world")
        b = compute_dataset_fingerprint(b"hello world")
        assert a == b
        assert len(a) == 64

    def test_different_bytes_different_fingerprint(self) -> None:
        a = compute_dataset_fingerprint(b"hello")
        b = compute_dataset_fingerprint(b"world")
        assert a != b
