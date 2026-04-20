"""Artifact owner manifest loader/validator (gov_5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Final

import yaml

DEFAULT_OWNER_MANIFEST_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "owner_manifest.yaml"
)


class OwnerManifestError(ValueError):
    """Raised when owner_manifest.yaml is malformed."""


@dataclass(frozen=True)
class OwnerManifest:
    owner: str
    date: str
    hash: str
    artifacts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.owner.strip():
            raise OwnerManifestError("owner must be non-empty")
        try:
            date.fromisoformat(self.date)
        except ValueError as exc:
            raise OwnerManifestError(f"date must be ISO YYYY-MM-DD, got {self.date!r}") from exc
        expected = manifest_hash(self.owner, self.date)
        if self.hash != expected:
            raise OwnerManifestError("hash mismatch: expected sha256(owner|date)")


def manifest_hash(owner: str, manifest_date: str) -> str:
    payload = f"{owner}|{manifest_date}".encode()
    return sha256(payload).hexdigest()


def load_owner_manifest(path: Path | None = None) -> OwnerManifest:
    manifest_path = path or DEFAULT_OWNER_MANIFEST_PATH
    if not manifest_path.is_file():
        raise OwnerManifestError(f"owner manifest not found: {manifest_path}")
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise OwnerManifestError("owner manifest must be a mapping")
    expected_keys = {"owner", "date", "hash", "artifacts"}
    if set(loaded.keys()) != expected_keys:
        raise OwnerManifestError(f"owner manifest keys must be exactly {sorted(expected_keys)}")
    owner = loaded.get("owner")
    declared_date = loaded.get("date")
    declared_hash = loaded.get("hash")
    artifacts = loaded.get("artifacts")
    if (
        not isinstance(owner, str)
        or not isinstance(declared_date, str)
        or not isinstance(declared_hash, str)
    ):
        raise OwnerManifestError("owner/date/hash must be strings")
    if len(declared_hash) != 64 or any(ch not in "0123456789abcdef" for ch in declared_hash):
        raise OwnerManifestError("hash must be a lowercase sha256 hex digest")
    if not isinstance(artifacts, list):
        raise OwnerManifestError("artifacts must be a list")
    parsed_artifacts: list[tuple[str, str]] = []
    for item in artifacts:
        if not isinstance(item, dict) or set(item.keys()) != {"path", "owner"}:
            raise OwnerManifestError("artifact entries must define only path/owner")
        path = item.get("path")
        artifact_owner = item.get("owner")
        if not isinstance(path, str) or not path.strip():
            raise OwnerManifestError("artifact path must be a non-empty string")
        if not isinstance(artifact_owner, str) or not artifact_owner.strip():
            raise OwnerManifestError("artifact owner must be a non-empty string")
        parsed_artifacts.append((path, artifact_owner))
    return OwnerManifest(
        owner=owner,
        date=declared_date,
        hash=declared_hash,
        artifacts=tuple(parsed_artifacts),
    )
