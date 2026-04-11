"""Integrity oracle for KLR mutations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from neurophase.reset.state import SystemState


class IntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class MutationRecord:
    before_hash: str
    after_hash: str
    reason: str


class IntegrityOracle:
    def __init__(self) -> None:
        self._trail: list[MutationRecord] = []

    def checksum_state(self, state: SystemState) -> str:
        return hashlib.sha256(state.weights.tobytes()).hexdigest()

    def verify_state(self, state: SystemState, checksum: str) -> bool:
        return self.checksum_state(state) == checksum

    def log_mutation(self, before_hash: str, after_hash: str, reason: str) -> None:
        self._trail.append(MutationRecord(before_hash, after_hash, reason))

    def audit_trail(self) -> list[MutationRecord]:
        return list(self._trail)

    def assert_mutation_logged(self, before_hash: str, after_hash: str) -> None:
        for row in self._trail:
            if row.before_hash == before_hash and row.after_hash == after_hash:
                return
        raise IntegrityError("checkpoint mutation without log_mutation")
