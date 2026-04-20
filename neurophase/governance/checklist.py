"""Explicit governance checklist loader/validator (gov_2/final_1/final_2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

from neurophase.governance.ablation import (
    AblationPolicyError,
    load_ablation_policy,
)

DEFAULT_CHECKLIST_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "GOVERNANCE_CHECKLIST.yaml"
)
_VALID_STATUS: Final[set[str]] = {"pass", "fail", "n/a"}


class GovernanceChecklistError(ValueError):
    """Raised when GOVERNANCE_CHECKLIST.yaml fails explicit schema validation."""


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    status: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise GovernanceChecklistError("checklist item id must be non-empty")
        if self.status not in _VALID_STATUS:
            raise GovernanceChecklistError(
                f"checklist item {self.id!r} has invalid status {self.status!r}; expected one of {_VALID_STATUS}"
            )


@dataclass(frozen=True)
class GovernanceChecklist:
    version: int
    verdict: str
    source_checklist_path: str
    required_source_items: int
    source_items_count: int
    items: tuple[ChecklistItem, ...]

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise GovernanceChecklistError("version must be positive")
        if self.verdict != "DONE":
            raise GovernanceChecklistError("verdict must be DONE")
        if not self.source_checklist_path.strip():
            raise GovernanceChecklistError("source_checklist_path must be non-empty")
        if self.required_source_items < 211:
            raise GovernanceChecklistError("required_source_items must be >= 211")
        if self.source_items_count != self.required_source_items:
            raise GovernanceChecklistError("source_items_count must equal required_source_items")
        if len(self.items) == 0:
            raise GovernanceChecklistError("items must be non-empty")

    def by_id(self, item_id: str) -> ChecklistItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise GovernanceChecklistError(f"checklist id not found: {item_id!r}")


def load_checklist(path: Path | None = None) -> GovernanceChecklist:
    checklist_path = path or DEFAULT_CHECKLIST_PATH
    if not checklist_path.is_file():
        raise GovernanceChecklistError(f"governance checklist not found: {checklist_path}")

    payload = yaml.safe_load(checklist_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GovernanceChecklistError("checklist root must be a mapping")

    version = payload.get("version")
    verdict = payload.get("verdict")
    source_checklist_path = payload.get("source_checklist_path")
    required_source_items = payload.get("required_source_items")
    source_items_count = payload.get("source_items_count")
    items_raw = payload.get("items")
    if not isinstance(version, int):
        raise GovernanceChecklistError("version must be an integer")
    if not isinstance(verdict, str):
        raise GovernanceChecklistError("verdict must be a string")
    if not isinstance(source_checklist_path, str):
        raise GovernanceChecklistError("source_checklist_path must be a string")
    if not isinstance(required_source_items, int):
        raise GovernanceChecklistError("required_source_items must be an integer")
    if not isinstance(source_items_count, int):
        raise GovernanceChecklistError("source_items_count must be an integer")
    if not isinstance(items_raw, list):
        raise GovernanceChecklistError("items must be a list")

    items: list[ChecklistItem] = []
    seen: set[str] = set()
    for raw in items_raw:
        if not isinstance(raw, dict):
            raise GovernanceChecklistError("each checklist item must be a mapping")
        item_id = raw.get("id")
        status = raw.get("status")
        note = raw.get("note", "")
        if not isinstance(item_id, str) or not isinstance(status, str) or not isinstance(note, str):
            raise GovernanceChecklistError("item id/status/note must be strings")
        if item_id in seen:
            raise GovernanceChecklistError(f"duplicate checklist id: {item_id}")
        seen.add(item_id)
        items.append(ChecklistItem(id=item_id, status=status, note=note))

    checklist = GovernanceChecklist(
        version=version,
        verdict=verdict,
        source_checklist_path=source_checklist_path,
        required_source_items=required_source_items,
        source_items_count=source_items_count,
        items=tuple(items),
    )

    # Explicit required claims for gov/final closure.
    for required in ("gov_2", "final_1", "final_2"):
        if checklist.by_id(required).status != "pass":
            raise GovernanceChecklistError(f"{required} must be status=pass")
    source_path = checklist_path.parent / checklist.source_checklist_path
    if not source_path.is_file():
        raise GovernanceChecklistError(
            f"source checklist not found: {checklist.source_checklist_path}"
        )
    source_payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_payload, dict):
        raise GovernanceChecklistError("source checklist root must be a mapping")
    source_items = source_payload.get("items")
    if not isinstance(source_items, list):
        raise GovernanceChecklistError("source checklist items must be a list")
    if len(source_items) != checklist.source_items_count:
        raise GovernanceChecklistError(
            f"source checklist must contain exactly {checklist.source_items_count} items, got {len(source_items)}"
        )
    bad_statuses = {
        "fail",
        "partial",
    }
    offenders: list[str] = []
    for raw in source_items:
        if not isinstance(raw, dict):
            raise GovernanceChecklistError("source checklist item must be a mapping")
        sid = raw.get("id")
        sstatus = raw.get("status")
        if not isinstance(sid, str) or not isinstance(sstatus, str):
            raise GovernanceChecklistError("source checklist item id/status must be strings")
        if sstatus in bad_statuses:
            offenders.append(sid)
    if offenders:
        raise GovernanceChecklistError(
            f"source checklist has non-pass statuses for {len(offenders)} items: {offenders[:5]}"
        )
    try:
        load_ablation_policy()
    except AblationPolicyError as exc:
        raise GovernanceChecklistError(f"ablation policy validation failed: {exc}") from exc
    return checklist
