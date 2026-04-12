# neurophase — Observatory Export Contract

**Module.** `neurophase.observatory`
**Role.** Outbound boundary for an external self-observing collector.
**Version.** `neurophase.observatory.OBSERVATORY_SCHEMA_VERSION = "1.0.0"`
**Status.** Load-bearing as of kernelization v1.

## What the observatory is (and is not)

*Is.* A one-way export surface. An external collector — a multi-repo γ
collector, a dashboard, an archival log sink — plugs in via the
:class:`ObservatorySink` protocol and receives typed events.

*Is not.* A distributed cognition system, a remote-control channel,
or a multi-node consensus layer. The export is strictly outbound;
there is no write-back path from the collector into the kernel.

## Wire format

Every event is the dict returned by `ObservatoryEvent.to_json_dict()`:

```json
{
  "kind": "runtime.tick",
  "schema_version": "1.0.0",
  "frame_schema_version": "1.0.0",
  "source": "neurophase",
  "payload": { ...canonical frame dict v1.0.0... }
}
```

Two version fields:

* `schema_version` — the observatory envelope version. Governs the
  outer shape (`kind`, `source`, `payload`).
* `frame_schema_version` — the canonical runtime frame version carried
  inside `payload`. Governs the inner shape (all 22 runtime frame
  keys).

They are independent so a minor canonical-schema bump does not force
collectors to upgrade.

## Typical integration

```python
from neurophase.api import RuntimeOrchestrator, OrchestratorConfig, PipelineConfig, PolicyConfig
from neurophase.observatory import ObservatoryExporter

class HttpSink:
    def __init__(self, url): self.url = url
    def send(self, event): httpx.post(self.url, json=event.to_json_dict())

orch = RuntimeOrchestrator(OrchestratorConfig(pipeline=PipelineConfig(), policy=PolicyConfig()))
exporter = ObservatoryExporter(HttpSink("https://collector/events"), source="neurophase@host-42")

for _ in range(n):
    frame = orch.tick(...)
    exporter.emit(frame)   # pure projection + validated dispatch
```

The exporter:

* validates the frame against the canonical schema before shipping;
* holds no mutable state between emits;
* never retries, buffers, or deduplicates — those concerns belong in
  the sink.

## Audit handshake

The `payload.ledger_record_hash` field lets a collector reconstruct the
SHA256-chained audit ledger independently. A collector can:

1. Store every observed event.
2. Replay the kernel ledger via `neurophase.audit.replay.replay_ledger`.
3. Verify the replayed frames byte-match the observed `payload` dicts.

That closes the audit loop without any special observatory protocol —
just the canonical frame schema.

## Enforcement

`tests/test_observatory_export.py` — 10 tests covering:

* schema version is semver;
* event carries both versions + kind + source + canonical payload;
* event dict round-trips through `json.dumps`;
* empty `source` is rejected;
* exporter delivers to the sink, holds no state, and is deterministic
  under replay (byte-identical event sequences).
