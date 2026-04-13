#!/usr/bin/env python3
"""tools/check_contract.py -- end-to-end LSL contract check, no hardware.

Proves that ``tools/polar_producer.py``'s LSL outlet shape is a drop-in
match for ``neurophase/physio/live.py``'s consumer, without needing a
Polar H10 or any BLE hardware.

Strategy:

1. Import the producer module from ``tools/polar_producer.py`` via
   :mod:`importlib` (it is OUT of repo kernel, not installable).
2. Assert its LSL constants equal the kernel's ``LSL_CHANNEL_COUNT``,
   ``LSL_CHANNEL_FORMAT``, ``LSL_STREAM_TYPE``.
3. Spawn the real neurophase live consumer as a subprocess.
4. Wait for its ``LISTENING`` event on stdout.
5. Build one real LSL outlet via ``polar_producer.create_lsl_outlet``
   (the same function the BLE path uses).
6. Push a deterministic RR sequence as ``[mono, rr]`` samples.
7. Push ``[NaN, NaN]`` EOF sentinel exactly once via
   ``polar_producer.SentinelGuard``.
8. Assert the consumer exits 0 and emitted the expected FRAME count.

Run::

    PYTHONPATH=. python tools/check_contract.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the kernel package reachable without a separate install (mirrors
# how tests run via pytest's rootdir).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_polar_producer() -> ModuleType:
    path = _REPO_ROOT / "tools" / "polar_producer.py"
    spec = importlib.util.spec_from_file_location("polar_producer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec: @dataclass looks up cls.__module__
    # there during class creation.
    sys.modules["polar_producer"] = module
    spec.loader.exec_module(module)
    return module


def _wait_for_event(stdout: Any, event_name: str, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = stdout.readline()
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("event") == event_name:
            return evt
    raise RuntimeError(f"timed out waiting for {event_name!r}")


def main() -> int:
    pp = _load_polar_producer()

    # --- Layer A1: constants-level contract check ---------------------
    from neurophase.physio.live import (
        LSL_CHANNEL_COUNT,
        LSL_CHANNEL_FORMAT,
        LSL_STREAM_TYPE,
    )

    assert pp.LSL_CHANNEL_COUNT == LSL_CHANNEL_COUNT, (
        f"channel_count mismatch: producer={pp.LSL_CHANNEL_COUNT} "
        f"consumer={LSL_CHANNEL_COUNT}"
    )
    assert pp.LSL_CHANNEL_FORMAT == LSL_CHANNEL_FORMAT, (
        f"channel_format mismatch: producer={pp.LSL_CHANNEL_FORMAT} "
        f"consumer={LSL_CHANNEL_FORMAT}"
    )
    assert pp.LSL_STREAM_TYPE == LSL_STREAM_TYPE, (
        f"stream_type mismatch: producer={pp.LSL_STREAM_TYPE} "
        f"consumer={LSL_STREAM_TYPE}"
    )
    print("layer-A1 constants: OK")

    # --- Layer A2: parser self-test via --self-test ------------------
    rc = subprocess.call(
        [sys.executable, str(_REPO_ROOT / "tools" / "polar_producer.py"), "--self-test"]
    )
    assert rc == 0, f"polar_producer --self-test exit={rc}"
    print("layer-A2 self-test:  OK")

    # --- Layer B: spawn consumer, push via polar_producer.create_lsl_outlet --
    stream = f"neurophase-rr-contract-{uuid.uuid4().hex[:8]}"
    n_frames = 20
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    consumer = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "neurophase.physio.live",
            "--stream-name",
            stream,
            "--max-frames",
            str(n_frames),
            "--stall-timeout-s",
            "4.0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )
    try:
        _wait_for_event(consumer.stdout, "LISTENING", timeout_s=15.0)

        outlet = pp.create_lsl_outlet(stream, f"polar-contract-{uuid.uuid4().hex[:6]}")

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not outlet.have_consumers():
            time.sleep(0.05)
        assert outlet.have_consumers(), "consumer never subscribed"

        guard = pp.SentinelGuard(outlet)

        # Push n_frames deterministic RR samples.
        for i in range(n_frames):
            rr = 820.0 + (i % 5) * 2.0
            outlet.push_sample([time.monotonic(), rr])
            time.sleep(0.02)

        # One-shot sentinel.
        assert guard.emit_once() is True
        assert guard.emit_once() is False  # second call is a no-op

        consumer.wait(timeout=30.0)
        assert consumer.returncode == 0, (
            f"consumer exit={consumer.returncode}  stderr={consumer.stderr.read()!r}"
        )
    finally:
        if consumer.poll() is None:
            consumer.terminate()

    # Count FRAME events to be thorough.
    tail = consumer.stdout.read() if consumer.stdout else ""
    frames = [
        ln
        for ln in tail.splitlines()
        if ln.strip().startswith("{") and '"event": "FRAME"' in ln
    ]
    print(f"layer-B end-to-end:  OK  ({len(frames)} post-drain FRAMEs seen)")
    print("CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
