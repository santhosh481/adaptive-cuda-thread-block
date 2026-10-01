"""Tests for src.profiling.gpu_info.

These tests run on a machine with NO GPU.  They must therefore only assert
things that are true regardless of hardware:

* The module imports.
* ``collect_environment`` never raises and always returns a GpuEnvironment.
* Every field is either a concrete value or the ``UNAVAILABLE`` sentinel.
* ``format_report`` produces a non-empty string that mentions the verdict.
* ``main --json`` writes valid JSON to stdout.

They deliberately do NOT assert that a GPU is present.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout

from src.profiling.gpu_info import (
    UNAVAILABLE,
    GpuEnvironment,
    collect_environment,
    format_report,
    main,
)


def test_collect_environment_never_raises():
    env = collect_environment()
    assert isinstance(env, GpuEnvironment)


def test_environment_fields_are_filled_or_explicitly_unavailable():
    env = collect_environment()
    # These three are always known, regardless of GPU presence.
    assert env.host_platform
    assert env.host_python_version
    assert env.host_project_root

    # These may legitimately be UNAVAILABLE on a GPU-less machine.
    for value in (
        env.driver_version,
        env.cuda_toolkit_version,
    ):
        assert isinstance(value, str)
        # The sentinel is a specific string; anything else must be a real value.
        assert value == UNAVAILABLE or value.strip() != ""


def test_gpu_count_is_none_or_int():
    env = collect_environment()
    assert env.gpu_count is None or (isinstance(env.gpu_count, int) and env.gpu_count >= 0)


def test_gpu_ready_matches_evidence():
    env = collect_environment()
    if env.gpu_ready:
        # If we claim readiness, both pieces of evidence must be present.
        assert env.gpus, "gpu_ready=True requires nvidia-smi to have listed a GPU"
        assert env.device_query_report is not None
        assert env.device_query_report.get("devices")
    # The other direction is a policy decision (driver + device_query must
    # both succeed), not a hardware fact; we don't assert it here.


def test_format_report_contains_verdict_line():
    env = collect_environment()
    text = format_report(env)
    assert "GPU / CUDA ENVIRONMENT REPORT" in text
    assert "GPU READY" in text


def test_main_json_emits_valid_json():
    buf = io.StringIO()
    with redirect_stdout(buf):
        # main() writes to stdout and returns an int exit code.
        code = main(["--json"])
    assert code in (0, 1)
    data = json.loads(buf.getvalue())
    assert "host_platform" in data
    assert "gpu_ready" in data


def test_main_human_emits_report():
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main([])
    assert code in (0, 1)
    assert "GPU / CUDA ENVIRONMENT REPORT" in buf.getvalue()


def test_unavailable_sentinel_is_a_plain_string():
    # Guard against accidental "improvements" that replace the sentinel
    # with something the rest of the codebase does not expect.
    assert isinstance(UNAVAILABLE, str)
    assert UNAVAILABLE == "Unavailable"