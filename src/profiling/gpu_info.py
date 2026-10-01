"""Detect and report the local CUDA / NVIDIA GPU environment.

This module is designed to be importable and runnable on ANY machine:

* on a Windows laptop with no NVIDIA GPU and no CUDA toolkit,
* on a Linux workstation with a GPU but a partially broken install,
* on a Google Colab runtime with a fully functional GPU stack.

It never invents a value.  Anything it cannot detect is reported as
``"Unavailable"`` (or ``None`` in the structured ``--json`` output).

Data sources, in order of preference:

1. ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``
   Host driver view of the GPU.  Works even when the CUDA toolkit is
   absent, as long as the NVIDIA driver is installed.
2. ``nvcc --version``
   Presence and version of the CUDA *toolkit* (the compiler).
3. ``build/bin/device_query[.exe] --json``
   Our own CUDA program.  This is the only source that can read the
   per-device limits (SM count, max threads per block, registers per
   block, ...) because those require the CUDA runtime.  If the binary
   is not present, every field it would have filled is left as
   ``"Unavailable"``.

Typical use
-----------
    python -m src.profiling.gpu_info
    python -m src.profiling.gpu_info --json
    python -m src.profiling.gpu_info --project-root .
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNAVAILABLE = "Unavailable"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GpuEnvironment:
    """Structured environment report.

    Every field is either a concrete value or ``UNAVAILABLE`` / ``None``.
    There is no default that could be mistaken for a measurement.
    """

    # --- host -------------------------------------------------------------
    host_platform: str = UNAVAILABLE
    host_python_version: str = UNAVAILABLE
    host_project_root: str = UNAVAILABLE

    # --- driver (nvidia-smi) ---------------------------------------------
    nvidia_smi_available: bool = False
    nvidia_smi_path: str | None = None
    driver_version: str = UNAVAILABLE
    gpu_count: int | None = None
    gpus: list[dict[str, Any]] = field(default_factory=list)

    # --- CUDA toolkit (nvcc) ---------------------------------------------
    nvcc_available: bool = False
    nvcc_path: str | None = None
    cuda_toolkit_version: str = UNAVAILABLE

    # --- our own device_query --------------------------------------------
    device_query_binary: str | None = None
    device_query_report: dict[str, Any] | None = None

    # --- verdict ---------------------------------------------------------
    gpu_ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    """Run a command, capturing stdout/stderr.  Never raises."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out: {' '.join(cmd)}"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", f"{type(exc).__name__}: {exc}"


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

_NVIDIA_SMI_QUERY_FIELDS = [
    "name",
    "compute_cap",
    "memory.total",
    "driver_version",
    "uuid",
]


def probe_nvidia_smi(env: GpuEnvironment) -> None:
    """Fill the driver-level fields of ``env`` using nvidia-smi."""

    exe = shutil.which("nvidia-smi")
    if exe is None:
        return

    env.nvidia_smi_available = True
    env.nvidia_smi_path = exe

    rc, out, err = _run(
        [
            exe,
            f"--query-gpu={','.join(_NVIDIA_SMI_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if rc != 0:
        # nvidia-smi exists but cannot talk to the driver (e.g. no GPU).
        return

    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_NVIDIA_SMI_QUERY_FIELDS):
            continue
        row: dict[str, Any] = {}
        for key, value in zip(_NVIDIA_SMI_QUERY_FIELDS, parts):
            if value in ("", "[N/A]", "N/A"):
                row[key] = UNAVAILABLE
            else:
                row[key] = value
        rows.append(row)

    if rows:
        env.gpus = rows
        env.gpu_count = len(rows)
        env.driver_version = rows[0].get("driver_version", UNAVAILABLE)


def probe_nvcc(env: GpuEnvironment) -> None:
    """Fill the CUDA-toolkit fields of ``env`` using nvcc."""

    exe = shutil.which("nvcc")
    if exe is None:
        return

    env.nvcc_available = True
    env.nvcc_path = exe

    rc, out, err = _run([exe, "--version"])
    text = out if out else err
    if rc != 0 and not text:
        return

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Cuda compilation tools") or "release" in line.lower():
            env.cuda_toolkit_version = line
            break
    else:
        first = _first_nonempty_line(text)
        if first:
            env.cuda_toolkit_version = first


def _find_device_query(project_root: Path) -> Path | None:
    """Look for our compiled device_query binary in the usual places."""
    candidates = [
        project_root / "build" / "bin" / "device_query",
        project_root / "build" / "bin" / "device_query.exe",
        project_root / "build" / "bin" / "Release" / "device_query.exe",
        project_root / "build" / "bin" / "Debug" / "device_query.exe",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def probe_device_query(env: GpuEnvironment, project_root: Path) -> None:
    """Run our own device_query --json, if it exists, and capture its output."""
    binary = _find_device_query(project_root)
    if binary is None:
        return

    env.device_query_binary = str(binary)

    rc, out, err = _run([str(binary), "--json"], timeout=30.0)
    if rc != 0:
        return

    try:
        report = json.loads(out)
    except json.JSONDecodeError:
        return

    if isinstance(report, dict):
        env.device_query_report = report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_environment(project_root: Path | str | None = None) -> GpuEnvironment:
    """Collect the whole environment report.

    Parameters
    ----------
    project_root
        Repository root used to locate ``build/bin/device_query``.  When
        ``None`` the parent of this package's directory is used.
    """
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]

    env = GpuEnvironment()
    env.host_platform = f"{platform.system()} {platform.release()} ({platform.machine()})"
    env.host_python_version = platform.python_version()
    env.host_project_root = str(root)

    probe_nvidia_smi(env)
    probe_nvcc(env)
    probe_device_query(env, root)

    # "Ready" means: the driver sees at least one GPU AND our CUDA program
    # successfully reported at least one device.  Anything less is honestly
    # reported as not ready.
    driver_sees_gpu = bool(env.gpus)
    device_query_ok = bool(
        env.device_query_report
        and isinstance(env.device_query_report.get("devices"), list)
        and len(env.device_query_report["devices"]) > 0
    )
    env.gpu_ready = driver_sees_gpu and device_query_ok

    return env


def format_report(env: GpuEnvironment) -> str:
    """Human-readable multi-line report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("GPU / CUDA ENVIRONMENT REPORT")
    lines.append("=" * 70)

    lines.append("")
    lines.append("[Host]")
    lines.append(f"  Platform           : {env.host_platform}")
    lines.append(f"  Python             : {env.host_python_version}")
    lines.append(f"  Project root       : {env.host_project_root}")

    lines.append("")
    lines.append("[NVIDIA driver  (nvidia-smi)]")
    lines.append(f"  Available          : {env.nvidia_smi_available}")
    lines.append(f"  Path               : {env.nvidia_smi_path or UNAVAILABLE}")
    lines.append(f"  Driver version     : {env.driver_version}")
    lines.append(f"  GPU count          : {env.gpu_count if env.gpu_count is not None else UNAVAILABLE}")
    if env.gpus:
        for i, g in enumerate(env.gpus):
            lines.append(f"  GPU[{i}]:")
            for k in ("name", "compute_cap", "memory.total", "uuid"):
                lines.append(f"      {k:<16}: {g.get(k, UNAVAILABLE)}")

    lines.append("")
    lines.append("[CUDA toolkit  (nvcc)]")
    lines.append(f"  Available          : {env.nvcc_available}")
    lines.append(f"  Path               : {env.nvcc_path or UNAVAILABLE}")
    lines.append(f"  Version            : {env.cuda_toolkit_version}")

    lines.append("")
    lines.append("[Device query  (build/bin/device_query --json)]")
    lines.append(f"  Binary             : {env.device_query_binary or UNAVAILABLE}")
    if env.device_query_report:
        rep = env.device_query_report
        lines.append(f"  CUDA runtime       : {rep.get('cuda_runtime_version', UNAVAILABLE)}")
        lines.append(f"  CUDA driver        : {rep.get('cuda_driver_version', UNAVAILABLE)}")
        lines.append(f"  Device count       : {rep.get('device_count', UNAVAILABLE)}")
        for i, dev in enumerate(rep.get("devices", [])):
            lines.append(f"  Device[{i}] name    : {dev.get('name', UNAVAILABLE)}")
            lines.append(f"  Device[{i}] cc      : {dev.get('compute_capability', UNAVAILABLE)}")
            lines.append(f"  Device[{i}] SMs     : {dev.get('multiprocessor_count', UNAVAILABLE)}")
            lines.append(f"  Device[{i}] tpb max : {dev.get('max_threads_per_block', UNAVAILABLE)}")
            lines.append(f"  Device[{i}] t/SM    : {dev.get('max_threads_per_multiprocessor', UNAVAILABLE)}")
            lines.append(f"  Device[{i}] smem/b  : {dev.get('shared_memory_per_block', UNAVAILABLE)}")
            lines.append(f"  Device[{i}] regs/b  : {dev.get('registers_per_block', UNAVAILABLE)}")
            lines.append(f"  Kernel smoke test  : {dev.get('kernel_smoke_test', UNAVAILABLE)}")
    else:
        lines.append("  (device_query binary not found or failed; build it with "
                     "'cmake --build build --config Release' on a GPU machine)")

    lines.append("")
    lines.append("[Verdict]")
    lines.append(f"  GPU READY          : {env.gpu_ready}")
    if not env.gpu_ready:
        lines.append("  Reason             : at least one of (driver sees GPU,")
        lines.append("                       device_query succeeded) is false.")
        lines.append("  Action             : run the CUDA parts on a GPU machine")
        lines.append("                       such as Google Colab (see README.md).")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.profiling.gpu_info",
        description="Detect and report the local CUDA / NVIDIA GPU environment.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable report.",
    )
    p.add_argument(
        "--project-root",
        default=None,
        help="Override the repository root (default: two levels above this file).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns the process exit code."""
    args = _build_arg_parser().parse_args(argv)
    env = collect_environment(args.project_root)

    if args.json:
        json.dump(env.to_dict(), sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(format_report(env))
        sys.stdout.write("\n")

    return 0 if env.gpu_ready else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())