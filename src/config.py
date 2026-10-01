"""Configuration loading and validation.

The single source of truth for every experiment parameter is
``config/experiment.yaml``.  Nothing in this project may hard-code a
candidate block size, an iteration count or a path: everything is read
from the configuration that is loaded here.

Public API
----------
load_config(path=None) -> ExperimentConfig
    Load and validate the configuration.  Raises ``ConfigError`` on any
    problem, so callers never have to guess whether the file was read
    correctly.

ExperimentConfig
    Attribute-style access to the configuration sections:

        cfg.project.name
        cfg.benchmark.candidate_block_sizes
        cfg.benchmark.warmup_iterations
        cfg.paths.raw_data
        cfg.raw("gpu_environment")   # -> Path, resolved, parent created
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "experiment.yaml"

# Block sizes must be positive multiples of 32 (warp size) and must not
# exceed the CUDA hardware limit for threads per block.  These two numbers
# are the CUDA specification limits, not a specific GPU's limits; the GPU
# specific limits are read at runtime by src/profiling/gpu_info.py.
WARP_SIZE: int = 32
CUDA_MAX_THREADS_PER_BLOCK: int = 1024


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when the configuration is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Small helper: immutable, attribute-accessible view over a mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Section:
    """Recursively wrap a YAML mapping so its keys are attributes."""

    _data: Mapping[str, Any] = field(repr=False)

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._data[item]
        except KeyError as exc:
            raise AttributeError(
                f"Configuration section has no key {item!r}. "
                f"Available keys: {sorted(self._data)}"
            ) from exc
        if isinstance(value, Mapping):
            return Section(value)
        return value

    def as_dict(self) -> dict[str, Any]:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Top-level configuration object
# ---------------------------------------------------------------------------

class ExperimentConfig:
    """Typed, validated view over ``config/experiment.yaml``."""

    def __init__(self, raw: Mapping[str, Any], source_path: Path) -> None:
        self._raw: dict[str, Any] = dict(raw)
        self.source_path: Path = source_path

    # -- sections ----------------------------------------------------------

    @property
    def project(self) -> Section:
        return Section(self._raw["project"])

    @property
    def benchmark(self) -> Section:
        return Section(self._raw["benchmark"])

    @property
    def paths(self) -> Section:
        return Section(self._raw["paths"])

    # -- convenience -------------------------------------------------------

    @property
    def candidate_block_sizes(self) -> tuple[int, ...]:
        return tuple(self.benchmark.candidate_block_sizes)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._raw)

    # -- path resolution ---------------------------------------------------

    def resolve(self, relative: str | os.PathLike[str]) -> Path:
        """Resolve a repository-relative path against PROJECT_ROOT."""
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def ensure_dir(self, relative: str | os.PathLike[str]) -> Path:
        """Resolve a repository-relative directory and create it if needed."""
        p = self.resolve(relative)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:  # pragma: no cover - debug convenience
        return f"ExperimentConfig(source={self.source_path!s}, keys={sorted(self._raw)})"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = ("project", "benchmark", "paths")
_REQUIRED_BENCHMARK = (
    "candidate_block_sizes",
    "warmup_iterations",
    "measurement_iterations",
    "repetitions",
)
_REQUIRED_PATHS = ("raw_data", "processed_data", "results", "models")


def validate_block_sizes(values: Iterable[Any]) -> tuple[int, ...]:
    """Validate a candidate-block-size sequence.

    Rules
    -----
    * The sequence must be non-empty.
    * Every entry must be an integer (bool is rejected even though it is a
      subclass of int).
    * Every entry must be a positive multiple of the warp size (32).
    * Every entry must be <= CUDA_MAX_THREADS_PER_BLOCK (1024).
    * Duplicates are rejected.

    Returns the values as a tuple of ``int`` in the original order.
    """
    values = list(values)
    if not values:
        raise ConfigError("candidate_block_sizes must not be empty")

    seen: set[int] = set()
    out: list[int] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ConfigError(
                f"candidate_block_sizes[{i}] = {v!r} is not an integer"
            )
        if v <= 0:
            raise ConfigError(
                f"candidate_block_sizes[{i}] = {v} must be positive"
            )
        if v % WARP_SIZE != 0:
            raise ConfigError(
                f"candidate_block_sizes[{i}] = {v} is not a multiple of the "
                f"warp size ({WARP_SIZE})"
            )
        if v > CUDA_MAX_THREADS_PER_BLOCK:
            raise ConfigError(
                f"candidate_block_sizes[{i}] = {v} exceeds the CUDA limit of "
                f"{CUDA_MAX_THREADS_PER_BLOCK} threads per block"
            )
        if v in seen:
            raise ConfigError(
                f"candidate_block_sizes contains duplicate value {v}"
            )
        seen.add(v)
        out.append(v)

    return tuple(out)


def _validate_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}")
    return value


def _validate_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"Section {name!r} must be a mapping, got {type(value).__name__}")
    return value


def _require_keys(section_name: str, section: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in section]
    if missing:
        raise ConfigError(
            f"Section {section_name!r} is missing required keys: {missing}"
        )


def _validate(raw: Mapping[str, Any]) -> None:
    """Validate the whole configuration tree."""

    _require_keys("<top level>", raw, _REQUIRED_TOP_LEVEL)

    project = _validate_mapping("project", raw["project"])
    _require_keys("project", project, ("name",))
    if not isinstance(project["name"], str) or not project["name"].strip():
        raise ConfigError("project.name must be a non-empty string")

    benchmark = _validate_mapping("benchmark", raw["benchmark"])
    _require_keys("benchmark", benchmark, _REQUIRED_BENCHMARK)
    validate_block_sizes(benchmark["candidate_block_sizes"])
    _validate_positive_int("benchmark.warmup_iterations", benchmark["warmup_iterations"])
    _validate_positive_int(
        "benchmark.measurement_iterations", benchmark["measurement_iterations"]
    )
    _validate_positive_int("benchmark.repetitions", benchmark["repetitions"])

    paths = _validate_mapping("paths", raw["paths"])
    _require_keys("paths", paths, _REQUIRED_PATHS)
    for key in _REQUIRED_PATHS:
        if not isinstance(paths[key], str) or not paths[key].strip():
            raise ConfigError(f"paths.{key} must be a non-empty string")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def load_config(path: str | os.PathLike[str] | None = None) -> ExperimentConfig:
    """Load and validate the experiment configuration.

    Parameters
    ----------
    path
        Path to a YAML file.  If ``None`` the default
        ``config/experiment.yaml`` next to the repository root is used.

    Raises
    ------
    ConfigError
        If the file cannot be read, is not valid YAML, is not a mapping,
        or fails any of the validation rules above.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")

    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {cfg_path}: {exc}") from exc

    if raw is None:
        raise ConfigError(f"Configuration file is empty: {cfg_path}")
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"Top level of {cfg_path} must be a mapping, got {type(raw).__name__}"
        )

    _validate(raw)
    return ExperimentConfig(raw=raw, source_path=cfg_path)