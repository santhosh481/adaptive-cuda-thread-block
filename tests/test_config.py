"""Tests for src.config: loading and validation of config/experiment.yaml."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config import (
    CUDA_MAX_THREADS_PER_BLOCK,
    ConfigError,
    ExperimentConfig,
    load_config,
    validate_block_sizes,
)


# ---------------------------------------------------------------------------
# Loading the real shipped configuration
# ---------------------------------------------------------------------------

def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.project.name
    assert "Adaptive CUDA Thread-Block" in cfg.project.name


def test_default_config_candidate_block_sizes():
    cfg = load_config()
    assert cfg.candidate_block_sizes == (64, 128, 256, 512)


def test_default_config_benchmark_counters():
    cfg = load_config()
    assert cfg.benchmark.warmup_iterations == 10
    assert cfg.benchmark.measurement_iterations == 100
    assert cfg.benchmark.repetitions == 5


def test_default_config_paths_are_relative():
    cfg = load_config()
    for key in ("raw_data", "processed_data", "results", "models"):
        value = getattr(cfg.paths, key)
        assert not Path(value).is_absolute(), f"paths.{key} must be relative"


def test_resolve_makes_path_absolute_under_project_root():
    cfg = load_config()
    resolved = cfg.resolve(cfg.paths.raw_data)
    assert resolved.is_absolute()
    assert resolved.name == "raw"


def test_ensure_dir_creates_directory(tmp_path):
    cfg = load_config()
    target = cfg.resolve(cfg.paths.raw_data)
    # The directory should already exist in the repository.
    assert target.is_dir()


# ---------------------------------------------------------------------------
# Loading from an explicit path
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "experiment.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


_VALID = """
project:
  name: "Test project"
benchmark:
  candidate_block_sizes: [64, 128, 256, 512]
  warmup_iterations: 10
  measurement_iterations: 100
  repetitions: 5
paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  results: "results"
  models: "models"
"""


def test_load_from_explicit_path(tmp_path):
    p = _write(tmp_path, _VALID)
    cfg = load_config(p)
    assert cfg.project.name == "Test project"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_empty_file_raises(tmp_path):
    p = tmp_path / "experiment.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_config(p)


def test_invalid_yaml_raises(tmp_path):
    p = tmp_path / "experiment.yaml"
    p.write_text("project: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(p)


def test_top_level_must_be_mapping(tmp_path):
    p = tmp_path / "experiment.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(p)


# ---------------------------------------------------------------------------
# Missing keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("section,key", [
    ("project", "name"),
    ("benchmark", "candidate_block_sizes"),
    ("benchmark", "warmup_iterations"),
    ("benchmark", "measurement_iterations"),
    ("benchmark", "repetitions"),
    ("paths", "raw_data"),
    ("paths", "processed_data"),
    ("paths", "results"),
    ("paths", "models"),
])
def test_missing_key_raises(tmp_path, section, key):
    data = yaml.safe_load(_VALID)
    del data[section][key]
    p = tmp_path / "experiment.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required keys"):
        load_config(p)


# ---------------------------------------------------------------------------
# validate_block_sizes
# ---------------------------------------------------------------------------

def test_validate_block_sizes_accepts_expected_values():
    assert validate_block_sizes([64, 128, 256, 512]) == (64, 128, 256, 512)


@pytest.mark.parametrize("bad", [
    [],
    [0],
    [-64],
    [33],
    [100],
    [1024 + 32],
    [64, 64],
    [64, "128"],
    [64, 128.0],
    [64, True],
])
def test_validate_block_sizes_rejects_bad_input(bad):
    with pytest.raises(ConfigError):
        validate_block_sizes(bad)


def test_validate_block_sizes_accepts_max():
    assert validate_block_sizes([CUDA_MAX_THREADS_PER_BLOCK]) == (CUDA_MAX_THREADS_PER_BLOCK,)


# ---------------------------------------------------------------------------
# Accessing a missing attribute
# ---------------------------------------------------------------------------

def test_section_missing_attribute_raises():
    cfg = load_config()
    with pytest.raises(AttributeError):
        _ = cfg.project.does_not_exist