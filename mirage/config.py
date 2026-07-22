"""Typed, portable project configuration for MIRAGE-M1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _nosync_candidates(path: Path) -> tuple[Path, ...]:
    """Return alternatives created by macOS local-only folders.

    Adding .nosync to a directory is a common way to keep large data out of
    iCloud Drive. The repository continues to declare portable data paths; this
    helper only selects the local alternative when the declared path is absent.
    """

    candidates = [path]
    parts = path.parts
    first_name = 1 if path.is_absolute() else 0
    for index in range(first_name, len(parts)):
        part = parts[index]
        if part in {".", ".."} or part.endswith(".nosync"):
            continue
        candidates.append(Path(*parts[:index], f"{part}.nosync", *parts[index + 1 :]))
    return tuple(candidates)


def _resolve_local_path(value: str | Path, *, archive: bool = False) -> Path:
    """Prefer the configured path, with transparent macOS .nosync fallbacks."""

    configured = Path(value)
    candidates = list(_nosync_candidates(configured))
    if archive:
        candidates.extend(
            (
                configured.with_name(f"{configured.name}.nosync"),
                configured.with_name(f"{configured.stem}.nosync{configured.suffix}"),
                configured.with_name(f"{configured.name}.nosync{configured.suffix}"),
            )
        )
    return next((candidate for candidate in candidates if candidate.exists()), configured)


@dataclass(frozen=True)
class MirageConfig:
    seed: int
    data: dict[str, Any]
    split: dict[str, Any]
    normalization: dict[str, Any]
    pretraining: dict[str, Any]
    model: dict[str, Any]
    outputs: dict[str, Any]

    @property
    def archive(self) -> Path:
        return _resolve_local_path(self.data["archive"], archive=True)

    @property
    def raw_dir(self) -> Path:
        return _resolve_local_path(self.data["raw_dir"])

    @property
    def cache_dir(self) -> Path:
        return _resolve_local_path(self.data["cache_dir"])

    @property
    def ssl_store_dir(self) -> Path:
        return _resolve_local_path(self.data["ssl_store_dir"])

    @property
    def output_root(self) -> Path:
        return Path(str(self.outputs["root"]))

    @property
    def forecasting_dir(self) -> Path:
        return Path(str(self.outputs["forecasting"]))

    @property
    def pretraining_dir(self) -> Path:
        return Path(str(self.outputs["pretraining"]))

    @property
    def audit_dir(self) -> Path:
        return Path(str(self.outputs["audit"]))


def load_config(path: str | Path | None = None) -> MirageConfig:
    """Load the checked project configuration."""

    source = DEFAULT_CONFIG_PATH if path is None else Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must contain a mapping")
    required = {
        "seed",
        "data",
        "split",
        "normalization",
        "pretraining",
        "model",
        "outputs",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    config = MirageConfig(
        seed=int(payload["seed"]),
        data=dict(payload["data"]),
        split=dict(payload["split"]),
        normalization=dict(payload["normalization"]),
        pretraining=dict(payload["pretraining"]),
        model=dict(payload["model"]),
        outputs=dict(payload["outputs"]),
    )
    _validate_config(config)
    return config


def _validate_config(config: MirageConfig) -> None:
    if not 0.0 < float(config.split["train_fraction"]) < 1.0:
        raise ValueError("split.train_fraction must lie in (0, 1)")
    if not 0.0 < float(config.split["calibration_fraction"]) < 1.0:
        raise ValueError("split.calibration_fraction must lie in (0, 1)")
    if float(config.split["train_fraction"]) + float(config.split["calibration_fraction"]) >= 1.0:
        raise ValueError("outer split must leave a final test partition")
    if int(config.pretraining["target_folds"]) < 2:
        raise ValueError("pretraining.target_folds must be at least two")
    if int(config.pretraining["coverage_epochs"]) < int(config.pretraining["target_folds"]):
        raise ValueError("coverage_epochs must complete at least one target-fold cycle")
    if float(config.pretraining["forecast_hours"]) >= float(config.pretraining["block_hours"]):
        raise ValueError("forecast_hours must be shorter than block_hours")
