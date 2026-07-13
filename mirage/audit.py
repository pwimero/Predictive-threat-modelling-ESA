"""Machine-checkable leakage and full-observation coverage audits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from mirage.normalization import ChannelNormalizer
from mirage.protocol import TemporalProtocol, canonical_sha256

AUDIT_SCHEMA_VERSION = "mirage-leakage-audit/1.0"
FINGERPRINT_SCHEMA_VERSION = "mirage-split-fingerprint/1.0"


def _timestamp(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


@dataclass(frozen=True)
class CoverageReport:
    """Counts proving whether one target-fold cycle used every observation once."""

    raw_training_observations: int
    target_assignments: int
    used_as_reconstruction_targets: int
    used_more_than_once: int
    never_used: int
    coverage_fraction: float
    partial_coverage: bool

    def __post_init__(self) -> None:
        counts = (
            self.raw_training_observations,
            self.target_assignments,
            self.used_as_reconstruction_targets,
            self.used_more_than_once,
            self.never_used,
        )
        if any(count < 0 for count in counts):
            raise ValueError("coverage counts cannot be negative")
        if not 0.0 <= self.coverage_fraction <= 1.0:
            raise ValueError("coverage_fraction must lie in [0, 1]")
        if (
            self.used_as_reconstruction_targets + self.never_used
            != self.raw_training_observations
        ):
            raise ValueError("coverage counts do not reconcile")

    @property
    def complete(self) -> bool:
        return (
            not self.partial_coverage
            and self.coverage_fraction == 1.0
            and self.never_used == 0
            and self.used_more_than_once == 0
            and self.target_assignments == self.raw_training_observations
        )

    def assert_complete(self) -> None:
        if not self.complete:
            raise AssertionError(
                "incomplete target coverage: "
                f"coverage={self.coverage_fraction:.6f}, never={self.never_used}, "
                f"repeated={self.used_more_than_once}, assignments={self.target_assignments}"
            )

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "raw_training_observations": self.raw_training_observations,
            "target_assignments": self.target_assignments,
            "used_as_reconstruction_targets": self.used_as_reconstruction_targets,
            "used_more_than_once": self.used_more_than_once,
            "never_used": self.never_used,
            "coverage_fraction": self.coverage_fraction,
            "partial_coverage": self.partial_coverage,
            "complete": self.complete,
        }


def audit_target_coverage(target_counts: ArrayLike) -> CoverageReport:
    """Audit a count per allowed raw observation after a coverage cycle."""

    counts = np.asarray(target_counts)
    if counts.ndim != 1:
        raise ValueError("target_counts must be one-dimensional")
    if not np.issubdtype(counts.dtype, np.integer):
        if not np.all(np.isfinite(counts)) or not np.all(counts == np.floor(counts)):
            raise ValueError("target counts must be finite integers")
        counts = counts.astype(np.int64)
    if np.any(counts < 0):
        raise ValueError("target counts cannot be negative")

    raw_count = int(counts.size)
    used_count = int(np.count_nonzero(counts))
    repeated_count = int(np.count_nonzero(counts > 1))
    never_count = raw_count - used_count
    assignments = int(np.sum(counts, dtype=np.int64))
    fraction = 1.0 if raw_count == 0 else used_count / raw_count
    partial = never_count > 0 or repeated_count > 0 or assignments != raw_count
    return CoverageReport(
        raw_training_observations=raw_count,
        target_assignments=assignments,
        used_as_reconstruction_targets=used_count,
        used_more_than_once=repeated_count,
        never_used=never_count,
        coverage_fraction=float(fraction),
        partial_coverage=partial,
    )


def audit_observation_assignments(
    expected_observation_ids: Sequence[object], target_observation_ids: Iterable[object]
) -> CoverageReport:
    """Convenience coverage audit for small fixtures with explicit observation IDs."""

    if len(expected_observation_ids) != len(set(expected_observation_ids)):
        raise ValueError("expected observation IDs must be unique")
    positions = {observation_id: index for index, observation_id in enumerate(expected_observation_ids)}
    counts = np.zeros(len(expected_observation_ids), dtype=np.int64)
    for observation_id in target_observation_ids:
        if observation_id not in positions:
            raise ValueError(f"target is outside the allowed training corpus: {observation_id!r}")
        counts[positions[observation_id]] += 1
    return audit_target_coverage(counts)


@dataclass(frozen=True)
class TemporalOverlapReport:
    """Observed extrema used to prove that SSL data precedes protection."""

    max_pretraining_context_timestamp: pd.Timestamp
    max_pretraining_target_timestamp: pd.Timestamp
    max_pretraining_command_timestamp: pd.Timestamp | None
    min_calibration_episode_timestamp: pd.Timestamp
    protected_start: pd.Timestamp

    def __post_init__(self) -> None:
        for field_name in (
            "max_pretraining_context_timestamp",
            "max_pretraining_target_timestamp",
            "min_calibration_episode_timestamp",
            "protected_start",
        ):
            object.__setattr__(self, field_name, _timestamp(getattr(self, field_name), name=field_name))
        if self.max_pretraining_command_timestamp is not None:
            object.__setattr__(
                self,
                "max_pretraining_command_timestamp",
                _timestamp(
                    self.max_pretraining_command_timestamp,
                    name="max_pretraining_command_timestamp",
                ),
            )

    @property
    def passed(self) -> bool:
        command_safe = (
            self.max_pretraining_command_timestamp is None
            or self.max_pretraining_command_timestamp < self.protected_start
        )
        return (
            self.max_pretraining_context_timestamp < self.protected_start
            and self.max_pretraining_target_timestamp < self.protected_start
            and command_safe
            and self.min_calibration_episode_timestamp >= self.protected_start
        )

    def assert_passed(self) -> None:
        if self.max_pretraining_context_timestamp >= self.protected_start:
            raise AssertionError("pretraining context crosses the protected boundary")
        if self.max_pretraining_target_timestamp >= self.protected_start:
            raise AssertionError("pretraining target crosses the protected boundary")
        if (
            self.max_pretraining_command_timestamp is not None
            and self.max_pretraining_command_timestamp >= self.protected_start
        ):
            raise AssertionError("pretraining command crosses the protected boundary")
        if self.min_calibration_episode_timestamp < self.protected_start:
            raise AssertionError("calibration episode begins before the protected boundary")

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "max_pretraining_context_timestamp": self.max_pretraining_context_timestamp.isoformat(),
            "max_pretraining_target_timestamp": self.max_pretraining_target_timestamp.isoformat(),
            "max_pretraining_command_timestamp": (
                None
                if self.max_pretraining_command_timestamp is None
                else self.max_pretraining_command_timestamp.isoformat()
            ),
            "min_calibration_episode_timestamp": self.min_calibration_episode_timestamp.isoformat(),
            "protected_start": self.protected_start.isoformat(),
            "passed": self.passed,
        }


def assert_temporal_boundaries(
    protocol: TemporalProtocol,
    *,
    max_pretraining_context_timestamp: object,
    max_pretraining_target_timestamp: object,
    max_pretraining_command_timestamp: object | None,
    min_calibration_episode_timestamp: object,
) -> TemporalOverlapReport:
    """Construct a report and fail immediately on any temporal overlap."""

    report = TemporalOverlapReport(
        max_pretraining_context_timestamp=_timestamp(
            max_pretraining_context_timestamp, name="max_pretraining_context_timestamp"
        ),
        max_pretraining_target_timestamp=_timestamp(
            max_pretraining_target_timestamp, name="max_pretraining_target_timestamp"
        ),
        max_pretraining_command_timestamp=(
            None
            if max_pretraining_command_timestamp is None
            else _timestamp(
                max_pretraining_command_timestamp, name="max_pretraining_command_timestamp"
            )
        ),
        min_calibration_episode_timestamp=_timestamp(
            min_calibration_episode_timestamp, name="min_calibration_episode_timestamp"
        ),
        protected_start=protocol.protected_start,
    )
    report.assert_passed()
    return report


def assert_partition_isolation(
    protocol: TemporalProtocol,
    *,
    training_event_ids: Iterable[str],
    calibration_event_ids: Iterable[str] = (),
    test_event_ids: Iterable[str] = (),
) -> None:
    """Reject event use outside each declared immutable outer partition."""

    declared_train = set(protocol.outer_train_event_ids)
    declared_calibration = set(protocol.calibration_event_ids)
    declared_test = set(protocol.test_event_ids)
    used_train = set(training_event_ids)
    used_calibration = set(calibration_event_ids)
    used_test = set(test_event_ids)

    foreign_training = used_train.difference(declared_train)
    if foreign_training:
        raise AssertionError(f"non-training event IDs entered model fitting: {sorted(foreign_training)}")
    if not used_calibration.issubset(declared_calibration):
        raise AssertionError("calibration use contains IDs outside the calibration partition")
    if not used_test.issubset(declared_test):
        raise AssertionError("test use contains IDs outside the test partition")
    if used_train.intersection(used_calibration | used_test):
        raise AssertionError("an event ID was reused across fitting and protected evaluation")


def assert_train_only_normalizer(
    normalizer: ChannelNormalizer, protocol: TemporalProtocol
) -> None:
    """Require explicit scaler bounds wholly inside the SSL training interval."""

    if normalizer.fit_start is None or normalizer.fit_end is None:
        raise AssertionError("normalizer has no auditable fitting bounds")
    if normalizer.fit_start < protocol.pretraining_start:
        raise AssertionError("normalizer uses values before the declared training interval")
    if normalizer.fit_end >= protocol.protected_start:
        raise AssertionError("normalizer includes protected future values")


@dataclass(frozen=True)
class SplitFingerprint:
    """Canonical description binding every artifact to one data protocol."""

    protocol_hash: str
    archive_checksum: str
    outer_train_event_ids: tuple[str, ...]
    calibration_event_ids: tuple[str, ...]
    test_event_ids: tuple[str, ...]
    calibration_cutoff: pd.Timestamp
    test_cutoff: pd.Timestamp
    raw_interval_start: pd.Timestamp
    raw_interval_end: pd.Timestamp
    protected_start: pd.Timestamp
    channel_order: tuple[str, ...]
    command_order: tuple[str, ...]
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "calibration_cutoff",
            "test_cutoff",
            "raw_interval_start",
            "raw_interval_end",
            "protected_start",
        ):
            object.__setattr__(self, field_name, _timestamp(getattr(self, field_name), name=field_name))
        if len(self.channel_order) != len(set(self.channel_order)):
            raise ValueError("channel_order contains duplicates")
        if len(self.command_order) != len(set(self.command_order)):
            raise ValueError("command_order contains duplicates")
        if not self.protocol_hash or not self.archive_checksum:
            raise ValueError("protocol hash and archive checksum are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "protocol_hash": self.protocol_hash,
            "archive_checksum": self.archive_checksum,
            "event_partitions": {
                "outer_train": list(self.outer_train_event_ids),
                "calibration": list(self.calibration_event_ids),
                "test": list(self.test_event_ids),
            },
            "partition_cutoffs": {
                "calibration": self.calibration_cutoff.isoformat(),
                "test": self.test_cutoff.isoformat(),
            },
            "raw_interval": {
                "start": self.raw_interval_start.isoformat(),
                "end": self.raw_interval_end.isoformat(),
                "protected_start": self.protected_start.isoformat(),
            },
            "channel_order": list(self.channel_order),
            "command_order": list(self.command_order),
            "configuration": dict(self.configuration),
        }

    @property
    def fingerprint_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict() | {"fingerprint_hash": self.fingerprint_hash}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def build_split_fingerprint(
    protocol: TemporalProtocol,
    *,
    archive_checksum: str,
    channel_order: Sequence[str],
    command_order: Sequence[str],
    configuration: Mapping[str, Any],
) -> SplitFingerprint:
    return SplitFingerprint(
        protocol_hash=protocol.protocol_hash,
        archive_checksum=archive_checksum,
        outer_train_event_ids=protocol.outer_train_event_ids,
        calibration_event_ids=protocol.calibration_event_ids,
        test_event_ids=protocol.test_event_ids,
        calibration_cutoff=protocol.first_calibration_event_start,
        test_cutoff=protocol.first_test_event_start,
        raw_interval_start=protocol.pretraining_start,
        raw_interval_end=protocol.pretraining_end,
        protected_start=protocol.protected_start,
        channel_order=tuple(channel_order),
        command_order=tuple(command_order),
        configuration=dict(configuration),
    )


@dataclass(frozen=True)
class LeakageAuditReport:
    """Compact final audit artifact linked to protocol, scaler, and split."""

    protocol_hash: str
    split_fingerprint_hash: str
    scaler_hash: str
    coverage: CoverageReport
    temporal_overlap: TemporalOverlapReport
    forecasting_coverage: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        forecasting_safe = self.forecasting_coverage is None or (
            float(self.forecasting_coverage.get("coverage_fraction", 0.0)) == 1.0
            and int(self.forecasting_coverage.get("total_observations", -1))
            == self.coverage.raw_training_observations
        )
        return self.coverage.complete and self.temporal_overlap.passed and forecasting_safe

    def assert_passed(self) -> None:
        self.coverage.assert_complete()
        self.temporal_overlap.assert_passed()
        if not self.passed:
            raise AssertionError("full-data forecasting coverage does not reconcile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "protocol_hash": self.protocol_hash,
            "split_fingerprint_hash": self.split_fingerprint_hash,
            "scaler_hash": self.scaler_hash,
            "coverage": self.coverage.to_dict(),
            "temporal_overlap": self.temporal_overlap.to_dict(),
            "forecasting_coverage": self.forecasting_coverage,
            "passed": self.passed,
        }

    def save(self, path: str | Path) -> Path:
        self.assert_passed()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target
