"""Temporal partitions for the ESA Mission 1 experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

PROTOCOL_SCHEMA_VERSION = "mirage-temporal-protocol/1.0"


class TemporalEvent(Protocol):
    """Minimum event metadata required to construct a temporal protocol."""

    @property
    def event_id(self) -> str: ...

    @property
    def start(self) -> pd.Timestamp: ...


def _timestamp(value: object, *, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise ValueError(f"{name} must be a valid timestamp")
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _timestamp(value, name="timestamp").isoformat()
    if isinstance(value, datetime):
        return _timestamp(value, name="datetime").isoformat()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_ready(item) for item in value)
    return value


def canonical_json(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON representation used by all MIRAGE hashes."""

    return json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping after deterministic JSON canonicalisation."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RollingOriginFold:
    """One event-level model-selection fold contained in outer training."""

    index: int
    fit_event_ids: tuple[str, ...]
    validation_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ValueError("fold index must be positive")
        if not self.fit_event_ids or not self.validation_event_ids:
            raise ValueError("rolling-origin folds require non-empty fit and validation sets")
        if set(self.fit_event_ids).intersection(self.validation_event_ids):
            raise ValueError("fit and validation event IDs must be disjoint")


@dataclass(frozen=True)
class TemporalProtocol:
    """Immutable outer split and raw-data boundaries for one paper run."""

    outer_train_event_ids: tuple[str, ...]
    calibration_event_ids: tuple[str, ...]
    test_event_ids: tuple[str, ...]

    mission_start: pd.Timestamp
    pretraining_start: pd.Timestamp
    pretraining_end: pd.Timestamp
    protected_start: pd.Timestamp

    first_calibration_event_start: pd.Timestamp
    first_test_event_start: pd.Timestamp

    event_pre_hours: float
    event_post_hours: float
    command_history_hours: float
    ssl_context_hours: float
    ssl_forecast_hours: float

    def __post_init__(self) -> None:
        for field_name in (
            "mission_start",
            "pretraining_start",
            "pretraining_end",
            "protected_start",
            "first_calibration_event_start",
            "first_test_event_start",
        ):
            object.__setattr__(
                self, field_name, _timestamp(getattr(self, field_name), name=field_name)
            )

        partitions = (
            self.outer_train_event_ids,
            self.calibration_event_ids,
            self.test_event_ids,
        )
        if any(not partition for partition in partitions):
            raise ValueError("outer train, calibration, and test partitions must be non-empty")
        flattened = tuple(event_id for partition in partitions for event_id in partition)
        if any(not isinstance(event_id, str) or not event_id for event_id in flattened):
            raise ValueError("event IDs must be non-empty strings")
        if len(flattened) != len(set(flattened)):
            raise ValueError("event IDs must occur in exactly one outer partition")

        durations = (
            self.event_pre_hours,
            self.event_post_hours,
            self.command_history_hours,
            self.ssl_context_hours,
            self.ssl_forecast_hours,
        )
        if any(not math.isfinite(duration) or duration < 0 for duration in durations):
            raise ValueError("protocol durations must be finite and non-negative")
        if self.ssl_context_hours <= 0:
            raise ValueError("ssl_context_hours must be positive")

        if not (
            self.mission_start
            <= self.pretraining_start
            <= self.pretraining_end
            < self.protected_start
            <= self.first_calibration_event_start
            <= self.first_test_event_start
        ):
            raise ValueError("temporal protocol boundaries are not chronologically ordered")

    @property
    def protected_context_hours(self) -> float:
        """Longest event-side context that must remain protected."""

        return max(self.event_pre_hours, self.command_history_hours)

    def to_dict(self) -> dict[str, Any]:
        """Return the stable, JSON-ready protocol payload without its hash."""

        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "outer_train_event_ids": list(self.outer_train_event_ids),
            "calibration_event_ids": list(self.calibration_event_ids),
            "test_event_ids": list(self.test_event_ids),
            "mission_start": self.mission_start.isoformat(),
            "pretraining_start": self.pretraining_start.isoformat(),
            "pretraining_end": self.pretraining_end.isoformat(),
            "protected_start": self.protected_start.isoformat(),
            "first_calibration_event_start": self.first_calibration_event_start.isoformat(),
            "first_test_event_start": self.first_test_event_start.isoformat(),
            "event_pre_hours": self.event_pre_hours,
            "event_post_hours": self.event_post_hours,
            "command_history_hours": self.command_history_hours,
            "ssl_context_hours": self.ssl_context_hours,
            "ssl_forecast_hours": self.ssl_forecast_hours,
        }

    def canonical_json(self) -> str:
        """Return the exact representation hashed for artifact compatibility."""

        return canonical_json(self.to_dict())

    @property
    def protocol_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        """Persist the protocol and a self-verifying canonical hash."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict() | {"protocol_hash": self.protocol_hash}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TemporalProtocol:
        schema = payload.get("schema_version")
        if schema != PROTOCOL_SCHEMA_VERSION:
            raise ValueError(f"unsupported temporal protocol schema: {schema!r}")
        protocol = cls(
            outer_train_event_ids=tuple(str(item) for item in payload["outer_train_event_ids"]),
            calibration_event_ids=tuple(str(item) for item in payload["calibration_event_ids"]),
            test_event_ids=tuple(str(item) for item in payload["test_event_ids"]),
            mission_start=_timestamp(payload["mission_start"], name="mission_start"),
            pretraining_start=_timestamp(payload["pretraining_start"], name="pretraining_start"),
            pretraining_end=_timestamp(payload["pretraining_end"], name="pretraining_end"),
            protected_start=_timestamp(payload["protected_start"], name="protected_start"),
            first_calibration_event_start=_timestamp(
                payload["first_calibration_event_start"], name="first_calibration_event_start"
            ),
            first_test_event_start=_timestamp(
                payload["first_test_event_start"], name="first_test_event_start"
            ),
            event_pre_hours=float(payload["event_pre_hours"]),
            event_post_hours=float(payload["event_post_hours"]),
            command_history_hours=float(payload["command_history_hours"]),
            ssl_context_hours=float(payload["ssl_context_hours"]),
            ssl_forecast_hours=float(payload["ssl_forecast_hours"]),
        )
        saved_hash = payload.get("protocol_hash")
        if saved_hash is not None and saved_hash != protocol.protocol_hash:
            raise ValueError("temporal protocol hash mismatch")
        return protocol

    @classmethod
    def load(cls, path: str | Path) -> TemporalProtocol:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("temporal protocol file must contain a JSON object")
        return cls.from_dict(payload)


def build_temporal_protocol(
    events: Sequence[TemporalEvent],
    *,
    mission_start: object,
    event_pre_hours: float,
    event_post_hours: float,
    command_history_hours: float,
    ssl_context_hours: float,
    ssl_forecast_hours: float,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> TemporalProtocol:
    """Split events first, then derive all raw-data boundaries from that split."""

    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave a test partition")
    if len(events) < 3:
        raise ValueError("at least three events are required for the outer temporal split")

    ordered = sorted(
        events, key=lambda event: (_timestamp(event.start, name="event start"), event.event_id)
    )
    event_ids = [event.event_id for event in ordered]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("event IDs must be unique")

    train_end = int(len(ordered) * train_fraction)
    calibration_end = train_end + int(len(ordered) * calibration_fraction)
    if train_end < 1 or calibration_end <= train_end or calibration_end >= len(ordered):
        raise ValueError("split fractions produce an empty outer partition")

    first_calibration_start = _timestamp(ordered[train_end].start, name="calibration start")
    first_test_start = _timestamp(ordered[calibration_end].start, name="test start")
    protected_context_hours = max(event_pre_hours, command_history_hours)
    protected_start = first_calibration_start - pd.Timedelta(hours=protected_context_hours)
    pretraining_end = protected_start - pd.Timedelta(hours=ssl_context_hours + ssl_forecast_hours)

    return TemporalProtocol(
        outer_train_event_ids=tuple(event_ids[:train_end]),
        calibration_event_ids=tuple(event_ids[train_end:calibration_end]),
        test_event_ids=tuple(event_ids[calibration_end:]),
        mission_start=_timestamp(mission_start, name="mission_start"),
        pretraining_start=_timestamp(mission_start, name="mission_start"),
        pretraining_end=pretraining_end,
        protected_start=protected_start,
        first_calibration_event_start=first_calibration_start,
        first_test_event_start=first_test_start,
        event_pre_hours=float(event_pre_hours),
        event_post_hours=float(event_post_hours),
        command_history_hours=float(command_history_hours),
        ssl_context_hours=float(ssl_context_hours),
        ssl_forecast_hours=float(ssl_forecast_hours),
    )


def rolling_origin_folds(
    outer_train_event_ids: Sequence[str], *, fold_count: int = 3
) -> tuple[RollingOriginFold, ...]:
    """Create expanding-window folds inside the outer-training events."""

    event_ids = tuple(outer_train_event_ids)
    if fold_count < 1:
        raise ValueError("fold_count must be positive")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("outer-training event IDs must be unique")
    if len(event_ids) < 2 * fold_count + 1:
        raise ValueError("too few outer-training events for distinct rolling-origin folds")

    fit_fractions: tuple[float, ...]
    if fold_count == 1:
        fit_fractions = (0.8,)
    else:
        fit_fractions = tuple(0.5 + (0.3 * index / (fold_count - 1)) for index in range(fold_count))
    fit_ends = tuple(int(len(event_ids) * fraction) for fraction in fit_fractions)
    if len(set(fit_ends)) != fold_count:
        raise ValueError("too few events to construct distinct rolling-origin cutoffs")

    folds: list[RollingOriginFold] = []
    for index, fit_end in enumerate(fit_ends):
        validation_end = fit_ends[index + 1] if index + 1 < fold_count else len(event_ids)
        folds.append(
            RollingOriginFold(
                index=index + 1,
                fit_event_ids=event_ids[:fit_end],
                validation_event_ids=event_ids[fit_end:validation_end],
            )
        )
    return tuple(folds)
