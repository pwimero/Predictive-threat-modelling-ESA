"""Exact train-period-only robust channel normalisation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from mirage.protocol import TemporalProtocol, canonical_sha256

NORMALIZER_SCHEMA_VERSION = "mirage-channel-normalizer/1.0"


class ChannelSource(Protocol):
    """Structural type implemented by :class:`mirage.preprocessing.Mission1Source`."""

    @property
    def channel_names(self) -> tuple[str, ...]: ...

    def load_channel(self, channel: str) -> pd.DataFrame: ...


def _normalise_timestamp(value: object, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


@dataclass(frozen=True)
class ChannelStatistics:
    """Exact descriptive and scaling statistics for one telemetry channel."""

    median: float
    scale: float
    q01: float
    q99: float
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("normalisation statistics require at least one finite value")
        if not all(math.isfinite(value) for value in (self.median, self.scale, self.q01, self.q99)):
            raise ValueError("normalisation statistics must be finite")
        if self.scale <= 0:
            raise ValueError("channel scale must be positive")
        if self.q01 > self.q99:
            raise ValueError("q01 cannot exceed q99")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "median": self.median,
            "scale": self.scale,
            "q01": self.q01,
            "q99": self.q99,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChannelStatistics:
        return cls(
            median=float(payload["median"]),
            scale=float(payload["scale"]),
            q01=float(payload["q01"]),
            q99=float(payload["q99"]),
            count=int(payload["count"]),
        )


def exact_channel_statistics(values: ArrayLike, *, epsilon: float = 1.0e-6) -> ChannelStatistics:
    """Calculate the plan's exact robust statistics from finite values."""

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        raise ValueError("cannot fit a channel normalizer without finite values")

    median = float(np.median(numeric))
    q01, q25, q75, q99 = np.quantile(numeric, (0.01, 0.25, 0.75, 0.99))
    iqr = float(q75 - q25)
    mad_scale = float(1.4826 * np.median(np.abs(numeric - median)))
    standard_deviation = float(np.std(numeric, ddof=0))
    robust_scale = max(iqr, mad_scale, standard_deviation, epsilon)
    # The declared constant-channel fallback is 1.0 rather than epsilon.
    scale = 1.0 if robust_scale <= epsilon else robust_scale
    return ChannelStatistics(
        median=median,
        scale=scale,
        q01=float(q01),
        q99=float(q99),
        count=int(numeric.size),
    )


@dataclass(frozen=True)
class ChannelNormalizer:
    """One immutable transform shared by SSL and every event partition."""

    channel_order: tuple[str, ...]
    parameters: Mapping[str, ChannelStatistics]
    clip: float = 10.0
    epsilon: float = 1.0e-6
    fit_start: pd.Timestamp | None = None
    fit_end: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if not self.channel_order or len(self.channel_order) != len(set(self.channel_order)):
            raise ValueError("channel_order must contain unique channel names")
        if any(not channel for channel in self.channel_order):
            raise ValueError("channel names must be non-empty")
        missing = set(self.channel_order).difference(self.parameters)
        extra = set(self.parameters).difference(self.channel_order)
        if missing or extra:
            raise ValueError(
                f"normalizer parameters do not match channel order; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if not math.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("clip must be finite and positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")

        ordered_parameters = {
            channel: self.parameters[channel] for channel in self.channel_order
        }
        object.__setattr__(self, "parameters", MappingProxyType(ordered_parameters))
        if self.fit_start is not None:
            object.__setattr__(
                self, "fit_start", _normalise_timestamp(self.fit_start, name="fit_start")
            )
        if self.fit_end is not None:
            object.__setattr__(self, "fit_end", _normalise_timestamp(self.fit_end, name="fit_end"))
        if (self.fit_start is None) != (self.fit_end is None):
            raise ValueError("fit_start and fit_end must either both be set or both be omitted")
        if self.fit_start is not None and self.fit_end is not None and self.fit_start > self.fit_end:
            raise ValueError("fit_start cannot be after fit_end")

    @classmethod
    def fit(
        cls,
        values_by_channel: Mapping[str, ArrayLike],
        *,
        channel_order: Sequence[str] | None = None,
        clip: float = 10.0,
        epsilon: float = 1.0e-6,
    ) -> ChannelNormalizer:
        """Fit exact statistics from already-clipped training arrays."""

        order = tuple(values_by_channel) if channel_order is None else tuple(channel_order)
        if set(order) != set(values_by_channel):
            raise ValueError("channel_order must exactly match values_by_channel")
        parameters = {
            channel: exact_channel_statistics(values_by_channel[channel], epsilon=epsilon)
            for channel in order
        }
        return cls(order, parameters, clip=clip, epsilon=epsilon)

    fit_channels = fit

    @staticmethod
    def fit_channel(values: ArrayLike, *, epsilon: float = 1.0e-6) -> ChannelStatistics:
        """Fit one exact channel transform, primarily for streaming store builders."""

        return exact_channel_statistics(values, epsilon=epsilon)

    @classmethod
    def fit_timestamped(
        cls,
        samples_by_channel: Mapping[str, tuple[ArrayLike, ArrayLike]],
        *,
        fit_start: object,
        fit_end: object,
        channel_order: Sequence[str] | None = None,
        clip: float = 10.0,
        epsilon: float = 1.0e-6,
    ) -> ChannelNormalizer:
        """Fit using only samples inside the inclusive declared training interval."""

        start = _normalise_timestamp(fit_start, name="fit_start")
        end = _normalise_timestamp(fit_end, name="fit_end")
        if start > end:
            raise ValueError("fit_start cannot be after fit_end")
        order = tuple(samples_by_channel) if channel_order is None else tuple(channel_order)
        if set(order) != set(samples_by_channel):
            raise ValueError("channel_order must exactly match samples_by_channel")

        clipped: dict[str, NDArray[np.float64]] = {}
        for channel in order:
            timestamps, values = samples_by_channel[channel]
            index = pd.DatetimeIndex(timestamps)
            if index.tz is not None:
                index = index.tz_convert("UTC").tz_localize(None)
            numeric = np.asarray(values, dtype=np.float64).reshape(-1)
            if len(index) != numeric.size:
                raise ValueError(f"timestamps and values do not align for {channel}")
            in_training = (index >= start) & (index <= end)
            clipped[channel] = numeric[np.asarray(in_training)]

        normalizer = cls.fit(clipped, channel_order=order, clip=clip, epsilon=epsilon)
        return cls(
            channel_order=normalizer.channel_order,
            parameters=normalizer.parameters,
            clip=normalizer.clip,
            epsilon=normalizer.epsilon,
            fit_start=start,
            fit_end=end,
        )

    def transform(self, channel: str | int, values: ArrayLike) -> NDArray[np.float32]:
        """Apply the saved transform; no statistics are recomputed."""

        channel_name = self.channel_order[channel] if isinstance(channel, int) else channel
        if channel_name not in self.parameters:
            raise ValueError(f"unknown normalizer channel: {channel_name}")
        statistics = self.parameters[channel_name]
        numeric = np.asarray(values, dtype=np.float64)
        transformed = np.clip(
            (numeric - statistics.median) / statistics.scale,
            -self.clip,
            self.clip,
        )
        return transformed.astype(np.float32, copy=False)

    def transform_channel(
        self, channel: str | int, values: ArrayLike
    ) -> NDArray[np.float32]:
        """Plan-named alias for :meth:`transform`."""

        return self.transform(channel, values)

    def transform_matrix(self, values: ArrayLike) -> NDArray[np.float32]:
        """Transform a matrix whose last dimension follows ``channel_order``."""

        numeric = np.asarray(values, dtype=np.float64)
        if numeric.ndim < 1 or numeric.shape[-1] != len(self.channel_order):
            raise ValueError("matrix final dimension must match channel_order")
        result = np.empty(numeric.shape, dtype=np.float32)
        for index, channel in enumerate(self.channel_order):
            result[..., index] = self.transform(channel, numeric[..., index])
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZER_SCHEMA_VERSION,
            "channel_order": list(self.channel_order),
            "clip": self.clip,
            "epsilon": self.epsilon,
            "fit_start": None if self.fit_start is None else self.fit_start.isoformat(),
            "fit_end": None if self.fit_end is None else self.fit_end.isoformat(),
            "channels": {
                channel: self.parameters[channel].to_dict() for channel in self.channel_order
            },
        }

    @property
    def scaler_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def hash(self) -> str:
        """Short artifact-compatible alias for ``scaler_hash``."""

        return self.scaler_hash

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict() | {"scaler_hash": self.scaler_hash}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChannelNormalizer:
        schema = payload.get("schema_version")
        if schema != NORMALIZER_SCHEMA_VERSION:
            raise ValueError(f"unsupported channel normalizer schema: {schema!r}")
        raw_channels = payload.get("channels")
        if not isinstance(raw_channels, Mapping):
            raise ValueError("normalizer channels must be a mapping")
        normalizer = cls(
            channel_order=tuple(str(item) for item in payload["channel_order"]),
            parameters={
                str(channel): ChannelStatistics.from_dict(statistics)
                for channel, statistics in raw_channels.items()
            },
            clip=float(payload["clip"]),
            epsilon=float(payload["epsilon"]),
            fit_start=(None if payload.get("fit_start") is None else pd.Timestamp(payload["fit_start"])),
            fit_end=(None if payload.get("fit_end") is None else pd.Timestamp(payload["fit_end"])),
        )
        saved_hash = payload.get("scaler_hash")
        if saved_hash is not None and saved_hash != normalizer.scaler_hash:
            raise ValueError("channel normalizer hash mismatch")
        return normalizer

    @classmethod
    def load(cls, path: str | Path) -> ChannelNormalizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("channel normalizer file must contain a JSON object")
        return cls.from_dict(payload)

    @classmethod
    def fit_from_source(
        cls,
        source: ChannelSource,
        protocol: TemporalProtocol,
        *,
        clip: float = 10.0,
        epsilon: float = 1.0e-6,
    ) -> ChannelNormalizer:
        """Fit one channel at a time from the verified Mission 1 source."""

        return fit_train_only_normalizer(
            source.channel_names,
            source.load_channel,
            protocol,
            clip=clip,
            epsilon=epsilon,
        )


def fit_train_only_normalizer(
    channel_order: Sequence[str],
    channel_loader: Callable[[str], pd.DataFrame],
    protocol: TemporalProtocol,
    *,
    clip: float = 10.0,
    epsilon: float = 1.0e-6,
) -> ChannelNormalizer:
    """Stream channels and fit on every value strictly before protection."""

    parameters: dict[str, ChannelStatistics] = {}
    fit_end = protocol.protected_start - pd.Timedelta(nanoseconds=1)
    for channel in channel_order:
        frame = channel_loader(channel)
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1:
            raise ValueError(f"channel loader returned an invalid frame for {channel}")
        index = pd.DatetimeIndex(frame.index)
        if index.tz is not None:
            index = index.tz_convert("UTC").tz_localize(None)
        include = (index >= protocol.pretraining_start) & (index <= fit_end)
        parameters[channel] = exact_channel_statistics(
            frame.iloc[np.asarray(include), 0].to_numpy(), epsilon=epsilon
        )
    return ChannelNormalizer(
        channel_order=tuple(channel_order),
        parameters=parameters,
        clip=clip,
        epsilon=epsilon,
        fit_start=protocol.pretraining_start,
        fit_end=fit_end,
    )
