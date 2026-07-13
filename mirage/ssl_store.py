"""lossless, memory-mapped Mission 1 storage for self-supervised learning."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mirage.preprocessing import MISSION1_ARCHIVE_MD5, Mission1Source

if TYPE_CHECKING:
    from mirage.normalization import ChannelNormalizer
    from mirage.protocol import TemporalProtocol


STORE_SCHEMA_VERSION = "mirage-ssl-store/1.0"
BLOCK_INDEX_SCHEMA_VERSION = "mirage-ssl-block-index/1.0"


class _Normalizer(Protocol):
    @property
    def scaler_hash(self) -> str: ...

    def save(self, path: str | Path) -> Any: ...


def _timestamp(value: Any, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"{field} must be a valid timestamp")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _timestamp_ns(value: Any, *, field: str) -> int:
    return int(_timestamp(value, field=field).value)


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _timestamp(value, field="timestamp").isoformat()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _protocol_payload(protocol: Any) -> dict[str, Any]:
    if hasattr(protocol, "to_dict"):
        payload = protocol.to_dict()
    elif dataclasses.is_dataclass(protocol):
        payload = _json_value(protocol)
    else:
        fields = (
            "outer_train_event_ids",
            "calibration_event_ids",
            "test_event_ids",
            "mission_start",
            "pretraining_start",
            "pretraining_end",
            "protected_start",
            "first_calibration_event_start",
            "first_test_event_start",
            "event_pre_hours",
            "event_post_hours",
            "command_history_hours",
            "ssl_context_hours",
            "ssl_forecast_hours",
        )
        payload = {field: _json_value(getattr(protocol, field)) for field in fields if hasattr(protocol, field)}
    if not isinstance(payload, dict):
        raise TypeError("TemporalProtocol.to_dict() must return a mapping")
    return _json_value(payload)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _protocol_hash(protocol: Any, payload: dict[str, Any]) -> str:
    value = getattr(protocol, "protocol_hash", None)
    if callable(value):
        value = value()
    if value:
        return str(value)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalizer_hash(normalizer: _Normalizer | None, scaler_path: Path) -> str:
    if normalizer is not None:
        value = getattr(normalizer, "scaler_hash", None)
        if callable(value):
            value = value()
        if value:
            return str(value)
    if scaler_path.is_file():
        return hashlib.sha256(scaler_path.read_bytes()).hexdigest()
    return "not-fitted"


def _deduplicate_channel(frame: pd.DataFrame) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """Return timestamp-sorted numeric values, keeping the first duplicate."""

    if frame.shape[1] != 1:
        raise ValueError("Mission 1 channel frames must have exactly one value column")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    timestamps = index.asi8.astype(np.int64, copy=False)
    numeric = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.float32)
    if not index.is_monotonic_increasing:
        order = np.argsort(timestamps, kind="mergesort")
        timestamps = timestamps[order]
        numeric = numeric[order]
    if len(timestamps) > 1:
        keep = np.concatenate((np.asarray([True]), timestamps[1:] != timestamps[:-1]))
        timestamps = timestamps[keep]
        numeric = numeric[keep]
    return timestamps, numeric


def _block_arrays(
    *,
    pretraining_start_ns: int,
    pretraining_end_ns: int,
    protected_start_ns: int,
    block_hours: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if not np.isfinite(block_hours) or block_hours <= 0:
        raise ValueError("block_hours must be positive and finite")
    nominal_duration_ns = int(pd.Timedelta(hours=float(block_hours)).value)
    if nominal_duration_ns <= 0:
        raise ValueError("block_hours is below nanosecond resolution")
    if pretraining_end_ns < pretraining_start_ns:
        raise ValueError("pretraining_end must not precede pretraining_start")
    interval_ns = protected_start_ns - pretraining_start_ns
    if interval_ns <= 0:
        raise ValueError("protected_start must follow pretraining_start")
    protected_window_ns = protected_start_ns - pretraining_end_ns
    minimum_duration_ns = max(nominal_duration_ns, protected_window_ns)
    count = max(1, interval_ns // minimum_duration_ns)

    # Equal partitions keep the full safe interval without dropping the tail.
    offsets = np.asarray(
        [(interval_ns * position) // count for position in range(count + 1)],
        dtype=np.int64,
    )
    boundaries = pretraining_start_ns + offsets
    starts = boundaries[:-1]
    ends = boundaries[1:]
    if len(starts) and int(starts[-1]) > pretraining_end_ns:
        raise ValueError(
            "a self-supervised block starts after pretraining_end; correct the temporal protocol"
        )
    if int(ends[-1]) != protected_start_ns:
        raise AssertionError("SSL block partition failed to tile the protected interval")
    return starts, ends


def build_block_index(
    store_dir: str | Path,
    *,
    pretraining_start: Any,
    pretraining_end: Any,
    protected_start: Any,
    block_hours: float,
) -> Path:
    """Build position ranges for every non-overlapping training block."""

    destination = Path(store_dir)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    channel_order = tuple(str(name) for name in manifest["channel_order"])
    start_ns = _timestamp_ns(pretraining_start, field="pretraining_start")
    last_start_ns = _timestamp_ns(pretraining_end, field="pretraining_end")
    protected_ns = _timestamp_ns(protected_start, field="protected_start")
    block_starts, block_ends = _block_arrays(
        pretraining_start_ns=start_ns,
        pretraining_end_ns=last_start_ns,
        protected_start_ns=protected_ns,
        block_hours=block_hours,
    )
    channel_starts = np.zeros((len(block_starts), len(channel_order)), dtype=np.int64)
    channel_ends = np.zeros_like(channel_starts)
    for channel_id in range(len(channel_order)):
        timestamps = np.load(
            destination / "channels" / f"channel_{channel_id:03d}" / "timestamps_ns.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        channel_starts[:, channel_id] = np.searchsorted(timestamps, block_starts, side="left")
        channel_ends[:, channel_id] = np.searchsorted(timestamps, block_ends, side="left")

    command_timestamps = np.load(
        destination / "commands" / "timestamps_ns.npy", mmap_mode="r", allow_pickle=False
    )
    command_starts = np.searchsorted(command_timestamps, block_starts, side="left").astype(
        np.int64
    )
    command_ends = np.searchsorted(command_timestamps, block_ends, side="left").astype(np.int64)
    target_points_by_block = (channel_ends - channel_starts).sum(axis=1, dtype=np.int64)
    path = destination / "block_index.npz"
    np.savez(
        path,
        schema_version=np.asarray(BLOCK_INDEX_SCHEMA_VERSION),
        block_ids=np.arange(len(block_starts), dtype=np.int64),
        block_starts_ns=block_starts,
        block_ends_ns=block_ends,
        channel_start_positions=channel_starts,
        channel_end_positions=channel_ends,
        command_start_positions=command_starts,
        command_end_positions=command_ends,
        target_points_by_block=target_points_by_block,
    )
    manifest["block_hours"] = float(block_hours)
    manifest["block_count"] = len(block_starts)
    durations_ns = block_ends - block_starts
    manifest["minimum_block_hours"] = float(durations_ns.min() / 3_600_000_000_000)
    manifest["maximum_block_hours"] = float(durations_ns.max() / 3_600_000_000_000)
    manifest["pretraining_points_total"] = int(target_points_by_block.sum())
    manifest["max_pretraining_target_timestamp"] = _max_indexed_timestamp(
        destination, channel_ends, channel_order
    )
    manifest["max_pretraining_command_timestamp"] = _max_indexed_command_timestamp(
        destination, command_ends
    )
    _write_json(destination / "manifest.json", manifest)
    return path


def _max_indexed_timestamp(
    store_dir: Path,
    channel_ends: NDArray[np.int64],
    channel_order: tuple[str, ...],
) -> str | None:
    maximum: int | None = None
    for channel_id in range(len(channel_order)):
        end = int(channel_ends[-1, channel_id]) if len(channel_ends) else 0
        if end == 0:
            continue
        timestamps = np.load(
            store_dir / "channels" / f"channel_{channel_id:03d}" / "timestamps_ns.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        value = int(timestamps[end - 1])
        maximum = value if maximum is None else max(maximum, value)
    return None if maximum is None else pd.Timestamp(maximum).isoformat()


def _max_indexed_command_timestamp(
    store_dir: Path, command_ends: NDArray[np.int64]
) -> str | None:
    end = int(command_ends[-1]) if len(command_ends) else 0
    if end == 0:
        return None
    timestamps = np.load(
        store_dir / "commands" / "timestamps_ns.npy", mmap_mode="r", allow_pickle=False
    )
    return pd.Timestamp(int(timestamps[end - 1])).isoformat()


def build_ssl_store(
    source: Mission1Source,
    protocol: TemporalProtocol | Any,
    normalizer: ChannelNormalizer | _Normalizer | None = None,
    output_dir: str | Path = "data/mission1/ssl_store",
    *,
    block_hours: float = 6.0,
    archive_md5: str = MISSION1_ARCHIVE_MD5,
    normalization_clip: float = 10.0,
    normalization_epsilon: float = 1.0e-6,
) -> Path:
    """Convert the protected Mission 1 prefix into memory-mappable arrays."""

    destination = Path(output_dir)
    channels_dir = destination / "channels"
    commands_dir = destination / "commands"
    channels_dir.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)

    mission_start_ns = _timestamp_ns(protocol.mission_start, field="mission_start")
    pretraining_start_ns = _timestamp_ns(
        protocol.pretraining_start, field="pretraining_start"
    )
    protected_start_ns = _timestamp_ns(protocol.protected_start, field="protected_start")
    if protected_start_ns <= mission_start_ns:
        raise ValueError("protected_start must follow mission_start")

    raw_points_by_channel: dict[str, int] = {}
    bounds_by_channel: dict[str, dict[str, str | None]] = {}
    auto_fit_normalizer = normalizer is None
    fitted_parameters: dict[str, Any] = {}
    for channel_id, channel in enumerate(source.channel_names):
        frame = source.load_channel(channel)
        timestamps, values = _deduplicate_channel(frame)
        del frame
        allowed = (timestamps >= mission_start_ns) & (timestamps <= protected_start_ns)
        timestamps = timestamps[allowed]
        values = values[allowed]
        channel_dir = channels_dir / f"channel_{channel_id:03d}"
        channel_dir.mkdir(parents=True, exist_ok=True)
        np.save(channel_dir / "timestamps_ns.npy", timestamps, allow_pickle=False)
        np.save(channel_dir / "values.npy", values, allow_pickle=False)
        raw_points_by_channel[channel] = len(values)
        bounds_by_channel[channel] = {
            "start": None if len(timestamps) == 0 else pd.Timestamp(int(timestamps[0])).isoformat(),
            "end": None if len(timestamps) == 0 else pd.Timestamp(int(timestamps[-1])).isoformat(),
        }
        if auto_fit_normalizer:
            from mirage.normalization import ChannelNormalizer

            in_pretraining = (timestamps >= pretraining_start_ns) & (
                timestamps < protected_start_ns
            )
            fitted_parameters[channel] = ChannelNormalizer.fit_channel(
                values[in_pretraining], epsilon=normalization_epsilon
            )

    if auto_fit_normalizer:
        from mirage.normalization import ChannelNormalizer

        normalizer = ChannelNormalizer(
            channel_order=source.channel_names,
            parameters=fitted_parameters,
            clip=normalization_clip,
            epsilon=normalization_epsilon,
            fit_start=_timestamp(protocol.pretraining_start, field="pretraining_start"),
            fit_end=pd.Timestamp(protected_start_ns - 1),
        )
    elif (normalizer_order := getattr(normalizer, "channel_order", None)) is not None and (
        tuple(normalizer_order) != source.channel_names
    ):
        raise ValueError("normalizer channel order does not match Mission 1 source")
    fit_end = None if normalizer is None else getattr(normalizer, "fit_end", None)
    if fit_end is not None:
        fit_end_ns = _timestamp_ns(fit_end, field="normalizer.fit_end")
        if fit_end_ns >= protected_start_ns:
            raise ValueError("normalizer includes values at or after protected_start")

    command_events = source.load_command_events()
    command_index = pd.DatetimeIndex(command_events["timestamp"])
    if command_index.tz is not None:
        command_index = command_index.tz_convert("UTC").tz_localize(None)
    command_timestamps = command_index.asi8.astype(np.int64, copy=False)
    command_ids = command_events["command_id"].to_numpy(dtype=np.int32)
    allowed_commands = (command_timestamps >= mission_start_ns) & (
        command_timestamps <= protected_start_ns
    )
    command_timestamps = command_timestamps[allowed_commands]
    command_ids = command_ids[allowed_commands]
    order = np.lexsort((command_ids, command_timestamps))
    command_timestamps = command_timestamps[order]
    command_ids = command_ids[order]
    command_dtype = np.int16 if len(source.command_names) <= np.iinfo(np.int16).max else np.int32
    np.save(commands_dir / "timestamps_ns.npy", command_timestamps, allow_pickle=False)
    np.save(commands_dir / "command_ids.npy", command_ids.astype(command_dtype), allow_pickle=False)

    protocol_payload = _protocol_payload(protocol)
    if hasattr(protocol, "save"):
        protocol.save(destination / "protocol.json")
    else:
        _write_json(destination / "protocol.json", protocol_payload)
    scaler_path = destination / "scaler.json"
    if normalizer is not None:
        normalizer.save(scaler_path)
    manifest: dict[str, Any] = {
        "schema_version": STORE_SCHEMA_VERSION,
        "archive_md5": archive_md5,
        "mission": "ESA Mission 1",
        "channel_count": len(source.channel_names),
        "command_count": len(source.command_names),
        "mission_start": _timestamp(protocol.mission_start, field="mission_start").isoformat(),
        "pretraining_start": _timestamp(
            protocol.pretraining_start, field="pretraining_start"
        ).isoformat(),
        "protected_start": _timestamp(protocol.protected_start, field="protected_start").isoformat(),
        "pretraining_end": _timestamp(
            protocol.pretraining_end, field="pretraining_end"
        ).isoformat(),
        "raw_points_by_channel": raw_points_by_channel,
        "raw_points_total": int(sum(raw_points_by_channel.values())),
        "command_executions_total": len(command_ids),
        "channel_bounds": bounds_by_channel,
        "channel_order": list(source.channel_names),
        "command_order": list(source.command_names),
        "protocol_hash": _protocol_hash(protocol, protocol_payload),
        "scaler_hash": _normalizer_hash(normalizer, scaler_path),
    }
    _write_json(destination / "manifest.json", manifest)
    build_block_index(
        destination,
        pretraining_start=protocol.pretraining_start,
        pretraining_end=protocol.pretraining_end,
        protected_start=protocol.protected_start,
        block_hours=block_hours,
    )
    return destination


class SSLStore:
    """Read-only view over a validated MIRAGE self-supervised store."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest: dict[str, Any] = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported SSL store schema: {self.manifest.get('schema_version')!r}"
            )
        self.channel_order = tuple(str(name) for name in self.manifest["channel_order"])
        self.command_order = tuple(str(name) for name in self.manifest["command_order"])
        with np.load(self.root / "block_index.npz", allow_pickle=False) as index:
            schema = str(index["schema_version"].item())
            if schema != BLOCK_INDEX_SCHEMA_VERSION:
                raise ValueError(f"unsupported SSL block index schema: {schema!r}")
            self.block_ids = index["block_ids"].astype(np.int64, copy=True)
            self.block_starts_ns = index["block_starts_ns"].astype(np.int64, copy=True)
            self.block_ends_ns = index["block_ends_ns"].astype(np.int64, copy=True)
            self.channel_start_positions = index["channel_start_positions"].astype(
                np.int64, copy=True
            )
            self.channel_end_positions = index["channel_end_positions"].astype(
                np.int64, copy=True
            )
            self.command_start_positions = index["command_start_positions"].astype(
                np.int64, copy=True
            )
            self.command_end_positions = index["command_end_positions"].astype(
                np.int64, copy=True
            )
            self.target_points_by_block = index["target_points_by_block"].astype(
                np.int64, copy=True
            )
        expected_shape = (len(self.block_ids), len(self.channel_order))
        if self.channel_start_positions.shape != expected_shape:
            raise ValueError("SSL block/channel start index shape does not match manifest")
        if self.channel_end_positions.shape != expected_shape:
            raise ValueError("SSL block/channel end index shape does not match manifest")
        if np.any(self.channel_end_positions < self.channel_start_positions):
            raise ValueError("SSL block index contains reversed channel positions")
        if len(self.block_ids) and (
            np.any(self.block_starts_ns[1:] < self.block_ends_ns[:-1])
            or np.any(self.block_ends_ns <= self.block_starts_ns)
        ):
            raise ValueError("SSL blocks overlap or have non-positive duration")
        protected_ns = _timestamp_ns(self.manifest["protected_start"], field="protected_start")
        if len(self.block_ends_ns) and int(self.block_ends_ns.max()) > protected_ns:
            raise ValueError("SSL block index crosses the protected boundary")
        self._channel_cache: dict[
            int, tuple[NDArray[np.int64], NDArray[np.float32]]
        ] = {}
        self._command_cache: tuple[
            NDArray[np.int64], NDArray[np.signedinteger[Any]]
        ] | None = None

    @classmethod
    def open(cls, root: str | Path) -> SSLStore:
        return cls(root)

    load = open

    def channel_arrays(
        self, channel_id: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        if not 0 <= channel_id < len(self.channel_order):
            raise IndexError(f"channel_id out of range: {channel_id}")
        if channel_id in self._channel_cache:
            return self._channel_cache[channel_id]
        directory = self.root / "channels" / f"channel_{channel_id:03d}"
        timestamps = np.load(directory / "timestamps_ns.npy", mmap_mode="r", allow_pickle=False)
        values = np.load(directory / "values.npy", mmap_mode="r", allow_pickle=False)
        if timestamps.dtype != np.int64 or values.dtype != np.float32:
            raise ValueError(f"invalid array dtype for channel {channel_id}")
        if len(timestamps) != len(values):
            raise ValueError(f"timestamp/value length mismatch for channel {channel_id}")
        self._channel_cache[channel_id] = (timestamps, values)
        return self._channel_cache[channel_id]

    def command_arrays(self) -> tuple[NDArray[np.int64], NDArray[np.signedinteger[Any]]]:
        if self._command_cache is not None:
            return self._command_cache
        timestamps = np.load(
            self.root / "commands" / "timestamps_ns.npy", mmap_mode="r", allow_pickle=False
        )
        command_ids = np.load(
            self.root / "commands" / "command_ids.npy", mmap_mode="r", allow_pickle=False
        )
        if timestamps.dtype != np.int64 or not np.issubdtype(command_ids.dtype, np.signedinteger):
            raise ValueError("invalid command store dtypes")
        if len(timestamps) != len(command_ids):
            raise ValueError("command timestamp/ID length mismatch")
        self._command_cache = (timestamps, command_ids)
        return self._command_cache

    def channel_block(
        self, block_index: int, channel_id: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        timestamps, values = self.channel_arrays(channel_id)
        start = int(self.channel_start_positions[block_index, channel_id])
        end = int(self.channel_end_positions[block_index, channel_id])
        return timestamps[start:end], values[start:end]

    def command_block(
        self, block_index: int
    ) -> tuple[NDArray[np.int64], NDArray[np.signedinteger[Any]]]:
        timestamps, command_ids = self.command_arrays()
        start = int(self.command_start_positions[block_index])
        end = int(self.command_end_positions[block_index])
        return timestamps[start:end], command_ids[start:end]

    @property
    def pretraining_points_total(self) -> int:
        return int(self.target_points_by_block.sum())


SSLMemmapStore = SSLStore
