"""ESA Mission 1 streaming loaders and event-centred episode construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MISSION1_ARCHIVE_MD5 = "80750189d171f5f398fb3d96c49df12b"
_MISSION1_METADATA_FILES = (
    "anomaly_types.csv",
    "channels.csv",
    "labels.csv",
    "telecommands.csv",
)


def archive_md5(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the streaming MD5 required to verify the published ESA archive."""

    source = Path(path)
    digest = hashlib.md5(usedforsecurity=False)
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_mission1_archive(
    archive_path: str | Path,
    raw_dir: str | Path,
    *,
    expected_md5: str = MISSION1_ARCHIVE_MD5,
) -> Path:
    """Verify and extract the Mission 1 ZIP if needed."""

    archive = Path(archive_path)
    destination = Path(raw_dir)
    if all((destination / name).is_file() for name in _MISSION1_METADATA_FILES):
        return destination
    if not archive.is_file():
        raise FileNotFoundError(
            f"Mission 1 archive not found: {archive}. Download ESA-Mission1.zip "
            "from https://zenodo.org/records/12528696"
        )
    observed_md5 = archive_md5(archive)
    if observed_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"Mission 1 archive checksum mismatch: expected {expected_md5}, observed {observed_md5}"
        )

    extraction_root = destination.parent
    extraction_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source:
        root = extraction_root.resolve()
        for member in source.infolist():
            target = (extraction_root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"unsafe archive member path: {member.filename}")
        source.extractall(extraction_root)
    missing = [name for name in _MISSION1_METADATA_FILES if not (destination / name).is_file()]
    if missing:
        raise ValueError(f"Mission 1 archive is missing required files: {missing}")
    return destination


@dataclass(frozen=True)
class EventMetadata:
    event_id: str
    event_class: str
    event_subclass: str
    event_type: str
    start: pd.Timestamp
    end: pd.Timestamp
    affected_channels: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeConfig:
    pre_hours: float = 6.0
    post_hours: float = 6.0
    telecommand_history_hours: float = 24.0
    max_steps: int = 128
    max_commands: int = 64
    protocol: Literal["online_start", "online_end", "retrospective"] = "retrospective"

    def __post_init__(self) -> None:
        if min(self.pre_hours, self.post_hours, self.telecommand_history_hours) < 0:
            raise ValueError("episode context durations must be non-negative")
        if self.max_steps < 8 or self.max_commands < 1:
            raise ValueError("max_steps must be >= 8 and max_commands must be positive")
        if self.protocol not in {"online_start", "online_end", "retrospective"}:
            raise ValueError("unknown event-time protocol")


@dataclass(frozen=True)
class EventEpisode:
    event_id: str
    event_class: str
    event_type: str
    start_time: str
    telemetry: NDArray[np.float32]
    timestamps: NDArray[np.float32]
    observation_mask: NDArray[np.bool_]
    time_deltas: NDArray[np.float32]
    command_ids: NDArray[np.int64]
    command_times: NDArray[np.float32]
    affected_channel_mask: NDArray[np.float32]
    protocol: str = "retrospective"

    def __post_init__(self) -> None:
        steps, channels = self.telemetry.shape
        if self.timestamps.shape != (steps,):
            raise ValueError("timestamps must have one value per telemetry step")
        if self.observation_mask.shape != (steps, channels):
            raise ValueError("observation_mask must match telemetry")
        if self.time_deltas.shape != (steps, channels):
            raise ValueError("time_deltas must match telemetry")
        if self.affected_channel_mask.shape != (channels,):
            raise ValueError("affected_channel_mask must have one value per channel")
        if self.command_ids.shape != self.command_times.shape:
            raise ValueError("command IDs and times must align")
        if self.protocol not in {"online_start", "online_end", "retrospective"}:
            raise ValueError("unknown event-time protocol")


class Mission1Source:
    """Read the official nested-ZIP Mission 1 distribution without expanding pickles."""

    def __init__(self, root: str | Path = "data/mission1/raw/ESA-Mission1") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Mission 1 directory not found: {self.root}")

    @cached_property
    def channels(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "channels.csv")

    @cached_property
    def telecommands(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "telecommands.csv")

    @cached_property
    def labels(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "labels.csv")

    @cached_property
    def anomaly_types(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "anomaly_types.csv")

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.channels["Channel"].astype(str))

    @property
    def command_names(self) -> tuple[str, ...]:
        return tuple(self.telecommands["Telecommand"].astype(str))

    @cached_property
    def mission_start(self) -> pd.Timestamp:
        """Return the first observation in ESA's stable channel ordering."""

        first_channel = self.load_channel(self.channel_names[0])
        if first_channel.empty:
            raise ValueError("the first Mission 1 channel contains no observations")
        return pd.Timestamp(first_channel.index[0])

    @staticmethod
    def _read_nested_pickle(path: Path, member: str) -> pd.DataFrame:
        # The pickles are from the checksum-verified official ESA archive.
        with ZipFile(path) as archive:
            payload = archive.read(member)
        frame = pd.read_pickle(BytesIO(payload))
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1:
            raise ValueError(f"unexpected Mission 1 payload in {path}")
        frame = frame.sort_index()
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
        return frame

    def load_channel(self, channel: str) -> pd.DataFrame:
        if channel not in self.channel_names:
            raise ValueError(f"unknown Mission 1 channel: {channel}")
        return self._read_nested_pickle(self.root / "channels" / f"{channel}.zip", channel)

    def load_telecommand(self, command: str) -> pd.DataFrame:
        if command not in self.command_names:
            raise ValueError(f"unknown Mission 1 telecommand: {command}")
        return self._read_nested_pickle(self.root / "telecommands" / f"{command}.zip", command)

    def load_command_events(self, cache_path: str | Path | None = None) -> pd.DataFrame:
        """Load every command execution once and optionally cache a compact CSV."""

        target = None if cache_path is None else Path(cache_path)
        if target is not None and target.exists():
            cached = pd.read_csv(target, parse_dates=["timestamp"])
            cached["timestamp"] = pd.DatetimeIndex(cached["timestamp"]).tz_localize(None)
            return cached
        records: list[pd.DataFrame] = []
        for command_id, command in enumerate(self.command_names):
            frame = self.load_telecommand(command)
            records.append(
                pd.DataFrame(
                    {
                        "timestamp": frame.index,
                        "command_id": np.full(len(frame), command_id, dtype=np.int64),
                    }
                )
            )
        result = (
            pd.concat(records, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        )
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(target, index=False)
        return result


def build_event_index(source: Mission1Source) -> tuple[EventMetadata, ...]:
    """Collapse channel-level labels into one record per event."""

    labels = source.labels.copy()
    labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True).dt.tz_localize(None)
    labels["EndTime"] = pd.to_datetime(labels["EndTime"], utc=True).dt.tz_localize(None)
    event_types = source.anomaly_types.set_index("ID")
    events: list[EventMetadata] = []
    for event_id, group in labels.groupby("ID", sort=False):
        if event_id not in event_types.index:
            raise ValueError(f"event {event_id} has labels but no anomaly type")
        details = event_types.loc[event_id]
        events.append(
            EventMetadata(
                event_id=str(event_id),
                event_class=str(details["Class"]),
                event_subclass=str(details["Subclass"]),
                event_type=str(details["Category"]),
                start=pd.Timestamp(group["StartTime"].min()),
                end=pd.Timestamp(group["EndTime"].max()),
                affected_channels=tuple(sorted(group["Channel"].astype(str).unique())),
            )
        )
    return tuple(sorted(events, key=lambda event: (event.start, event.event_id)))


def chronological_split(
    events: tuple[EventMetadata, ...] | list[EventMetadata],
    *,
    train_fraction: float = 0.60,
    calibration_fraction: float = 0.20,
) -> dict[str, tuple[str, ...]]:
    """Create leakage-safe event-level chronological partitions."""

    if not 0 < train_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("split fractions must lie in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must leave a test partition")
    ordered = sorted(events, key=lambda event: (event.start, event.event_id))
    train_end = int(len(ordered) * train_fraction)
    calibration_end = train_end + int(len(ordered) * calibration_fraction)
    return {
        "train": tuple(event.event_id for event in ordered[:train_end]),
        "calibration": tuple(event.event_id for event in ordered[train_end:calibration_end]),
        "test": tuple(event.event_id for event in ordered[calibration_end:]),
    }


def eligible_event_classes(
    events: tuple[EventMetadata, ...] | list[EventMetadata],
    split: dict[str, tuple[str, ...]],
    *,
    min_train: int = 3,
    min_calibration: int = 1,
    min_test: int = 1,
) -> tuple[str, ...]:
    """Return classes with enough chronological support for calibrated testing."""

    by_id = {event.event_id: event for event in events}
    limits = {"train": min_train, "calibration": min_calibration, "test": min_test}
    counts: dict[str, dict[str, int]] = {}
    for partition, event_ids in split.items():
        for event_id in event_ids:
            event_class = by_id[event_id].event_class
            counts.setdefault(event_class, {})[partition] = (
                counts.setdefault(event_class, {}).get(partition, 0) + 1
            )
    return tuple(
        sorted(
            event_class
            for event_class, partition_counts in counts.items()
            if all(
                partition_counts.get(partition, 0) >= limit for partition, limit in limits.items()
            )
        )
    )


def _window(event: EventMetadata, config: EpisodeConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = event.start - pd.Timedelta(hours=config.pre_hours)
    if config.protocol == "online_start":
        end = event.start
    elif config.protocol == "online_end":
        end = event.end
    else:
        end = event.end + pd.Timedelta(hours=config.post_hours)
    return start, end


def _choose_grid(
    candidates: set[np.datetime64], event: EventMetadata, config: EpisodeConfig
) -> NDArray[np.datetime64]:
    start, end = _window(event, config)
    boundaries = [start.to_datetime64(), event.start.to_datetime64()]
    if config.protocol != "online_start":
        boundaries.append(event.end.to_datetime64())
    if config.protocol == "retrospective":
        boundaries.append(end.to_datetime64())
    mandatory = np.unique(np.asarray(boundaries, dtype="datetime64[ns]"))
    values = np.unique(
        np.concatenate((np.asarray(list(candidates), dtype="datetime64[ns]"), mandatory))
    )
    if len(values) <= config.max_steps:
        return values
    mandatory_ns = set(mandatory.astype(np.int64).tolist())
    optional = np.asarray(
        [value for value in values if int(value.astype(np.int64)) not in mandatory_ns]
    )
    slots = config.max_steps - len(mandatory)
    positions = np.linspace(0, len(optional) - 1, max(slots, 0), dtype=int)
    return np.unique(np.concatenate((mandatory, optional[positions])))


def build_event_episodes(
    source: Mission1Source,
    events: tuple[EventMetadata, ...] | list[EventMetadata],
    *,
    config: EpisodeConfig | None = None,
    command_cache: str | Path | None = None,
    normalizer: Any | None = None,
) -> tuple[EventEpisode, ...]:
    """Build bounded irregular episodes from the telemetry channels."""

    config = config or EpisodeConfig()
    selected = tuple(events)
    if not selected:
        return ()
    channels = source.channel_names
    candidate_times: dict[str, set[np.datetime64]] = {event.event_id: set() for event in selected}
    per_channel_points = max(2, config.max_steps // max(1, len(channels)))
    for channel in channels:
        frame = source.load_channel(channel)
        index = frame.index.to_numpy(dtype="datetime64[ns]")
        for event in selected:
            start, end = _window(event, config)
            left = int(index.searchsorted(start.to_datetime64(), side="left"))
            right = int(index.searchsorted(end.to_datetime64(), side="right"))
            if right <= left:
                continue
            positions = np.linspace(
                left, right - 1, min(per_channel_points, right - left), dtype=int
            )
            candidate_times[event.event_id].update(index[positions].tolist())
        del frame
    grids = {
        event.event_id: _choose_grid(candidate_times[event.event_id], event, config)
        for event in selected
    }
    telemetry = {
        event.event_id: np.zeros((len(grids[event.event_id]), len(channels)), dtype=np.float32)
        for event in selected
    }
    observed = {
        event.event_id: np.zeros_like(telemetry[event.event_id], dtype=np.bool_)
        for event in selected
    }
    deltas = {
        event.event_id: np.zeros_like(telemetry[event.event_id], dtype=np.float32)
        for event in selected
    }
    for channel_index, channel in enumerate(channels):
        frame = source.load_channel(channel)
        index = frame.index.to_numpy(dtype="datetime64[ns]")
        values = frame.iloc[:, 0].to_numpy(dtype=np.float32)
        if normalizer is not None:
            values = np.asarray(normalizer.transform_channel(channel, values), dtype=np.float32)
        for event in selected:
            grid = grids[event.event_id]
            positions = index.searchsorted(grid, side="right") - 1
            valid = positions >= 0
            clipped = positions.clip(min=0)
            telemetry[event.event_id][valid, channel_index] = values[clipped[valid]]
            exact = valid & (index[clipped] == grid)
            observed[event.event_id][:, channel_index] = exact
            elapsed = (grid[valid] - index[clipped[valid]]) / np.timedelta64(1, "h")
            deltas[event.event_id][valid, channel_index] = np.asarray(elapsed, dtype=np.float32)
        del frame
    commands = source.load_command_events(command_cache)
    command_timestamps = pd.DatetimeIndex(commands["timestamp"]).tz_localize(None)
    episodes: list[EventEpisode] = []
    channel_lookup = {name: index for index, name in enumerate(channels)}
    for event in selected:
        window_start, _ = _window(event, config)
        command_start = event.start - pd.Timedelta(hours=config.telecommand_history_hours)
        command_end = event.start if config.protocol == "online_start" else event.end
        command_rows = commands.loc[
            (command_timestamps >= command_start) & (command_timestamps <= command_end)
        ].tail(config.max_commands)
        command_times = (
            (pd.DatetimeIndex(command_rows["timestamp"]) - window_start).total_seconds() / 3600
        ).to_numpy(dtype=np.float32)
        grid = grids[event.event_id]
        times = ((grid - grid[0]) / np.timedelta64(1, "h")).astype(np.float32)
        affected = np.zeros(len(channels), dtype=np.float32)
        for channel in event.affected_channels:
            if channel in channel_lookup:
                affected[channel_lookup[channel]] = 1.0
        episodes.append(
            EventEpisode(
                event_id=event.event_id,
                event_class=event.event_class,
                event_type=event.event_type,
                start_time=event.start.isoformat(),
                telemetry=telemetry[event.event_id],
                timestamps=times,
                observation_mask=observed[event.event_id],
                time_deltas=deltas[event.event_id],
                command_ids=command_rows["command_id"].to_numpy(dtype=np.int64),
                command_times=command_times,
                affected_channel_mask=affected,
                protocol=config.protocol,
            )
        )
    return tuple(episodes)


def save_episode_cache(
    episodes: tuple[EventEpisode, ...],
    cache_dir: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for episode in episodes:
        path = destination / f"{episode.event_id}.npz"
        np.savez_compressed(
            path,
            telemetry=episode.telemetry,
            timestamps=episode.timestamps,
            observation_mask=episode.observation_mask,
            time_deltas=episode.time_deltas,
            command_ids=episode.command_ids,
            command_times=episode.command_times,
            affected_channel_mask=episode.affected_channel_mask,
        )
        manifest.append(
            {
                "event_id": episode.event_id,
                "event_class": episode.event_class,
                "event_type": episode.event_type,
                "start_time": episode.start_time,
                "protocol": episode.protocol,
                "file": path.name,
            }
        )
    manifest_path = destination / "episodes.json"
    payload = {
        "schema_version": "mirage-event-cache/2.0",
        "metadata": metadata or {},
        "episodes": manifest,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def load_episode_cache(cache_dir: str | Path) -> tuple[EventEpisode, ...]:
    source = Path(cache_dir)
    payload = json.loads((source / "episodes.json").read_text(encoding="utf-8"))
    manifest = payload if isinstance(payload, list) else payload["episodes"]
    episodes: list[EventEpisode] = []
    for record in manifest:
        with np.load(source / record["file"], allow_pickle=False) as payload:
            episodes.append(
                EventEpisode(
                    event_id=str(record["event_id"]),
                    event_class=str(record["event_class"]),
                    event_type=str(record["event_type"]),
                    start_time=str(record["start_time"]),
                    telemetry=payload["telemetry"],
                    timestamps=payload["timestamps"],
                    observation_mask=payload["observation_mask"],
                    time_deltas=payload["time_deltas"],
                    command_ids=payload["command_ids"],
                    command_times=payload["command_times"],
                    affected_channel_mask=payload["affected_channel_mask"],
                    protocol=str(record.get("protocol", "retrospective")),
                )
            )
    return tuple(episodes)
