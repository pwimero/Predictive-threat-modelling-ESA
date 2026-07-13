"""Exhaustive, leakage-safe samples over the Mission 1 SSL memmap store."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset

from mirage.ssl_store import SSLStore

if TYPE_CHECKING:
    from mirage.normalization import ChannelNormalizer


_NS_PER_HOUR = 3_600_000_000_000


class _Normalizer(Protocol):
    def transform(self, channel: str, values: NDArray[np.floating[Any]]) -> NDArray[np.float32]: ...


@dataclass(frozen=True)
class ObservationTarget:
    channel_id: int
    relative_time: float
    value: float


@dataclass(frozen=True)
class PretrainingSample:
    telemetry: NDArray[np.float32]
    timestamps: NDArray[np.float32]
    observation_mask: NDArray[np.bool_]
    time_deltas: NDArray[np.float32]
    sequence_mask: NDArray[np.bool_]

    command_ids: NDArray[np.int64]
    command_times: NDArray[np.float32]
    command_mask: NDArray[np.bool_]
    negative_command_ids: NDArray[np.int64]
    negative_command_times: NDArray[np.float32]
    negative_command_mask: NDArray[np.bool_]

    target_channel_ids: NDArray[np.int64]
    target_times: NDArray[np.float32]
    target_values: NDArray[np.float32]
    target_is_future: NDArray[np.bool_]

    block_id: int
    negative_block_id: int | None
    block_start_ns: int
    block_end_ns: int
    context_timestamps_ns: NDArray[np.int64]
    target_timestamps_ns: NDArray[np.int64]


@dataclass(frozen=True)
class PretrainingBatch:
    telemetry: Tensor
    timestamps: Tensor
    observation_mask: Tensor
    time_deltas: Tensor
    sequence_mask: Tensor

    command_ids: Tensor
    command_times: Tensor
    command_mask: Tensor
    negative_command_ids: Tensor
    negative_command_times: Tensor
    negative_command_mask: Tensor

    target_batch_index: Tensor
    target_channel_ids: Tensor
    target_times: Tensor
    target_values: Tensor
    target_is_future: Tensor

    block_ids: tuple[int, ...]
    negative_block_ids: tuple[int | None, ...]
    block_starts_ns: Tensor
    block_ends_ns: Tensor
    context_timestamps_ns: Tensor
    target_timestamps_ns: Tensor


def target_folds_for_observations(
    channel_ids: int | NDArray[np.integer[Any]],
    timestamps_ns: NDArray[np.integer[Any]],
    *,
    target_folds: int,
) -> NDArray[np.int64]:
    """Map each observation to one stable target fold."""

    if target_folds < 1:
        raise ValueError("target_folds must be positive")
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    channels = np.asarray(channel_ids, dtype=np.int64)
    channels = np.broadcast_to(channels, timestamps.shape)
    values = timestamps.view(np.uint64).copy()
    values ^= (channels.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)) * np.uint64(
        0xBF58476D1CE4E5B9
    )
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values % np.uint64(target_folds)).astype(np.int64)


def _rng(seed: int, epoch: int, block_id: int, stream: int = 0) -> np.random.Generator:
    components = [
        np.uint32(seed & 0xFFFFFFFF),
        np.uint32(epoch & 0xFFFFFFFF),
        np.uint32(block_id & 0xFFFFFFFF),
        np.uint32(stream & 0xFFFFFFFF),
    ]
    return np.random.default_rng(np.random.SeedSequence(components))


def _select_context_timestamps(
    candidates: NDArray[np.int64],
    *,
    maximum: int,
    seed: int,
    epoch: int,
    block_id: int,
    block_start_ns: int,
    block_end_ns: int,
) -> NDArray[np.int64]:
    if maximum < 2:
        raise ValueError("max_steps must be at least two")
    ordered = np.unique(
        np.concatenate(
            (
                np.asarray(candidates, dtype=np.int64),
                np.asarray([block_start_ns, block_end_ns - 1], dtype=np.int64),
            )
        )
    )
    if len(ordered) <= maximum:
        return ordered
    interior_slots = maximum - 2
    selected: list[int] = [int(ordered[0]), int(ordered[-1])]
    if interior_slots:
        generator = _rng(seed, epoch, block_id, stream=1)
        edges = np.linspace(float(ordered[0]), float(ordered[-1]) + 1.0, interior_slots + 1)
        for left, right in pairwise(edges):
            first = int(np.searchsorted(ordered, int(left), side="left"))
            final = int(np.searchsorted(ordered, int(right), side="left"))
            first = max(first, 1)
            final = min(final, len(ordered) - 1)
            if final > first:
                selected.append(int(ordered[int(generator.integers(first, final))]))
        if len(set(selected)) < maximum:
            remaining = np.setdiff1d(ordered[1:-1], np.asarray(selected), assume_unique=False)
            count = min(maximum - len(set(selected)), len(remaining))
            if count:
                positions = generator.choice(len(remaining), size=count, replace=False)
                selected.extend(int(value) for value in remaining[positions])
    result = np.asarray(sorted(set(selected)), dtype=np.int64)
    if len(result) > maximum:
        result = np.concatenate((result[: maximum - 1], result[-1:]))
    return result


def _cap_sequence(
    timestamps: NDArray[np.int64], values: NDArray[np.integer[Any]], maximum: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    if maximum < 1:
        raise ValueError("max_commands must be positive")
    if len(timestamps) <= maximum:
        return timestamps.astype(np.int64, copy=False), values.astype(np.int64, copy=False)
    positions = np.unique(np.linspace(0, len(timestamps) - 1, maximum, dtype=np.int64))
    return timestamps[positions].astype(np.int64), values[positions].astype(np.int64)


class CommandNegativeSampler:
    """Select matched wrong-command blocks without temporal overlap."""

    def __init__(
        self,
        store: SSLStore,
        *,
        block_indices: Sequence[int] | None = None,
        era_days: float = 30.0,
        count_tolerance: float = 0.25,
        seed: int = 42,
    ) -> None:
        if not np.isfinite(era_days) or era_days <= 0:
            raise ValueError("era_days must be positive and finite")
        if not np.isfinite(count_tolerance) or count_tolerance < 0:
            raise ValueError("count_tolerance must be non-negative and finite")
        self.store = store
        self.block_indices = np.asarray(
            tuple(range(len(store.block_ids))) if block_indices is None else block_indices,
            dtype=np.int64,
        )
        if np.any((self.block_indices < 0) | (self.block_indices >= len(store.block_ids))):
            raise IndexError("negative-sampling block index is out of range")
        self.era_ns = int(era_days * 24 * _NS_PER_HOUR)
        self.count_tolerance = float(count_tolerance)
        self.seed = int(seed)
        self.command_counts = store.command_end_positions - store.command_start_positions
        anchor = int(store.block_starts_ns[0]) if len(store.block_starts_ns) else 0
        self._era_by_index = (store.block_starts_ns - anchor) // self.era_ns
        self._blocks_by_era: dict[int, NDArray[np.int64]] = {}
        for era in np.unique(self._era_by_index[self.block_indices]):
            in_era = self._era_by_index[self.block_indices] == era
            self._blocks_by_era[int(era)] = self.block_indices[in_era]

    def valid_candidates(self, positive_index: int) -> NDArray[np.int64]:
        if not 0 <= positive_index < len(self.store.block_ids):
            raise IndexError("positive block index is out of range")
        positive_count = int(self.command_counts[positive_index])
        if positive_count <= 0:
            return np.empty(0, dtype=np.int64)
        starts = self.store.block_starts_ns
        ends = self.store.block_ends_ns
        positive_era = int(self._era_by_index[positive_index])
        candidates = self._blocks_by_era.get(positive_era, np.empty(0, dtype=np.int64))
        counts = self.command_counts[candidates]
        different = candidates != positive_index
        non_overlapping = (ends[candidates] <= starts[positive_index]) | (
            starts[candidates] >= ends[positive_index]
        )
        denominator = np.maximum(np.maximum(counts, positive_count), 1)
        similar_count = np.abs(counts - positive_count) / denominator <= self.count_tolerance
        has_commands = counts > 0
        return candidates[different & non_overlapping & similar_count & has_commands]

    def sample(self, positive_index: int, *, epoch: int = 0) -> int | None:
        candidates = self.valid_candidates(positive_index)
        if len(candidates) == 0:
            return None
        generator = _rng(self.seed, epoch, int(self.store.block_ids[positive_index]), stream=2)
        return int(candidates[int(generator.integers(0, len(candidates)))])

    def pair_is_valid(self, positive_index: int, negative_index: int) -> bool:
        return bool(np.any(self.valid_candidates(positive_index) == negative_index))


def _apply_normalizer(
    normalizer: ChannelNormalizer | _Normalizer | Any | None,
    channel: str,
    values: NDArray[np.float32],
) -> NDArray[np.float32]:
    if normalizer is None:
        return values.astype(np.float32, copy=True)
    if hasattr(normalizer, "transform"):
        transformed = normalizer.transform(channel, values)
    elif hasattr(normalizer, "transform_channel"):
        transformed = normalizer.transform_channel(channel, values)
    else:
        raise TypeError("normalizer must expose transform(channel, values)")
    result = np.asarray(transformed, dtype=np.float32)
    if result.shape != values.shape:
        raise ValueError(f"normalizer changed the shape of channel {channel}")
    return result


class PretrainingDataset(Dataset[PretrainingSample]):
    """One item per temporal block and one exhaustive target fold per epoch."""

    def __init__(
        self,
        store: SSLStore | str | Path,
        *,
        normalizer: ChannelNormalizer | _Normalizer | Any | None = None,
        target_folds: int = 8,
        max_steps: int = 256,
        max_commands: int = 128,
        forecast_hours: float = 1.0,
        seed: int = 42,
        block_ids: Sequence[int] | None = None,
        command_negative_era_days: float = 30.0,
        command_count_tolerance: float = 0.25,
    ) -> None:
        self.store = store if isinstance(store, SSLStore) else SSLStore(store)
        if target_folds < 1:
            raise ValueError("target_folds must be positive")
        if max_steps < 2:
            raise ValueError("max_steps must be at least two")
        if max_commands < 1:
            raise ValueError("max_commands must be positive")
        if not np.isfinite(forecast_hours) or forecast_hours < 0:
            raise ValueError("forecast_hours must be non-negative and finite")
        id_to_index = {int(block_id): index for index, block_id in enumerate(self.store.block_ids)}
        selected_ids = (
            tuple(int(value) for value in self.store.block_ids)
            if block_ids is None
            else tuple(int(value) for value in block_ids)
        )
        unknown = sorted(set(selected_ids).difference(id_to_index))
        if unknown:
            raise ValueError(f"unknown SSL block IDs: {unknown}")
        self.block_indices = np.asarray([id_to_index[block_id] for block_id in selected_ids])
        if normalizer is None and (self.store.root / "scaler.json").is_file():
            from mirage.normalization import ChannelNormalizer

            normalizer = ChannelNormalizer.load(self.store.root / "scaler.json")
        if (
            normalizer is not None
            and hasattr(normalizer, "channel_order")
            and tuple(normalizer.channel_order) != self.store.channel_order
        ):
            raise ValueError("normalizer channel order does not match the SSL store")
        self.normalizer = normalizer
        self.target_folds = int(target_folds)
        self.max_steps = int(max_steps)
        self.max_commands = int(max_commands)
        self.forecast_hours = float(forecast_hours)
        self.seed = int(seed)
        self.epoch = 0
        self.negative_sampler = CommandNegativeSampler(
            self.store,
            block_indices=self.block_indices.tolist(),
            era_days=command_negative_era_days,
            count_tolerance=command_count_tolerance,
            seed=self.seed,
        )

    @property
    def active_target_fold(self) -> int:
        return self.epoch % self.target_folds

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.block_indices)

    def _target_mask(
        self, channel_id: int, timestamps_ns: NDArray[np.int64]
    ) -> NDArray[np.bool_]:
        folds = target_folds_for_observations(
            channel_id, timestamps_ns, target_folds=self.target_folds
        )
        return folds == self.active_target_fold

    def iter_epoch_target_keys(self, epoch: int | None = None) -> Any:
        """Yield every target key without constructing dense context paths."""

        active_fold = self.active_target_fold if epoch is None else int(epoch) % self.target_folds
        for block_index in self.block_indices:
            for channel_id in range(len(self.store.channel_order)):
                timestamps, _ = self.store.channel_block(int(block_index), channel_id)
                folds = target_folds_for_observations(
                    channel_id, timestamps, target_folds=self.target_folds
                )
                for timestamp in timestamps[folds == active_fold]:
                    yield channel_id, int(timestamp)

    def coverage_report(self, *, epochs: int | None = None) -> dict[str, int | float | bool]:
        coverage_epochs = self.target_folds if epochs is None else int(epochs)
        if coverage_epochs < 0:
            raise ValueError("epochs must be non-negative")
        fold_counts = np.zeros(self.target_folds, dtype=np.int64)
        for block_index in self.block_indices:
            for channel_id in range(len(self.store.channel_order)):
                timestamps, _ = self.store.channel_block(int(block_index), channel_id)
                folds = target_folds_for_observations(
                    channel_id, timestamps, target_folds=self.target_folds
                )
                fold_counts += np.bincount(folds, minlength=self.target_folds)
        total = int(fold_counts.sum())
        complete_cycles, remaining = divmod(coverage_epochs, self.target_folds)
        target_uses_by_fold = np.full(self.target_folds, complete_cycles, dtype=np.int64)
        target_uses_by_fold[:remaining] += 1
        targeted = int(fold_counts[target_uses_by_fold > 0].sum())
        repeated = int(fold_counts[target_uses_by_fold > 1].sum())
        target_assignments = int(np.dot(fold_counts, target_uses_by_fold))
        never = max(0, total - targeted)
        return {
            "raw_training_observations": total,
            "used_as_reconstruction_targets": targeted,
            "target_assignments": target_assignments,
            "used_more_than_once": repeated,
            "never_used": never,
            "coverage_fraction": 1.0 if total == 0 else targeted / total,
            "coverage_epochs": coverage_epochs,
            "partial_coverage": targeted != total or repeated != 0,
        }

    def __getitem__(self, item: int) -> PretrainingSample:
        block_index = int(self.block_indices[item])
        block_id = int(self.store.block_ids[block_index])
        block_start_ns = int(self.store.block_starts_ns[block_index])
        block_end_ns = int(self.store.block_ends_ns[block_index])
        context_parts: list[NDArray[np.int64]] = []
        target_channel_parts: list[NDArray[np.int64]] = []
        target_timestamp_parts: list[NDArray[np.int64]] = []
        target_value_parts: list[NDArray[np.float32]] = []
        non_target_by_channel: list[
            tuple[NDArray[np.int64], NDArray[np.float32]]
        ] = []
        for channel_id, channel in enumerate(self.store.channel_order):
            timestamps, raw_values = self.store.channel_block(block_index, channel_id)
            target = self._target_mask(channel_id, timestamps)
            target_timestamps = np.asarray(timestamps[target], dtype=np.int64)
            target_values = _apply_normalizer(
                self.normalizer, channel, np.asarray(raw_values[target], dtype=np.float32)
            )
            target_channel_parts.append(
                np.full(len(target_timestamps), channel_id, dtype=np.int64)
            )
            target_timestamp_parts.append(target_timestamps)
            target_value_parts.append(target_values)
            context_timestamps = np.asarray(timestamps[~target], dtype=np.int64)
            context_values = _apply_normalizer(
                self.normalizer, channel, np.asarray(raw_values[~target], dtype=np.float32)
            )
            non_target_by_channel.append((context_timestamps, context_values))
            if len(context_timestamps):
                context_parts.append(context_timestamps)

        all_context = (
            np.concatenate(context_parts) if context_parts else np.empty(0, dtype=np.int64)
        )
        grid = _select_context_timestamps(
            all_context,
            maximum=self.max_steps,
            seed=self.seed,
            epoch=self.epoch,
            block_id=block_id,
            block_start_ns=block_start_ns,
            block_end_ns=block_end_ns,
        )
        channels = len(self.store.channel_order)
        telemetry = np.zeros((len(grid), channels), dtype=np.float32)
        observation_mask = np.zeros((len(grid), channels), dtype=np.bool_)
        time_deltas = np.zeros((len(grid), channels), dtype=np.float32)
        for channel_id, (timestamps, values) in enumerate(non_target_by_channel):
            if len(timestamps) == 0:
                continue
            positions = np.searchsorted(timestamps, grid, side="right") - 1
            valid = positions >= 0
            clipped = positions.clip(min=0)
            telemetry[valid, channel_id] = values[clipped[valid]]
            exact = valid & (timestamps[clipped] == grid)
            observation_mask[:, channel_id] = exact
            time_deltas[valid, channel_id] = (
                (grid[valid] - timestamps[clipped[valid]]) / _NS_PER_HOUR
            ).astype(np.float32)

        target_channels = np.concatenate(target_channel_parts)
        target_timestamps = np.concatenate(target_timestamp_parts)
        target_values = np.concatenate(target_value_parts)
        if len(target_timestamps):
            order = np.lexsort((target_channels, target_timestamps))
            target_channels = target_channels[order]
            target_timestamps = target_timestamps[order]
            target_values = target_values[order]
        target_times = ((target_timestamps - block_start_ns) / _NS_PER_HOUR).astype(np.float32)
        future_cutoff_ns = block_end_ns - int(self.forecast_hours * _NS_PER_HOUR)
        target_is_future = target_timestamps >= future_cutoff_ns

        command_timestamps, command_ids = self.store.command_block(block_index)
        command_timestamps, command_ids = _cap_sequence(
            np.asarray(command_timestamps, dtype=np.int64), command_ids, self.max_commands
        )
        command_times = ((command_timestamps - block_start_ns) / _NS_PER_HOUR).astype(np.float32)
        negative_index = self.negative_sampler.sample(block_index, epoch=self.epoch)
        if negative_index is None:
            negative_timestamps = np.empty(0, dtype=np.int64)
            negative_ids = np.empty(0, dtype=np.int64)
            negative_times = np.empty(0, dtype=np.float32)
            negative_block_id = None
        else:
            negative_timestamps_raw, negative_ids_raw = self.store.command_block(negative_index)
            negative_timestamps, negative_ids = _cap_sequence(
                np.asarray(negative_timestamps_raw, dtype=np.int64),
                negative_ids_raw,
                self.max_commands,
            )
            negative_start_ns = int(self.store.block_starts_ns[negative_index])
            negative_times = (
                (negative_timestamps - negative_start_ns) / _NS_PER_HOUR
            ).astype(np.float32)
            negative_block_id = int(self.store.block_ids[negative_index])

        return PretrainingSample(
            telemetry=telemetry,
            timestamps=((grid - block_start_ns) / _NS_PER_HOUR).astype(np.float32),
            observation_mask=observation_mask,
            time_deltas=time_deltas,
            sequence_mask=np.ones(len(grid), dtype=np.bool_),
            command_ids=command_ids.astype(np.int64, copy=False),
            command_times=command_times,
            command_mask=np.ones(len(command_ids), dtype=np.bool_),
            negative_command_ids=negative_ids.astype(np.int64, copy=False),
            negative_command_times=negative_times,
            negative_command_mask=np.ones(len(negative_ids), dtype=np.bool_),
            target_channel_ids=target_channels.astype(np.int64, copy=False),
            target_times=target_times,
            target_values=target_values.astype(np.float32, copy=False),
            target_is_future=target_is_future.astype(np.bool_, copy=False),
            block_id=block_id,
            negative_block_id=negative_block_id,
            block_start_ns=block_start_ns,
            block_end_ns=block_end_ns,
            context_timestamps_ns=grid,
            target_timestamps_ns=target_timestamps,
        )


def collate_pretraining(samples: Sequence[PretrainingSample]) -> PretrainingBatch:
    if not samples:
        raise ValueError("cannot collate an empty pretraining batch")
    batch_size = len(samples)
    channels = samples[0].telemetry.shape[1]
    if any(sample.telemetry.shape[1] != channels for sample in samples):
        raise ValueError("all pretraining samples must use the same channel order")
    max_steps = max(len(sample.timestamps) for sample in samples)
    max_commands = max(1, max(len(sample.command_ids) for sample in samples))
    max_negative_commands = max(1, max(len(sample.negative_command_ids) for sample in samples))
    telemetry = np.zeros((batch_size, max_steps, channels), dtype=np.float32)
    timestamps = np.zeros((batch_size, max_steps), dtype=np.float32)
    observation_mask = np.zeros((batch_size, max_steps, channels), dtype=np.bool_)
    time_deltas = np.zeros((batch_size, max_steps, channels), dtype=np.float32)
    sequence_mask = np.zeros((batch_size, max_steps), dtype=np.bool_)
    context_timestamps = np.zeros((batch_size, max_steps), dtype=np.int64)
    command_ids = np.zeros((batch_size, max_commands), dtype=np.int64)
    command_times = np.zeros((batch_size, max_commands), dtype=np.float32)
    command_mask = np.zeros((batch_size, max_commands), dtype=np.bool_)
    negative_command_ids = np.zeros((batch_size, max_negative_commands), dtype=np.int64)
    negative_command_times = np.zeros((batch_size, max_negative_commands), dtype=np.float32)
    negative_command_mask = np.zeros((batch_size, max_negative_commands), dtype=np.bool_)
    target_batch_parts: list[NDArray[np.int64]] = []
    target_channel_parts: list[NDArray[np.int64]] = []
    target_time_parts: list[NDArray[np.float32]] = []
    target_value_parts: list[NDArray[np.float32]] = []
    target_future_parts: list[NDArray[np.bool_]] = []
    target_timestamp_parts: list[NDArray[np.int64]] = []
    for row, sample in enumerate(samples):
        steps = len(sample.timestamps)
        commands = len(sample.command_ids)
        negative_commands = len(sample.negative_command_ids)
        targets = len(sample.target_channel_ids)
        telemetry[row, :steps] = sample.telemetry
        timestamps[row, :steps] = sample.timestamps
        observation_mask[row, :steps] = sample.observation_mask
        time_deltas[row, :steps] = sample.time_deltas
        sequence_mask[row, :steps] = sample.sequence_mask
        context_timestamps[row, :steps] = sample.context_timestamps_ns
        if commands:
            command_ids[row, :commands] = sample.command_ids
            command_times[row, :commands] = sample.command_times
            command_mask[row, :commands] = sample.command_mask
        if negative_commands:
            negative_command_ids[row, :negative_commands] = sample.negative_command_ids
            negative_command_times[row, :negative_commands] = sample.negative_command_times
            negative_command_mask[row, :negative_commands] = sample.negative_command_mask
        target_batch_parts.append(np.full(targets, row, dtype=np.int64))
        target_channel_parts.append(sample.target_channel_ids)
        target_time_parts.append(sample.target_times)
        target_value_parts.append(sample.target_values)
        target_future_parts.append(sample.target_is_future)
        target_timestamp_parts.append(sample.target_timestamps_ns)

    return PretrainingBatch(
        telemetry=torch.from_numpy(telemetry),
        timestamps=torch.from_numpy(timestamps),
        observation_mask=torch.from_numpy(observation_mask),
        time_deltas=torch.from_numpy(time_deltas),
        sequence_mask=torch.from_numpy(sequence_mask),
        command_ids=torch.from_numpy(command_ids),
        command_times=torch.from_numpy(command_times),
        command_mask=torch.from_numpy(command_mask),
        negative_command_ids=torch.from_numpy(negative_command_ids),
        negative_command_times=torch.from_numpy(negative_command_times),
        negative_command_mask=torch.from_numpy(negative_command_mask),
        target_batch_index=torch.from_numpy(np.concatenate(target_batch_parts)),
        target_channel_ids=torch.from_numpy(np.concatenate(target_channel_parts)),
        target_times=torch.from_numpy(np.concatenate(target_time_parts)),
        target_values=torch.from_numpy(np.concatenate(target_value_parts)),
        target_is_future=torch.from_numpy(np.concatenate(target_future_parts)),
        block_ids=tuple(sample.block_id for sample in samples),
        negative_block_ids=tuple(sample.negative_block_id for sample in samples),
        block_starts_ns=torch.tensor([sample.block_start_ns for sample in samples], dtype=torch.int64),
        block_ends_ns=torch.tensor([sample.block_end_ns for sample in samples], dtype=torch.int64),
        context_timestamps_ns=torch.from_numpy(context_timestamps),
        target_timestamps_ns=torch.from_numpy(np.concatenate(target_timestamp_parts)),
    )


collate_pretraining_samples = collate_pretraining
