"""PyTorch dataset and padding collator for MIRAGE event episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from mirage.preprocessing import EventEpisode


@dataclass(frozen=True)
class EpisodeBatch:
    telemetry: Tensor
    timestamps: Tensor
    observation_mask: Tensor
    time_deltas: Tensor
    sequence_mask: Tensor
    command_ids: Tensor
    command_times: Tensor
    command_mask: Tensor
    affected_channel_mask: Tensor
    labels: Tensor
    event_ids: tuple[str, ...]
    event_classes: tuple[str, ...]
    event_types: tuple[str, ...]
    protocols: tuple[str, ...]


class EpisodeDataset(Dataset[EventEpisode]):
    def __init__(
        self,
        episodes: tuple[EventEpisode, ...] | list[EventEpisode],
        *,
        class_names: tuple[str, ...] | None = None,
    ) -> None:
        self.episodes = tuple(episodes)
        self.class_names = class_names or tuple(
            sorted({episode.event_class for episode in episodes})
        )
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        unknown = sorted(
            {episode.event_class for episode in episodes}.difference(self.class_to_index)
        )
        if unknown:
            raise ValueError(f"episodes contain unknown classes: {unknown}")

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> EventEpisode:
        return self.episodes[index]


def collate_episodes(
    episodes: list[EventEpisode] | tuple[EventEpisode, ...],
    *,
    class_names: tuple[str, ...] | None = None,
) -> EpisodeBatch:
    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    selected = tuple(episodes)
    classes = class_names or tuple(sorted({episode.event_class for episode in selected}))
    class_to_index = {name: index for index, name in enumerate(classes)}
    batch = len(selected)
    max_steps = max(len(episode.timestamps) for episode in selected)
    max_commands = max(1, max(len(episode.command_ids) for episode in selected))
    channels = selected[0].telemetry.shape[1]
    telemetry = np.zeros((batch, max_steps, channels), dtype=np.float32)
    timestamps = np.zeros((batch, max_steps), dtype=np.float32)
    observation_mask = np.zeros((batch, max_steps, channels), dtype=np.bool_)
    time_deltas = np.zeros((batch, max_steps, channels), dtype=np.float32)
    sequence_mask = np.zeros((batch, max_steps), dtype=np.bool_)
    command_ids = np.zeros((batch, max_commands), dtype=np.int64)
    command_times = np.zeros((batch, max_commands), dtype=np.float32)
    command_mask = np.zeros((batch, max_commands), dtype=np.bool_)
    affected = np.zeros((batch, channels), dtype=np.float32)
    labels = np.zeros(batch, dtype=np.int64)
    for row, episode in enumerate(selected):
        if episode.telemetry.shape[1] != channels:
            raise ValueError("all episodes must use the same channel order")
        steps = len(episode.timestamps)
        commands = len(episode.command_ids)
        telemetry[row, :steps] = episode.telemetry
        timestamps[row, :steps] = episode.timestamps
        observation_mask[row, :steps] = episode.observation_mask
        time_deltas[row, :steps] = episode.time_deltas
        sequence_mask[row, :steps] = True
        if commands:
            command_ids[row, :commands] = episode.command_ids
            command_times[row, :commands] = episode.command_times
            command_mask[row, :commands] = True
        affected[row] = episode.affected_channel_mask
        labels[row] = class_to_index[episode.event_class]
    return EpisodeBatch(
        telemetry=torch.from_numpy(telemetry),
        timestamps=torch.from_numpy(timestamps),
        observation_mask=torch.from_numpy(observation_mask),
        time_deltas=torch.from_numpy(time_deltas),
        sequence_mask=torch.from_numpy(sequence_mask),
        command_ids=torch.from_numpy(command_ids),
        command_times=torch.from_numpy(command_times),
        command_mask=torch.from_numpy(command_mask),
        affected_channel_mask=torch.from_numpy(affected),
        labels=torch.from_numpy(labels),
        event_ids=tuple(episode.event_id for episode in selected),
        event_classes=tuple(episode.event_class for episode in selected),
        event_types=tuple(episode.event_type for episode in selected),
        protocols=tuple(episode.protocol for episode in selected),
    )
