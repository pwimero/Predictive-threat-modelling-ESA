"""Continuous-time telemetry encoders for MIRAGE-M1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor, nn


class EncoderVariant(StrEnum):
    """MIRAGE-M1 evidence variants."""

    TELEMETRY_ONLY = "telemetry_only"
    JOINT = "joint"
    AFFECTED_MASK = "affected_mask"


@dataclass(frozen=True)
class EncoderRepresentations:
    """Representations produced from the same event episode."""

    telemetry_only: Tensor
    joint: Tensor
    affected_mask: Tensor

    def select(self, variant: EncoderVariant | str) -> Tensor:
        """Return the representation for one named experimental variant."""

        selected = EncoderVariant(variant)
        if selected is EncoderVariant.TELEMETRY_ONLY:
            return self.telemetry_only
        if selected is EncoderVariant.JOINT:
            return self.joint
        return self.affected_mask


class NeuralJumpCDEEncoder(nn.Module):
    """Encode irregular telemetry with command-conditioned latent jumps."""

    def __init__(
        self,
        telemetry_channels: int,
        num_commands: int,
        *,
        hidden_dim: int = 32,
        representation_dim: int = 16,
        command_embedding_dim: int = 8,
        affected_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        if telemetry_channels < 1:
            raise ValueError("telemetry_channels must be positive")
        if num_commands < 1:
            raise ValueError("num_commands must be positive")
        if min(hidden_dim, representation_dim, command_embedding_dim, affected_embedding_dim) < 1:
            raise ValueError("all model dimensions must be positive")

        self.telemetry_channels = telemetry_channels
        self.num_commands = num_commands
        self.hidden_dim = hidden_dim
        self.representation_dim = representation_dim
        self.path_dim = 1 + (3 * telemetry_channels)
        self.control_rank = min(8, hidden_dim)

        self.initial_state = nn.Sequential(
            nn.Linear(self.path_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),
        )
        self.vector_field = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * self.control_rank),
            nn.Tanh(),
        )
        # A low-rank projection keeps the CDE compact.
        self.control_projection = nn.Linear(self.path_dim, self.control_rank, bias=False)
        self.euler_norm = nn.LayerNorm(hidden_dim)

        self.command_embedding = nn.Embedding(num_commands, command_embedding_dim)
        # Timing features describe relative time and command position.
        self.command_jump = nn.GRUCell(command_embedding_dim + 3, hidden_dim)

        self.representation_projection = nn.Sequential(
            nn.Linear(hidden_dim, representation_dim),
            nn.GELU(),
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
        )
        self.affected_encoder = nn.Sequential(
            nn.Linear(telemetry_channels, affected_embedding_dim),
            nn.GELU(),
            nn.LayerNorm(affected_embedding_dim),
        )
        self.affected_fusion = nn.Sequential(
            nn.Linear(representation_dim + affected_embedding_dim, representation_dim),
            nn.GELU(),
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
        )

    @staticmethod
    def _is_floating(tensor: Tensor) -> bool:
        return tensor.dtype.is_floating_point

    @staticmethod
    def _require(condition: Tensor | bool, message: str) -> None:
        if isinstance(condition, Tensor):
            condition = bool(condition.detach().all().item())
        if not condition:
            raise ValueError(message)

    def _validate_inputs(
        self,
        telemetry: Tensor,
        timestamps: Tensor,
        observation_mask: Tensor,
        time_deltas: Tensor,
        command_ids: Tensor,
        command_times: Tensor,
        command_mask: Tensor,
        affected_channel_mask: Tensor,
        sequence_mask: Tensor | None,
    ) -> Tensor:
        self._require(telemetry.ndim == 3, "telemetry must have shape [batch, steps, channels]")
        batch, steps, channels = telemetry.shape
        self._require(batch > 0 and steps > 0, "telemetry batches and sequences cannot be empty")
        self._require(
            channels == self.telemetry_channels,
            f"expected {self.telemetry_channels} telemetry channels, received {channels}",
        )
        self._require(self._is_floating(telemetry), "telemetry must be floating point")
        self._require(
            timestamps.shape == (batch, steps) and self._is_floating(timestamps),
            "timestamps must be floating point with shape [batch, steps]",
        )
        self._require(
            observation_mask.shape == telemetry.shape,
            "observation_mask must match telemetry shape",
        )
        self._require(
            time_deltas.shape == telemetry.shape and self._is_floating(time_deltas),
            "time_deltas must be floating point and match telemetry shape",
        )
        self._require(command_ids.ndim == 2, "command_ids must have shape [batch, commands]")
        self._require(
            command_ids.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8},
            "command_ids must use an integer dtype",
        )
        self._require(
            command_ids.shape[0] == batch,
            "command_ids batch dimension must match telemetry",
        )
        self._require(
            command_times.shape == command_ids.shape and self._is_floating(command_times),
            "command_times must be floating point and match command_ids shape",
        )
        self._require(
            command_mask.shape == command_ids.shape,
            "command_mask must match command_ids shape",
        )
        self._require(
            affected_channel_mask.shape == (batch, channels),
            "affected_channel_mask must have shape [batch, channels]",
        )

        tensors = (
            timestamps,
            observation_mask,
            time_deltas,
            command_ids,
            command_times,
            command_mask,
            affected_channel_mask,
        )
        self._require(
            all(tensor.device == telemetry.device for tensor in tensors),
            "all inputs must be on the same device",
        )

        valid_steps = (
            torch.ones((batch, steps), dtype=torch.bool, device=telemetry.device)
            if sequence_mask is None
            else sequence_mask.to(dtype=torch.bool)
        )
        if sequence_mask is not None:
            self._require(
                sequence_mask.shape == (batch, steps),
                "sequence_mask must have shape [batch, steps]",
            )
            self._require(
                sequence_mask.device == telemetry.device,
                "all inputs must be on the same device",
            )
        self._require(valid_steps[:, 0], "the first telemetry step must be valid")
        self._require(
            ~((~valid_steps[:, :-1]) & valid_steps[:, 1:]),
            "sequence_mask must contain one contiguous valid prefix",
        )

        observed = observation_mask.to(dtype=torch.bool) & valid_steps.unsqueeze(-1)
        self._require(
            torch.isfinite(telemetry[observed]),
            "observed telemetry values must be finite",
        )
        self._require(
            torch.isfinite(timestamps[valid_steps]),
            "valid timestamps must be finite",
        )
        if steps > 1:
            adjacent = valid_steps[:, 1:] & valid_steps[:, :-1]
            differences = timestamps[:, 1:] - timestamps[:, :-1]
            self._require(
                (~adjacent) | (differences >= 0),
                "valid timestamps must be non-decreasing",
            )
        valid_delta = valid_steps.unsqueeze(-1).expand_as(time_deltas)
        self._require(
            torch.isfinite(time_deltas[valid_delta]) & (time_deltas[valid_delta] >= 0),
            "valid time_deltas must be finite and non-negative",
        )

        affected = affected_channel_mask.to(dtype=telemetry.dtype)
        self._require(
            torch.isfinite(affected) & (affected >= 0) & (affected <= 1),
            "affected_channel_mask values must lie in [0, 1]",
        )

        valid_commands = command_mask.to(dtype=torch.bool)
        self._require(
            torch.isfinite(command_times[valid_commands]),
            "valid command times must be finite",
        )
        if valid_commands.any():
            selected_ids = command_ids[valid_commands]
            self._require(
                (selected_ids >= 0) & (selected_ids < self.num_commands),
                f"valid command IDs must be in [0, {self.num_commands - 1}]",
            )
        if command_ids.shape[1] > 1:
            self._require(
                ~((~valid_commands[:, :-1]) & valid_commands[:, 1:]),
                "command_mask must contain one contiguous valid prefix",
            )
            adjacent_commands = valid_commands[:, 1:] & valid_commands[:, :-1]
            command_differences = command_times[:, 1:] - command_times[:, :-1]
            self._require(
                (~adjacent_commands) | (command_differences >= 0),
                "valid command times must be non-decreasing",
            )
        return valid_steps

    def _control_path(
        self,
        telemetry: Tensor,
        timestamps: Tensor,
        observation_mask: Tensor,
        time_deltas: Tensor,
        sequence_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        lengths = sequence_mask.sum(dim=1)
        final_indices = lengths - 1
        start = timestamps[:, 0]
        end = timestamps.gather(1, final_indices.unsqueeze(1)).squeeze(1)
        raw_span = (end - start).clamp_min(0)
        span = raw_span.clamp_min(torch.finfo(timestamps.dtype).eps)

        relative_time = (timestamps - start.unsqueeze(1)) / span.unsqueeze(1)
        # Preserve episode duration while compressing long windows.
        time_coordinate = relative_time * torch.log1p(raw_span).unsqueeze(1)
        time_coordinate = torch.where(
            sequence_mask, time_coordinate, torch.zeros_like(time_coordinate)
        )
        observed = observation_mask.to(dtype=torch.bool) & sequence_mask.unsqueeze(-1)
        clean_telemetry = torch.where(observed, telemetry, torch.zeros_like(telemetry))

        # Log time deltas keep long gaps numerically stable.
        scaled_delta = torch.log1p(time_deltas.clamp_min(0)).clamp(max=20)
        scaled_delta = torch.where(
            sequence_mask.unsqueeze(-1), scaled_delta, torch.zeros_like(scaled_delta)
        )
        path = torch.cat(
            (
                time_coordinate.unsqueeze(-1),
                clean_telemetry,
                observed.to(dtype=telemetry.dtype),
                scaled_delta,
            ),
            dim=-1,
        )
        return path, start

    def _euler_step(self, latent: Tensor, control_increment: Tensor, active: Tensor) -> Tensor:
        vector_field = self.vector_field(latent).view(
            latent.shape[0], self.hidden_dim, self.control_rank
        )
        projected_control = self.control_projection(control_increment)
        update = torch.einsum("bhr,br->bh", vector_field, projected_control)
        update = update / math.sqrt(self.control_rank)
        candidate = self.euler_norm(latent + update)
        return torch.where(active.unsqueeze(-1), candidate, latent)

    def _command_features(
        self,
        command_ids: Tensor,
        command_times: Tensor,
        command_mask: Tensor,
        start: Tensor,
        dtype: torch.dtype,
    ) -> Tensor:
        valid = command_mask.to(dtype=torch.bool)
        safe_ids = command_ids.clamp(min=0, max=self.num_commands - 1).to(dtype=torch.long)
        embedded = self.command_embedding(safe_ids)
        command_offset = command_times - start.unsqueeze(1)
        relative_time = (torch.sign(command_offset) * torch.log1p(command_offset.abs())).clamp(
            -20, 20
        )

        previous_time = start
        previous_seen = torch.zeros_like(start, dtype=torch.bool)
        gaps: list[Tensor] = []
        counts: list[Tensor] = []
        cumulative = torch.zeros_like(start)
        total = valid.sum(dim=1).clamp_min(1).to(dtype=dtype)
        for index in range(command_ids.shape[1]):
            current_valid = valid[:, index]
            reference = torch.where(previous_seen, previous_time, start)
            gap = torch.log1p((command_times[:, index] - reference).clamp_min(0)).clamp(max=20)
            gaps.append(torch.where(current_valid, gap, torch.zeros_like(gap)))
            cumulative = cumulative + current_valid.to(dtype=dtype)
            counts.append(cumulative / total)
            previous_time = torch.where(current_valid, command_times[:, index], previous_time)
            previous_seen = previous_seen | current_valid

        if command_ids.shape[1] == 0:
            timing = torch.empty(
                (command_ids.shape[0], 0, 3), dtype=dtype, device=command_ids.device
            )
        else:
            gap_tensor = torch.stack(gaps, dim=1)
            count_tensor = torch.stack(counts, dim=1)
            timing = torch.stack((relative_time, gap_tensor, count_tensor), dim=-1).to(dtype=dtype)
            timing = torch.where(valid.unsqueeze(-1), timing, torch.zeros_like(timing))
        return torch.cat((embedded.to(dtype=dtype), timing), dim=-1)

    def _apply_command_jumps(
        self,
        latent: Tensor,
        features: Tensor,
        due: Tensor,
    ) -> Tensor:
        # Apply only commands due in this telemetry interval.
        remaining = due.clone()
        batch_indices = torch.arange(features.shape[0], device=features.device)
        while bool(remaining.any().detach().item()):
            has_command = remaining.any(dim=1)
            next_index = remaining.to(dtype=torch.int64).argmax(dim=1)
            command_input = features[batch_indices, next_index]
            candidate = self.command_jump(command_input, latent)
            latent = torch.where(has_command.unsqueeze(-1), candidate, latent)
            selected_batches = batch_indices[has_command]
            remaining[selected_batches, next_index[has_command]] = False
        return latent

    def forward(
        self,
        telemetry: Tensor,
        timestamps: Tensor,
        observation_mask: Tensor,
        time_deltas: Tensor,
        command_ids: Tensor,
        command_times: Tensor,
        command_mask: Tensor,
        affected_channel_mask: Tensor,
        sequence_mask: Tensor | None = None,
    ) -> EncoderRepresentations:
        """Encode a batch and return all three experimental representations."""

        valid_steps = self._validate_inputs(
            telemetry,
            timestamps,
            observation_mask,
            time_deltas,
            command_ids,
            command_times,
            command_mask,
            affected_channel_mask,
            sequence_mask,
        )
        path, start = self._control_path(
            telemetry, timestamps, observation_mask, time_deltas, valid_steps
        )
        telemetry_latent = self.initial_state(path[:, 0])
        joint_latent = telemetry_latent.clone()

        valid_commands = command_mask.to(dtype=torch.bool)
        command_features = self._command_features(
            command_ids,
            command_times,
            valid_commands,
            start,
            telemetry.dtype,
        )
        initial_commands = valid_commands & (command_times <= start.unsqueeze(1))
        joint_latent = self._apply_command_jumps(joint_latent, command_features, initial_commands)

        for step in range(1, telemetry.shape[1]):
            active = valid_steps[:, step]
            increment = path[:, step] - path[:, step - 1]
            telemetry_latent = self._euler_step(telemetry_latent, increment, active)
            joint_latent = self._euler_step(joint_latent, increment, active)
            commands_due = (
                valid_commands
                & active.unsqueeze(1)
                & (command_times > timestamps[:, step - 1].unsqueeze(1))
                & (command_times <= timestamps[:, step].unsqueeze(1))
            )
            joint_latent = self._apply_command_jumps(joint_latent, command_features, commands_due)

        telemetry_representation = self.representation_projection(telemetry_latent)
        joint_representation = self.representation_projection(joint_latent)
        affected_embedding = self.affected_encoder(affected_channel_mask.to(dtype=telemetry.dtype))
        affected_representation = self.affected_fusion(
            torch.cat((joint_representation, affected_embedding), dim=-1)
        )
        return EncoderRepresentations(
            telemetry_only=telemetry_representation,
            joint=joint_representation,
            affected_mask=affected_representation,
        )

    def encode_variant(
        self,
        variant: EncoderVariant | str,
        telemetry: Tensor,
        timestamps: Tensor,
        observation_mask: Tensor,
        time_deltas: Tensor,
        command_ids: Tensor,
        command_times: Tensor,
        command_mask: Tensor,
        affected_channel_mask: Tensor,
        sequence_mask: Tensor | None = None,
    ) -> Tensor:
        """Convenience wrapper returning one paper-ablation representation."""

        return self(
            telemetry,
            timestamps,
            observation_mask,
            time_deltas,
            command_ids,
            command_times,
            command_mask,
            affected_channel_mask,
            sequence_mask,
        ).select(variant)

    def encode_pretraining(
        self,
        telemetry: Tensor,
        timestamps: Tensor,
        observation_mask: Tensor,
        time_deltas: Tensor,
        command_ids: Tensor,
        command_times: Tensor,
        command_mask: Tensor,
        sequence_mask: Tensor,
    ) -> EncoderRepresentations:
        """Encode an SSL block without event-only oracle information."""

        if telemetry.ndim != 3:
            raise ValueError("telemetry must have shape [batch, steps, channels]")
        affected_channel_mask = torch.zeros(
            (telemetry.shape[0], telemetry.shape[2]),
            dtype=telemetry.dtype,
            device=telemetry.device,
        )
        return self(
            telemetry,
            timestamps,
            observation_mask,
            time_deltas,
            command_ids,
            command_times,
            command_mask,
            affected_channel_mask,
            sequence_mask,
        )
