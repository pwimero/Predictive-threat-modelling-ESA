"""Self-supervised pretraining for the MIRAGE encoder."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from mirage.encoders import EncoderRepresentations, NeuralJumpCDEEncoder
from mirage.protocol import canonical_sha256

_NS_PER_HOUR = 3_600_000_000_000


@dataclass(frozen=True)
class PretrainingObjectiveWeights:
    """Weights for the three declared MIRAGE self-supervised objectives."""

    reconstruction: float = 1.0
    future: float = 0.5
    command: float = 0.2

    def __post_init__(self) -> None:
        values = (self.reconstruction, self.future, self.command)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("pretraining objective weights must be finite and non-negative")
        if not any(value > 0 for value in values):
            raise ValueError("at least one pretraining objective must have positive weight")


@dataclass(frozen=True)
class CommandMatchResult:
    """Binary command-response objective and the logits used for AUROC."""

    loss: Tensor
    logits: Tensor
    labels: Tensor
    pair_count: int


@dataclass(frozen=True)
class PretrainingHistoryRecord:
    """One epoch of train and internal SSL-validation measurements."""

    epoch: int
    active_target_fold: int
    training_reconstruction_huber: float
    training_future_huber: float
    training_command_bce: float
    training_total_loss: float
    validation_reconstruction_huber: float
    validation_future_huber: float
    validation_command_bce: float
    validation_command_auroc: float | None
    validation_score: float


@dataclass(frozen=True)
class PretrainingResult:
    """Selected encoder and auditable metadata from one SSL run."""

    encoder: NeuralJumpCDEEncoder
    history: tuple[PretrainingHistoryRecord, ...]
    best_epoch: int
    best_validation_score: float
    completed_epochs: int
    target_folds: int
    seed: int
    forecast_hours: float
    channel_embedding_dim: int
    time_embedding_dim: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    partial_coverage: bool
    objective_weights: PretrainingObjectiveWeights
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class PretrainingArtifacts:
    """Paths produced by :func:`save_pretraining_artifacts`."""

    checkpoint: Path
    manifest: Path
    history: Path
    coverage_report: Path
    scaler: Path | None


@dataclass(frozen=True)
class _TargetTable:
    batch_index: Tensor
    channel_ids: Tensor
    times: Tensor
    values: Tensor

    @property
    def count(self) -> int:
        return int(self.values.numel())

    def select(self, mask: Tensor) -> _TargetTable:
        return _TargetTable(
            batch_index=self.batch_index[mask],
            channel_ids=self.channel_ids[mask],
            times=self.times[mask],
            values=self.values[mask],
        )


@dataclass(frozen=True)
class _BatchLosses:
    reconstruction: Tensor
    future: Tensor
    command: Tensor
    total: Tensor
    reconstruction_count: int
    future_count: int
    command_pair_count: int
    command_logits: Tensor
    command_labels: Tensor


@dataclass(frozen=True)
class _EpochMetrics:
    reconstruction: float
    future: float
    command: float
    total: float
    command_auroc: float | None


class ObservationDecoder(nn.Module):
    """Predict one real channel observation from a block representation."""

    def __init__(
        self,
        representation_dim: int,
        num_channels: int,
        channel_embedding_dim: int = 16,
        time_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if min(
            representation_dim,
            num_channels,
            channel_embedding_dim,
            time_embedding_dim,
        ) < 1:
            raise ValueError("decoder dimensions and num_channels must be positive")
        self.representation_dim = representation_dim
        self.num_channels = num_channels
        self.time_embedding_dim = time_embedding_dim
        self.channel_embedding = nn.Embedding(num_channels, channel_embedding_dim)
        frequency_count = (time_embedding_dim + 1) // 2
        frequencies = torch.pi * torch.pow(2.0, torch.arange(frequency_count).float())
        self.time_frequencies: Tensor
        self.register_buffer("time_frequencies", frequencies, persistent=False)
        input_dim = representation_dim + channel_embedding_dim + time_embedding_dim
        hidden_dim = max(32, representation_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _time_features(self, relative_times: Tensor) -> Tensor:
        angles = relative_times.unsqueeze(1) * self.time_frequencies.to(
            dtype=relative_times.dtype
        ).unsqueeze(0)
        features = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return features[:, : self.time_embedding_dim]

    def forward(
        self,
        representations: Tensor,
        target_batch_index: Tensor,
        target_channel_ids: Tensor,
        target_times: Tensor,
    ) -> Tensor:
        """Return one predicted normalised value for each real target."""

        if representations.ndim != 2 or representations.shape[1] != self.representation_dim:
            raise ValueError(
                f"representations must have shape [batch, {self.representation_dim}]"
            )
        target_batch_index = target_batch_index.reshape(-1).to(dtype=torch.long)
        target_channel_ids = target_channel_ids.reshape(-1).to(dtype=torch.long)
        target_times = target_times.reshape(-1).to(dtype=representations.dtype)
        count = target_times.numel()
        if target_batch_index.numel() != count or target_channel_ids.numel() != count:
            raise ValueError("all observation-target arrays must have equal length")
        if count == 0:
            return representations.new_empty((0,))
        if not bool(torch.isfinite(target_times).all().item()):
            raise ValueError("target times must be finite")
        if not bool(
            ((target_batch_index >= 0) & (target_batch_index < representations.shape[0]))
            .all()
            .item()
        ):
            raise ValueError("target batch indices are outside the representation batch")
        if not bool(
            ((target_channel_ids >= 0) & (target_channel_ids < self.num_channels)).all().item()
        ):
            raise ValueError(f"target channel IDs must lie in [0, {self.num_channels - 1}]")
        inputs = torch.cat(
            (
                representations[target_batch_index],
                self.channel_embedding(target_channel_ids).to(dtype=representations.dtype),
                self._time_features(target_times),
            ),
            dim=1,
        )
        return self.network(inputs).squeeze(1)


class CommandResponseDiscriminator(nn.Module):
    """Distinguish true command-conditioned latent changes from matched negatives."""

    def __init__(self, representation_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if representation_dim < 1:
            raise ValueError("representation_dim must be positive")
        width = hidden_dim or max(16, representation_dim)
        if width < 1:
            raise ValueError("hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(representation_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, representation_delta: Tensor) -> Tensor:
        if representation_delta.ndim != 2:
            raise ValueError("representation_delta must have shape [batch, representation]")
        return self.network(representation_delta).squeeze(1)


def observation_huber_loss(
    decoder: ObservationDecoder,
    representations: Tensor,
    target_batch_index: Tensor,
    target_channel_ids: Tensor,
    target_times: Tensor,
    target_values: Tensor,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Huber loss on actual raw observations only (never dense filled cells)."""

    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be finite and positive")
    values = target_values.reshape(-1).to(
        device=representations.device, dtype=representations.dtype
    )
    if values.numel() == 0:
        return representations.sum() * 0.0
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("target values must be finite")
    predictions = decoder(
        representations,
        target_batch_index.to(device=representations.device),
        target_channel_ids.to(device=representations.device),
        target_times.to(device=representations.device),
    )
    if predictions.shape != values.shape:
        raise ValueError("predictions and target values must have equal length")
    return F.huber_loss(predictions, values, reduction="mean", delta=delta)


def reconstruction_huber_loss(
    decoder: ObservationDecoder,
    representations: Tensor,
    target_batch_index: Tensor,
    target_channel_ids: Tensor,
    target_times: Tensor,
    target_values: Tensor,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Named wrapper for the masked raw-observation reconstruction objective."""

    return observation_huber_loss(
        decoder,
        representations,
        target_batch_index,
        target_channel_ids,
        target_times,
        target_values,
        delta=delta,
    )


def future_huber_loss(
    decoder: ObservationDecoder,
    representations: Tensor,
    target_batch_index: Tensor,
    target_channel_ids: Tensor,
    target_times: Tensor,
    target_values: Tensor,
    *,
    delta: float = 1.0,
) -> Tensor:
    """Named wrapper for future actual-observation forecasting."""

    return observation_huber_loss(
        decoder,
        representations,
        target_batch_index,
        target_channel_ids,
        target_times,
        target_values,
        delta=delta,
    )


def command_response_matching_loss(
    discriminator: CommandResponseDiscriminator,
    telemetry_representation: Tensor,
    true_joint_representation: Tensor,
    negative_joint_representation: Tensor,
    *,
    example_mask: Tensor | None = None,
) -> CommandMatchResult:
    """Binary loss for real versus era/count-matched wrong command sequences."""

    if not (
        telemetry_representation.shape
        == true_joint_representation.shape
        == negative_joint_representation.shape
    ):
        raise ValueError("telemetry, true-joint, and negative-joint shapes must match")
    if telemetry_representation.ndim != 2:
        raise ValueError("command representations must have shape [batch, representation]")
    if example_mask is None:
        selected = torch.ones(
            telemetry_representation.shape[0],
            dtype=torch.bool,
            device=telemetry_representation.device,
        )
    else:
        if example_mask.shape != (telemetry_representation.shape[0],):
            raise ValueError("example_mask must have shape [batch]")
        selected = example_mask.to(device=telemetry_representation.device, dtype=torch.bool)
    pair_count = int(selected.sum().item())
    if pair_count == 0:
        empty = telemetry_representation.new_empty((0,))
        zero = telemetry_representation.sum() * 0.0
        return CommandMatchResult(zero, empty, empty, 0)
    true_delta = true_joint_representation[selected] - telemetry_representation[selected]
    negative_delta = negative_joint_representation[selected] - telemetry_representation[selected]
    true_logits = discriminator(true_delta)
    negative_logits = discriminator(negative_delta)
    logits = torch.cat((true_logits, negative_logits))
    labels = torch.cat((torch.ones_like(true_logits), torch.zeros_like(negative_logits)))
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return CommandMatchResult(loss, logits, labels, pair_count)


def _get(batch: Any, name: str) -> Any:
    if isinstance(batch, Mapping):
        return batch[name]
    return getattr(batch, name)


def _optional(batch: Any, name: str, default: Any = None) -> Any:
    try:
        return _get(batch, name)
    except (AttributeError, KeyError):
        return default


def _tensor(batch: Any, name: str, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.as_tensor(_get(batch, name), dtype=dtype, device=device)


def _encoder_inputs(batch: Any, device: torch.device) -> tuple[Tensor, ...]:
    return (
        _tensor(batch, "telemetry", dtype=torch.float32, device=device),
        _tensor(batch, "timestamps", dtype=torch.float32, device=device),
        _tensor(batch, "observation_mask", dtype=torch.bool, device=device),
        _tensor(batch, "time_deltas", dtype=torch.float32, device=device),
        _tensor(batch, "command_ids", dtype=torch.long, device=device),
        _tensor(batch, "command_times", dtype=torch.float32, device=device),
        _tensor(batch, "command_mask", dtype=torch.bool, device=device),
        _tensor(batch, "sequence_mask", dtype=torch.bool, device=device),
    )


def _target_table(batch: Any, device: torch.device, *, prefix: str = "target") -> _TargetTable:
    names = {
        "batch_index": f"{prefix}_batch_index",
        "channel_ids": f"{prefix}_channel_ids",
        "times": f"{prefix}_times",
        "values": f"{prefix}_values",
    }
    if _optional(batch, names["values"]) is None:
        empty_float = torch.empty(0, dtype=torch.float32, device=device)
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        return _TargetTable(empty_long, empty_long, empty_float, empty_float)
    table = _TargetTable(
        batch_index=_tensor(batch, names["batch_index"], dtype=torch.long, device=device).reshape(-1),
        channel_ids=_tensor(batch, names["channel_ids"], dtype=torch.long, device=device).reshape(-1),
        times=_tensor(batch, names["times"], dtype=torch.float32, device=device).reshape(-1),
        values=_tensor(batch, names["values"], dtype=torch.float32, device=device).reshape(-1),
    )
    lengths = {
        table.batch_index.numel(),
        table.channel_ids.numel(),
        table.times.numel(),
        table.values.numel(),
    }
    if len(lengths) != 1:
        raise ValueError(f"{prefix} arrays must have equal length")
    return table


def _objective_targets(batch: Any, device: torch.device) -> tuple[_TargetTable, _TargetTable]:
    all_targets = _target_table(batch, device)
    for prefix in ("future_target", "forecast_target"):
        if _optional(batch, f"{prefix}_values") is not None:
            return all_targets, _target_table(batch, device, prefix=prefix)
    future_flags = _optional(batch, "target_is_future")
    if future_flags is None:
        empty = all_targets.select(torch.zeros(all_targets.count, dtype=torch.bool, device=device))
        return all_targets, empty
    flags = torch.as_tensor(future_flags, dtype=torch.bool, device=device).reshape(-1)
    if flags.numel() != all_targets.count:
        raise ValueError("target_is_future must match the flattened target table")
    # Future observations also receive the forecast loss.
    return all_targets, all_targets.select(flags)


def _negative_inputs(batch: Any, device: torch.device) -> tuple[Tensor, Tensor, Tensor] | None:
    if _optional(batch, "negative_command_ids") is None:
        return None
    return (
        _tensor(batch, "negative_command_ids", dtype=torch.long, device=device),
        _tensor(batch, "negative_command_times", dtype=torch.float32, device=device),
        _tensor(batch, "negative_command_mask", dtype=torch.bool, device=device),
    )


def _forecast_representations(
    encoder: NeuralJumpCDEEncoder,
    inputs: tuple[Tensor, ...],
    batch: Any,
    *,
    forecast_hours: float,
    device: torch.device,
) -> EncoderRepresentations:
    """Encode only the context preceding the held-out future interval."""

    block_starts = _optional(batch, "block_starts_ns")
    block_ends = _optional(batch, "block_ends_ns")
    if block_starts is None or block_ends is None:
        # Small fixtures may already provide a forecast-truncated path.
        return encoder.encode_pretraining(*inputs)
    starts = torch.as_tensor(block_starts, dtype=torch.int64, device=device).reshape(-1)
    ends = torch.as_tensor(block_ends, dtype=torch.int64, device=device).reshape(-1)
    if starts.numel() != inputs[0].shape[0] or ends.numel() != inputs[0].shape[0]:
        raise ValueError("block start/end arrays must match the pretraining batch")
    durations = (ends - starts).to(torch.float64) / _NS_PER_HOUR
    cutoff = (durations - forecast_hours).to(dtype=inputs[1].dtype)
    if not bool((cutoff > 0).all().item()):
        raise ValueError("forecast interval must be shorter than every SSL block")
    context_mask = inputs[7] & (inputs[1] < cutoff.unsqueeze(1))
    if not bool(context_mask[:, 0].all().item()):
        raise ValueError("every forecast block needs context before its forecast interval")
    command_mask = inputs[6] & (inputs[5] < cutoff.unsqueeze(1))
    return encoder.encode_pretraining(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        command_mask,
        context_mask,
    )


def _batch_objectives(
    encoder: NeuralJumpCDEEncoder,
    decoder: ObservationDecoder,
    discriminator: CommandResponseDiscriminator,
    batch: Any,
    *,
    weights: PretrainingObjectiveWeights,
    forecast_hours: float,
    device: torch.device,
) -> _BatchLosses:
    inputs = _encoder_inputs(batch, device)
    representations = encoder.encode_pretraining(*inputs)
    reconstruction_targets, future_targets = _objective_targets(batch, device)
    reconstruction = reconstruction_huber_loss(
        decoder,
        representations.joint,
        reconstruction_targets.batch_index,
        reconstruction_targets.channel_ids,
        reconstruction_targets.times,
        reconstruction_targets.values,
    )
    forecast_representations = (
        _forecast_representations(
            encoder,
            inputs,
            batch,
            forecast_hours=forecast_hours,
            device=device,
        )
        if future_targets.count
        else representations
    )
    future = future_huber_loss(
        decoder,
        forecast_representations.joint,
        future_targets.batch_index,
        future_targets.channel_ids,
        future_targets.times,
        future_targets.values,
    )

    negative = _negative_inputs(batch, device)
    if negative is None:
        empty = representations.joint.new_empty((0,))
        command_result = CommandMatchResult(representations.joint.sum() * 0.0, empty, empty, 0)
    else:
        negative_ids, negative_times, negative_mask = negative
        negative_representations = encoder.encode_pretraining(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            negative_ids,
            negative_times,
            negative_mask,
            inputs[7],
        )
        matched_mask = inputs[6].any(dim=1) & negative_mask.any(dim=1)
        explicit_match_mask = _optional(batch, "command_match_mask")
        if explicit_match_mask is not None:
            matched_mask &= torch.as_tensor(
                explicit_match_mask, dtype=torch.bool, device=device
            ).reshape(-1)
        command_result = command_response_matching_loss(
            discriminator,
            representations.telemetry_only,
            representations.joint,
            negative_representations.joint,
            example_mask=matched_mask,
        )
    total = (
        weights.reconstruction * reconstruction
        + weights.future * future
        + weights.command * command_result.loss
    )
    return _BatchLosses(
        reconstruction=reconstruction,
        future=future,
        command=command_result.loss,
        total=total,
        reconstruction_count=reconstruction_targets.count,
        future_count=future_targets.count,
        command_pair_count=command_result.pair_count,
        command_logits=command_result.logits,
        command_labels=command_result.labels,
    )


def _safe_auroc(logits: list[Tensor], labels: list[Tensor]) -> float | None:
    if not logits:
        return None
    predictions = torch.cat(logits).detach().cpu().numpy()
    targets = torch.cat(labels).detach().cpu().numpy()
    if len(set(targets.tolist())) < 2:
        return None
    return float(roc_auc_score(targets, predictions))


def _weighted_average(values: list[tuple[float, int]]) -> float:
    positive = [(value, count) for value, count in values if count > 0]
    if not positive:
        return 0.0
    return sum(value * count for value, count in positive) / sum(count for _, count in positive)


def _run_epoch(
    encoder: NeuralJumpCDEEncoder,
    decoder: ObservationDecoder,
    discriminator: CommandResponseDiscriminator,
    batches: Iterable[Any],
    *,
    weights: PretrainingObjectiveWeights,
    forecast_hours: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> _EpochMetrics:
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    discriminator.train(training)
    reconstruction_values: list[tuple[float, int]] = []
    future_values: list[tuple[float, int]] = []
    command_values: list[tuple[float, int]] = []
    total_values: list[tuple[float, int]] = []
    logits: list[Tensor] = []
    labels: list[Tensor] = []
    batch_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in batches:
            batch_count += 1
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            losses = _batch_objectives(
                encoder,
                decoder,
                discriminator,
                batch,
                weights=weights,
                forecast_hours=forecast_hours,
                device=device,
            )
            if optimizer is not None:
                losses.total.backward()
                nn.utils.clip_grad_norm_(
                    [
                        *encoder.parameters(),
                        *decoder.parameters(),
                        *discriminator.parameters(),
                    ],
                    max_norm=gradient_clip,
                )
                optimizer.step()
            reconstruction_values.append(
                (float(losses.reconstruction.detach()), losses.reconstruction_count)
            )
            future_values.append((float(losses.future.detach()), losses.future_count))
            command_values.append((float(losses.command.detach()), losses.command_pair_count))
            total_weight = max(
                1,
                losses.reconstruction_count + losses.future_count + losses.command_pair_count,
            )
            total_values.append((float(losses.total.detach()), total_weight))
            if losses.command_logits.numel():
                logits.append(losses.command_logits)
                labels.append(losses.command_labels)
    if batch_count == 0:
        raise ValueError("pretraining requires at least one batch per epoch")
    return _EpochMetrics(
        reconstruction=_weighted_average(reconstruction_values),
        future=_weighted_average(future_values),
        command=_weighted_average(command_values),
        total=_weighted_average(total_values),
        command_auroc=_safe_auroc(logits, labels),
    )


def _set_epoch(batches: Any, epoch: int) -> None:
    candidate = getattr(batches, "dataset", batches)
    setter = getattr(candidate, "set_epoch", None)
    if callable(setter):
        setter(epoch)


def is_partial_coverage(
    coverage_report: Mapping[str, Any] | object | None,
    *,
    completed_epochs: int,
    target_folds: int,
) -> bool:
    """Return whether a checkpoint lacks one audited full target-fold cycle."""

    if completed_epochs < target_folds:
        return True
    coverage = _coverage_mapping(coverage_report)
    if coverage is None:
        return False
    if bool(coverage.get("partial_coverage", False)):
        return True
    fraction = coverage.get("coverage_fraction")
    never_used = coverage.get("never_used", coverage.get("number_never_used"))
    repeated = coverage.get("used_more_than_once", coverage.get("number_used_more_than_once"))
    if fraction is not None and not math.isclose(float(fraction), 1.0, abs_tol=1e-12):
        return True
    if never_used is not None and int(never_used) != 0:
        return True
    return repeated is not None and int(repeated) != 0


def pretrain_encoder(
    encoder: NeuralJumpCDEEncoder,
    training_batches: Iterable[Any],
    validation_batches: Iterable[Any],
    *,
    epochs: int,
    output_dir: str | Path | None = None,
    target_folds: int = 8,
    forecast_hours: float = 1.0,
    channel_embedding_dim: int = 16,
    time_embedding_dim: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    gradient_clip: float = 1.0,
    reconstruction_weight: float = 1.0,
    forecast_weight: float = 0.5,
    command_weight: float = 0.2,
    early_stopping_patience: int | None = None,
    seed: int = 42,
    device: str | torch.device = "cpu",
    protocol_hash: str | None = None,
    scaler_hash: str | None = None,
    coverage_report: Mapping[str, Any] | object | None = None,
    manifest_context: Mapping[str, Any] | None = None,
    scaler: Mapping[str, Any] | object | str | Path | None = None,
) -> PretrainingResult:
    """Train the encoder on exhaustive raw-observation targets."""

    if epochs < 1 or target_folds < 1:
        raise ValueError("epochs and target_folds must be positive")
    if not math.isfinite(forecast_hours) or forecast_hours < 0:
        raise ValueError("forecast_hours must be finite and non-negative")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive when supplied")
    weights = PretrainingObjectiveWeights(
        reconstruction=reconstruction_weight,
        future=forecast_weight,
        command=command_weight,
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    selected_device = torch.device(device)
    encoder.to(selected_device)
    decoder = ObservationDecoder(
        encoder.representation_dim,
        encoder.telemetry_channels,
        channel_embedding_dim=channel_embedding_dim,
        time_embedding_dim=time_embedding_dim,
    ).to(selected_device)
    discriminator = CommandResponseDiscriminator(encoder.representation_dim).to(selected_device)
    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters(), *discriminator.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history: list[PretrainingHistoryRecord] = []
    best_encoder_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_score = math.inf
    eligible_without_improvement = 0
    completed_epochs = 0
    for epoch_index in range(epochs):
        _set_epoch(training_batches, epoch_index)
        _set_epoch(validation_batches, epoch_index)
        training = _run_epoch(
            encoder,
            decoder,
            discriminator,
            training_batches,
            weights=weights,
            forecast_hours=forecast_hours,
            device=selected_device,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
        )
        validation = _run_epoch(
            encoder,
            decoder,
            discriminator,
            validation_batches,
            weights=weights,
            forecast_hours=forecast_hours,
            device=selected_device,
            optimizer=None,
            gradient_clip=gradient_clip,
        )
        completed_epochs = epoch_index + 1
        # Model selection stays inside the allowed SSL interval.
        command_auroc = 0.5 if validation.command_auroc is None else validation.command_auroc
        score = (
            weights.reconstruction * validation.reconstruction
            + weights.future * validation.future
            + weights.command * (1.0 - command_auroc)
        )
        history.append(
            PretrainingHistoryRecord(
                epoch=completed_epochs,
                active_target_fold=epoch_index % target_folds,
                training_reconstruction_huber=training.reconstruction,
                training_future_huber=training.future,
                training_command_bce=training.command,
                training_total_loss=training.total,
                validation_reconstruction_huber=validation.reconstruction,
                validation_future_huber=validation.future,
                validation_command_bce=validation.command,
                validation_command_auroc=validation.command_auroc,
                validation_score=score,
            )
        )

        # Select a checkpoint only after every target fold has been covered.
        checkpoint_eligible = completed_epochs >= target_folds
        if (checkpoint_eligible or epochs < target_folds) and score < best_score:
            best_score = score
            best_epoch = completed_epochs
            best_encoder_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in encoder.state_dict().items()
            }
            eligible_without_improvement = 0
        elif checkpoint_eligible:
            eligible_without_improvement += 1
        if (
            checkpoint_eligible
            and early_stopping_patience is not None
            and eligible_without_improvement >= early_stopping_patience
        ):
            break

    if best_encoder_state is None:
        raise RuntimeError("self-supervised training produced no encoder checkpoint")
    encoder.load_state_dict(best_encoder_state)
    partial = is_partial_coverage(
        coverage_report,
        completed_epochs=best_epoch,
        target_folds=target_folds,
    )
    result = PretrainingResult(
        encoder=encoder,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_score=best_score,
        completed_epochs=completed_epochs,
        target_folds=target_folds,
        seed=seed,
        forecast_hours=forecast_hours,
        channel_embedding_dim=channel_embedding_dim,
        time_embedding_dim=time_embedding_dim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
        partial_coverage=partial,
        objective_weights=weights,
    )
    if output_dir is not None:
        artifacts = save_pretraining_artifacts(
            result,
            output_dir,
            protocol_hash=protocol_hash,
            scaler_hash=scaler_hash,
            coverage_report=coverage_report,
            manifest_context=manifest_context,
            scaler=scaler,
        )
        result = replace(result, checkpoint_path=artifacts.checkpoint)
    return result


def pretrain_encoder_from_config(
    encoder: NeuralJumpCDEEncoder,
    training_batches: Iterable[Any],
    validation_batches: Iterable[Any],
    *,
    config: Mapping[str, Any] | object,
    output_dir: str | Path | None = None,
    protocol_hash: str | None = None,
    scaler_hash: str | None = None,
    coverage_report: Mapping[str, Any] | object | None = None,
    manifest_context: Mapping[str, Any] | None = None,
    scaler: Mapping[str, Any] | object | str | Path | None = None,
    device: str | torch.device = "cpu",
) -> PretrainingResult:
    """Map ``MirageConfig`` (or its pretraining mapping) to the core SSL API."""

    if isinstance(config, Mapping):
        nested_settings = config.get("pretraining")
        settings = nested_settings if isinstance(nested_settings, Mapping) else config
        seed = int(config.get("seed", 42))
        nested_outputs = config.get("outputs", {})
        outputs = nested_outputs if isinstance(nested_outputs, Mapping) else {}
    else:
        settings_candidate = getattr(config, "pretraining", None)
        if not isinstance(settings_candidate, Mapping):
            raise TypeError("config must be a pretraining mapping or expose .pretraining")
        settings = settings_candidate
        seed = int(getattr(config, "seed", 42))
        outputs_candidate = getattr(config, "outputs", {})
        outputs = outputs_candidate if isinstance(outputs_candidate, Mapping) else {}
    if output_dir is None:
        configured_output = outputs.get("pretraining")
        if configured_output is None:
            raise ValueError("output_dir is required when config has no outputs.pretraining")
        output_dir = Path(str(configured_output))
    epochs = int(settings["coverage_epochs"]) + int(settings.get("extra_epochs", 0))
    dataset = getattr(training_batches, "dataset", training_batches)
    if coverage_report is None:
        reporter = getattr(dataset, "coverage_report", None)
        if callable(reporter):
            # Audit the first exhaustive target-fold cycle.
            coverage_report = reporter(epochs=int(settings["target_folds"]))
    if scaler is None:
        scaler = getattr(dataset, "normalizer", None)
    context = {"pretraining_config": dict(settings)}
    if manifest_context is not None:
        context.update(dict(manifest_context))
    return pretrain_encoder(
        encoder,
        training_batches,
        validation_batches,
        epochs=epochs,
        output_dir=output_dir,
        target_folds=int(settings["target_folds"]),
        forecast_hours=float(settings["forecast_hours"]),
        channel_embedding_dim=int(settings.get("channel_embedding_dim", 16)),
        time_embedding_dim=int(settings.get("time_embedding_dim", 16)),
        learning_rate=float(settings.get("learning_rate", 3e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
        gradient_clip=float(settings.get("gradient_clip", 1.0)),
        reconstruction_weight=float(settings.get("reconstruction_weight", 1.0)),
        forecast_weight=float(settings.get("forecast_weight", 0.5)),
        command_weight=float(settings.get("command_weight", 0.2)),
        seed=seed,
        device=device,
        protocol_hash=protocol_hash,
        scaler_hash=scaler_hash,
        coverage_report=coverage_report,
        manifest_context=context,
        scaler=scaler,
    )


def _json_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_data(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_data(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    prepared = _json_data(payload)
    if not isinstance(prepared, Mapping):
        raise TypeError("canonical hash payload must be a mapping")
    return canonical_sha256(prepared)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_scaler(scaler: Mapping[str, Any] | object | str | Path) -> Mapping[str, Any]:
    if isinstance(scaler, Mapping):
        return scaler
    to_dict = getattr(scaler, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise ValueError("scaler.to_dict() must return a mapping")
        return payload
    if not isinstance(scaler, (str, Path)):
        raise TypeError("scaler must be a mapping, path, or expose to_dict()")
    payload = json.loads(Path(scaler).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scaler JSON must contain an object")
    return payload


def _coverage_mapping(
    coverage_report: Mapping[str, Any] | object | None,
) -> Mapping[str, Any] | None:
    if coverage_report is None:
        return None
    if isinstance(coverage_report, Mapping):
        return coverage_report
    to_dict = getattr(coverage_report, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("coverage_report must be a mapping or expose to_dict()")
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("coverage_report.to_dict() must return a mapping")
    return payload


def save_pretraining_artifacts(
    result: PretrainingResult,
    output_dir: str | Path,
    *,
    protocol_hash: str | None = None,
    scaler_hash: str | None = None,
    coverage_report: Mapping[str, Any] | object | None = None,
    manifest_context: Mapping[str, Any] | None = None,
    scaler: Mapping[str, Any] | object | str | Path | None = None,
) -> PretrainingArtifacts:
    """Persist an encoder-only checkpoint and its audit trail."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target / "encoder.pt"
    torch.save(result.encoder.state_dict(), checkpoint_path)
    checkpoint_hash = _file_sha256(checkpoint_path)

    coverage_mapping = _coverage_mapping(coverage_report)
    coverage: dict[str, Any] = (
        {} if coverage_mapping is None else dict(_json_data(coverage_mapping))
    )
    coverage["schema_version"] = str(
        coverage.get("schema_version", "mirage-pretraining-coverage/1.0")
    )
    coverage["target_folds"] = result.target_folds
    coverage["selected_checkpoint_epoch"] = result.best_epoch
    coverage["completed_epochs"] = result.completed_epochs
    coverage["partial_coverage"] = is_partial_coverage(
        coverage_report,
        completed_epochs=result.best_epoch,
        target_folds=result.target_folds,
    )
    coverage_hash = _canonical_hash(coverage)
    coverage_path = target / "coverage_report.json"
    _write_json(coverage_path, coverage)

    history_payload: dict[str, Any] = {
        "schema_version": "mirage-pretraining-history/1.0",
        "best_epoch": result.best_epoch,
        "best_validation_score": result.best_validation_score,
        "records": [asdict(record) for record in result.history],
    }
    history_path = target / "training_history.json"
    _write_json(history_path, history_payload)

    scaler_path: Path | None = None
    if scaler is not None:
        scaler_payload = dict(_json_data(_read_scaler(scaler)))
        embedded_scaler_hash = scaler_payload.pop("scaler_hash", None)
        calculated_scaler_hash = _canonical_hash(scaler_payload)
        if embedded_scaler_hash is not None and embedded_scaler_hash != calculated_scaler_hash:
            raise ValueError("saved scaler hash does not match scaler contents")
        if scaler_hash is not None and scaler_hash != calculated_scaler_hash:
            raise ValueError("supplied scaler_hash does not match scaler contents")
        scaler_hash = calculated_scaler_hash
        scaler_path = target / "scaler.json"
        _write_json(scaler_path, scaler_payload | {"scaler_hash": scaler_hash})

    manifest: dict[str, Any] = {
        "schema_version": "mirage-pretraining/1.0",
        "architecture": "command-conditioned-neural-jump-cde",
        "encoder_only": True,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "protocol_hash": protocol_hash,
        "scaler_hash": scaler_hash,
        "coverage_hash": coverage_hash,
        "partial_coverage": coverage["partial_coverage"],
        "best_epoch": result.best_epoch,
        "completed_epochs": result.completed_epochs,
        "target_folds": result.target_folds,
        "seed": result.seed,
        "best_validation_score": result.best_validation_score,
        "objective_weights": asdict(result.objective_weights),
        "pretraining_config": {
            "forecast_hours": result.forecast_hours,
            "channel_embedding_dim": result.channel_embedding_dim,
            "time_embedding_dim": result.time_embedding_dim,
            "learning_rate": result.learning_rate,
            "weight_decay": result.weight_decay,
            "gradient_clip": result.gradient_clip,
        },
        "encoder": {
            "telemetry_channels": result.encoder.telemetry_channels,
            "num_commands": result.encoder.num_commands,
            "hidden_dim": result.encoder.hidden_dim,
            "representation_dim": result.encoder.representation_dim,
            "command_embedding_dim": result.encoder.command_embedding.embedding_dim,
            "affected_embedding_dim": result.encoder.affected_encoder[0].out_features,
        },
        "context": {} if manifest_context is None else dict(_json_data(manifest_context)),
    }
    manifest_path = target / "pretraining_manifest.json"
    _write_json(manifest_path, manifest)
    return PretrainingArtifacts(
        checkpoint=checkpoint_path,
        manifest=manifest_path,
        history=history_path,
        coverage_report=coverage_path,
        scaler=scaler_path,
    )


def validate_pretraining_artifacts(
    output_dir: str | Path,
    *,
    require_full_coverage: bool = True,
) -> Mapping[str, Any]:
    """Validate hashes and coverage flags without constructing an encoder."""

    target = Path(output_dir)
    manifest_path = target / "pretraining_manifest.json"
    coverage_path = target / "coverage_report.json"
    history_path = target / "training_history.json"
    for path in (manifest_path, coverage_path, history_path, target / "encoder.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"missing pretraining artifact: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "mirage-pretraining/1.0":
        raise ValueError("incompatible pretraining manifest schema")
    if not bool(manifest.get("encoder_only")):
        raise ValueError("pretraining checkpoint is not declared encoder-only")
    if manifest.get("checkpoint_sha256") != _file_sha256(target / "encoder.pt"):
        raise ValueError("pretraining checkpoint hash mismatch")
    if manifest.get("coverage_hash") != _canonical_hash(coverage):
        raise ValueError("pretraining coverage-report hash mismatch")
    if bool(manifest.get("partial_coverage")) != bool(coverage.get("partial_coverage")):
        raise ValueError("manifest and coverage report disagree")
    if require_full_coverage and bool(manifest.get("partial_coverage")):
        raise RuntimeError("pretraining checkpoint has partial target-fold coverage")
    return manifest


def load_pretrained_encoder_checkpoint(
    encoder: NeuralJumpCDEEncoder,
    checkpoint_or_dir: str | Path,
    *,
    allow_partial: bool = False,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Verify and load encoder-only weights; reject partial coverage by default."""

    supplied = Path(checkpoint_or_dir)
    output_dir = supplied if supplied.is_dir() else supplied.parent
    checkpoint = output_dir / "encoder.pt" if supplied.is_dir() else supplied
    manifest = validate_pretraining_artifacts(
        output_dir, require_full_coverage=not allow_partial
    )
    if checkpoint.name != manifest.get("checkpoint"):
        raise ValueError("checkpoint filename differs from the pretraining manifest")
    state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise ValueError("encoder checkpoint does not contain a tensor state dictionary")
    encoder.load_state_dict(state)
    return manifest


# Descriptive alias used by experiment scripts.
run_pretraining = pretrain_encoder
