"""end-to-end forecasting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from mirage.audit import (
    CoverageReport,
    LeakageAuditReport,
    assert_partition_isolation,
    assert_temporal_boundaries,
    assert_train_only_normalizer,
    build_split_fingerprint,
)
from mirage.config import MirageConfig
from mirage.encoders import NeuralJumpCDEEncoder
from mirage.forecasting import FullDataForecastResult, run_full_data_forecasting
from mirage.normalization import ChannelNormalizer
from mirage.preprocessing import (
    MISSION1_ARCHIVE_MD5,
    EpisodeConfig,
    EventEpisode,
    Mission1Source,
    build_event_episodes,
    build_event_index,
    eligible_event_classes,
    extract_mission1_archive,
    load_episode_cache,
    save_episode_cache,
)
from mirage.pretraining import pretrain_encoder_from_config
from mirage.pretraining_dataset import PretrainingDataset, collate_pretraining
from mirage.protocol import TemporalProtocol
from mirage.protocol import build_temporal_protocol as make_protocol
from mirage.ssl_store import SSLStore, build_ssl_store


def build_temporal_protocol(config: MirageConfig) -> TemporalProtocol:
    """Create the immutable time split used by every MIRAGE artifact."""

    extract_mission1_archive(config.data["archive"], config.raw_dir)
    source = Mission1Source(config.raw_dir)
    return make_protocol(
        build_event_index(source),
        mission_start=source.mission_start,
        event_pre_hours=float(config.data["pre_hours"]),
        event_post_hours=float(config.data["post_hours"]),
        command_history_hours=float(config.data["telecommand_history_hours"]),
        ssl_context_hours=float(config.pretraining["block_hours"])
        - float(config.pretraining["forecast_hours"]),
        ssl_forecast_hours=float(config.pretraining["forecast_hours"]),
        train_fraction=float(config.split["train_fraction"]),
        calibration_fraction=float(config.split["calibration_fraction"]),
    )


def prepare_ssl_store(config: MirageConfig, protocol: TemporalProtocol) -> Path:
    """Build the lossless raw store and train-only normalizer."""

    source = Mission1Source(config.raw_dir)
    store = build_ssl_store(
        source,
        protocol,
        output_dir=config.ssl_store_dir,
        block_hours=float(config.pretraining["block_hours"]),
        normalization_clip=float(config.normalization["clip"]),
        normalization_epsilon=float(config.normalization["epsilon"]),
    )
    normalizer = ChannelNormalizer.load(store / "scaler.json")
    assert_train_only_normalizer(normalizer, protocol)
    audit_dir = Path(config.outputs["audit"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    protocol.save(audit_dir / "protocol.json")
    build_split_fingerprint(
        protocol,
        archive_checksum=MISSION1_ARCHIVE_MD5,
        channel_order=source.channel_names,
        command_order=source.command_names,
        configuration={
            "data": config.data,
            "split": config.split,
            "normalization": config.normalization,
            "pretraining": config.pretraining,
        },
    ).save(audit_dir / "split_fingerprint.json")
    return store / "manifest.json"


def pretrain_mirage_encoder(
    config: MirageConfig, protocol: TemporalProtocol, *, device: str = "cpu"
) -> Path:
    """Pretrain the encoder over every allowed observation target."""

    store = SSLStore.open(config.ssl_store_dir)
    normalizer = ChannelNormalizer.load(config.ssl_store_dir / "scaler.json")
    if store.manifest.get("protocol_hash") != protocol.protocol_hash:
        raise ValueError("SSL store protocol hash differs from requested run")
    block_ids = tuple(int(value) for value in store.block_ids)
    validation_count = max(
        1, int(len(block_ids) * float(config.pretraining["ssl_validation_fraction"]))
    )
    fitting_ids, validation_ids = block_ids[:-validation_count], block_ids[-validation_count:]

    def dataset(ids: tuple[int, ...]) -> PretrainingDataset:
        return PretrainingDataset(
            store,
            normalizer=normalizer,
            target_folds=int(config.pretraining["target_folds"]),
            max_steps=int(config.pretraining["max_steps"]),
            max_commands=int(config.pretraining["max_commands"]),
            forecast_hours=float(config.pretraining["forecast_hours"]),
            seed=config.seed,
            block_ids=ids,
            command_negative_era_days=float(config.pretraining["command_negative_era_days"]),
            command_count_tolerance=float(config.pretraining["command_count_tolerance"]),
        )

    fitting, validation, complete = (
        dataset(fitting_ids),
        dataset(validation_ids),
        dataset(block_ids),
    )
    fitting_loader = DataLoader(
        fitting,
        batch_size=int(config.pretraining["batch_size"]),
        shuffle=False,
        collate_fn=collate_pretraining,
    )
    validation_loader = DataLoader(
        validation,
        batch_size=int(config.pretraining["batch_size"]),
        shuffle=False,
        collate_fn=collate_pretraining,
    )
    encoder = NeuralJumpCDEEncoder(
        telemetry_channels=len(store.channel_order),
        num_commands=len(store.command_order),
        hidden_dim=int(config.model["hidden_dim"]),
        representation_dim=int(config.model["hidden_dim"]),
        command_embedding_dim=int(config.model["command_embedding_dim"]),
        affected_embedding_dim=int(config.model["affected_embedding_dim"]),
    )
    result = pretrain_encoder_from_config(
        encoder,
        fitting_loader,
        validation_loader,
        config=config,
        output_dir=config.outputs["pretraining"],
        protocol_hash=protocol.protocol_hash,
        scaler_hash=normalizer.scaler_hash,
        coverage_report=complete.coverage_report(
            epochs=int(config.pretraining["coverage_epochs"])
            + int(config.pretraining["extra_epochs"])
        ),
        scaler=normalizer,
        device=device,
        manifest_context={
            "archive_md5": MISSION1_ARCHIVE_MD5,
            "ssl_store_manifest": store.manifest,
        },
    )
    if result.checkpoint_path is None:
        raise RuntimeError("pretraining did not write an encoder checkpoint")
    return result.checkpoint_path


def event_cache_dir(config: MirageConfig) -> Path:
    return config.cache_dir / "online_end"


def prepare_event_episodes(config: MirageConfig, protocol: TemporalProtocol) -> Path:
    """Build the compact labelled cache used only for anomaly evaluation."""

    normalizer = ChannelNormalizer.load(config.ssl_store_dir / "scaler.json")
    assert_train_only_normalizer(normalizer, protocol)
    source = Mission1Source(config.raw_dir)
    episodes = build_event_episodes(
        source,
        build_event_index(source),
        config=EpisodeConfig(
            pre_hours=float(config.data["pre_hours"]),
            post_hours=float(config.data["post_hours"]),
            telecommand_history_hours=float(config.data["telecommand_history_hours"]),
            max_steps=int(config.data["max_steps"]),
            max_commands=int(config.data["max_commands"]),
            protocol="online_end",
        ),
        command_cache=config.cache_dir / "telecommands.csv",
        normalizer=normalizer,
    )
    return save_episode_cache(
        episodes,
        event_cache_dir(config),
        metadata={
            "protocol_hash": protocol.protocol_hash,
            "scaler_hash": normalizer.scaler_hash,
            "archive_md5": MISSION1_ARCHIVE_MD5,
        },
    )


def _partitions(
    episodes: tuple[EventEpisode, ...], config: MirageConfig, protocol: TemporalProtocol
) -> dict[str, tuple[EventEpisode, ...]]:
    source = Mission1Source(config.raw_dir)
    events = build_event_index(source)
    split = {
        "train": protocol.outer_train_event_ids,
        "calibration": protocol.calibration_event_ids,
        "test": protocol.test_event_ids,
    }
    classes = eligible_event_classes(
        events, split, min_train=int(config.split["min_train_per_class"])
    )
    lookup = {event_id: name for name, ids in split.items() for event_id in ids}
    selected = {
        name: tuple(
            e for e in episodes if e.event_class in classes and lookup.get(e.event_id) == name
        )
        for name in split
    }
    if any(not values for values in selected.values()):
        raise ValueError("event cache has an empty chronological partition")
    return selected


def run_forecasting(
    config: MirageConfig, protocol: TemporalProtocol | None = None
) -> FullDataForecastResult:
    """Run the primary full-corpus forecasting experiment."""

    selected_protocol = protocol or build_temporal_protocol(config)
    episodes = load_episode_cache(event_cache_dir(config))
    return run_full_data_forecasting(
        config, selected_protocol, _partitions(episodes, config, selected_protocol)
    )


def audit_full_data_pipeline(config: MirageConfig) -> Path:
    """Fail closed unless full coverage, boundaries, and forecast artifacts agree."""

    protocol = build_temporal_protocol(config)
    store = SSLStore.open(config.ssl_store_dir)
    normalizer = ChannelNormalizer.load(config.ssl_store_dir / "scaler.json")
    assert_train_only_normalizer(normalizer, protocol)
    dataset = PretrainingDataset(
        store,
        normalizer=normalizer,
        target_folds=int(config.pretraining["target_folds"]),
        max_steps=int(config.pretraining["max_steps"]),
        max_commands=int(config.pretraining["max_commands"]),
        forecast_hours=float(config.pretraining["forecast_hours"]),
        seed=config.seed,
    )
    summary = dataset.coverage_report(
        epochs=int(config.pretraining["coverage_epochs"]) + int(config.pretraining["extra_epochs"])
    )
    coverage = CoverageReport(
        raw_training_observations=int(summary["raw_training_observations"]),
        target_assignments=int(summary["target_assignments"]),
        used_as_reconstruction_targets=int(summary["used_as_reconstruction_targets"]),
        used_more_than_once=int(summary["used_more_than_once"]),
        never_used=int(summary["never_used"]),
        coverage_fraction=float(summary["coverage_fraction"]),
        partial_coverage=bool(summary["partial_coverage"]),
    )
    temporal = assert_temporal_boundaries(
        protocol,
        max_pretraining_context_timestamp=pd.Timestamp(int(store.block_ends_ns.max()) - 1),
        max_pretraining_target_timestamp=store.manifest["max_pretraining_target_timestamp"],
        max_pretraining_command_timestamp=store.manifest.get("max_pretraining_command_timestamp"),
        min_calibration_episode_timestamp=protocol.protected_start,
    )
    episodes = load_episode_cache(event_cache_dir(config))
    partitions = _partitions(episodes, config, protocol)
    assert_partition_isolation(
        protocol,
        training_event_ids=(e.event_id for e in partitions["train"]),
        calibration_event_ids=(e.event_id for e in partitions["calibration"]),
        test_event_ids=(e.event_id for e in partitions["test"]),
    )
    forecast_path = Path(config.outputs["forecasting"]) / "forecasting_results.json"
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    if forecast.get("protocol_hash") != protocol.protocol_hash:
        raise AssertionError("forecasting protocol hash mismatch")
    report = LeakageAuditReport(
        protocol_hash=protocol.protocol_hash,
        split_fingerprint_hash=build_split_fingerprint(
            protocol,
            archive_checksum=MISSION1_ARCHIVE_MD5,
            channel_order=store.channel_order,
            command_order=store.command_order,
            configuration={"data": config.data, "split": config.split},
        ).fingerprint_hash,
        scaler_hash=normalizer.scaler_hash,
        coverage=coverage,
        temporal_overlap=temporal,
        forecasting_coverage=dict(forecast["coverage"]),
    )
    return report.save(Path(config.outputs["audit"]) / "audit_report.json")
