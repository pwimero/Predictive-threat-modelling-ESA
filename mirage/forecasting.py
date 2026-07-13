"""Full-corpus telemetry forecasting and residual-pattern anomaly prediction."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mirage.config import MirageConfig
from mirage.dataset import collate_episodes
from mirage.encoders import NeuralJumpCDEEncoder
from mirage.normalization import ChannelNormalizer
from mirage.preprocessing import EventEpisode, Mission1Source
from mirage.protocol import TemporalProtocol, rolling_origin_folds
from mirage.ssl_store import SSLStore

_NS_PER_HOUR = 3_600_000_000_000
_FEATURES = ("last", "mean", "std", "change", "log_observations", "log_commands")


@dataclass(frozen=True)
class FullDataForecastResult:
    """Persisted full-data forecasting result and primary paper metrics."""

    metrics: dict[str, Any]
    metrics_path: Path
    model_path: Path


def _block_features(
    store: SSLStore,
    normalizer: ChannelNormalizer,
    *,
    context_hours: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], dict[str, int]]:
    blocks = len(store.block_ids)
    channels = len(store.channel_order)
    features = np.full((blocks, channels, len(_FEATURES)), np.nan, dtype=np.float32)
    targets = np.full((blocks, channels), np.nan, dtype=np.float32)
    persistence = np.full_like(targets, np.nan)
    context_count = 0
    target_count = 0
    command_counts = np.zeros(blocks, dtype=np.float32)
    for block in range(blocks):
        timestamps, _ = store.command_block(block)
        cutoff = int(store.block_starts_ns[block] + context_hours * _NS_PER_HOUR)
        command_counts[block] = np.log1p(np.count_nonzero(timestamps < cutoff))
    for channel_id, channel in enumerate(store.channel_order):
        for block in range(blocks):
            timestamps, raw = store.channel_block(block, channel_id)
            cutoff = int(store.block_starts_ns[block] + context_hours * _NS_PER_HOUR)
            context_mask = timestamps < cutoff
            future_mask = ~context_mask
            context_count += int(context_mask.sum())
            target_count += int(future_mask.sum())
            if not context_mask.any() or not future_mask.any():
                continue
            values = normalizer.transform_channel(channel, np.asarray(raw, dtype=np.float32))
            context = values[context_mask]
            future = values[future_mask]
            features[block, channel_id] = (
                context[-1],
                context.mean(),
                context.std(),
                context[-1] - context[0],
                np.log1p(len(context)),
                command_counts[block],
            )
            targets[block, channel_id] = future.mean()
            persistence[block, channel_id] = context[-1]
    return (
        features,
        targets,
        persistence,
        {
            "context_observations": context_count,
            "forecast_target_observations": target_count,
            "total_observations": context_count + target_count,
        },
    )


def _fit_channel_models(
    features: NDArray[np.float32],
    targets: NDArray[np.float32],
    indices: NDArray[np.int64],
    *,
    alpha: float,
) -> list[Ridge]:
    models: list[Ridge] = []
    for channel in range(targets.shape[1]):
        valid = indices[np.isfinite(targets[indices, channel])]
        if len(valid) < 2:
            raise ValueError(f"channel {channel} has insufficient forecasting blocks")
        models.append(
            Ridge(alpha=alpha, solver="lsqr").fit(features[valid, channel], targets[valid, channel])
        )
    return models


def _predict_blocks(models: list[Ridge], features: NDArray[np.float32]) -> NDArray[np.float32]:
    predictions = np.full(features.shape[:2], np.nan, dtype=np.float32)
    for channel, model in enumerate(models):
        valid = np.isfinite(features[:, channel]).all(axis=1)
        predictions[valid, channel] = model.predict(features[valid, channel]).astype(np.float32)
    return predictions


def _errors(
    predictions: NDArray[np.float32],
    targets: NDArray[np.float32],
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    valid = np.isfinite(predictions[indices]) & np.isfinite(targets[indices])
    return np.abs(predictions[indices][valid] - targets[indices][valid]).astype(np.float64)


def _bootstrap_improvement(
    model_errors: NDArray[np.float64],
    persistence_errors: NDArray[np.float64],
    *,
    seed: int,
    samples: int,
) -> list[float]:
    if len(model_errors) != len(persistence_errors):
        raise ValueError("paired forecast errors must align")
    rng = np.random.default_rng(seed)
    gains = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(model_errors), size=len(model_errors))
        gains[index] = 1.0 - (model_errors[selected].mean() / persistence_errors[selected].mean())
    return np.quantile(gains, [0.025, 0.975]).tolist()


def _episode_forecast_features(
    episodes: tuple[EventEpisode, ...],
    models: list[Ridge],
    *,
    context_hours: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    model_rows: list[NDArray[np.float64]] = []
    persistence_rows: list[NDArray[np.float64]] = []
    model_errors: list[float] = []
    persistence_errors: list[float] = []
    for episode in episodes:
        event_start = 6.0
        context_start = event_start - context_hours
        context_mask = (episode.timestamps >= context_start) & (episode.timestamps < event_start)
        future_mask = (episode.timestamps >= event_start) & (
            episode.timestamps <= event_start + 1.0
        )
        if not context_mask.any() or not future_mask.any():
            raise ValueError(f"episode {episode.event_id} lacks context or forecast observations")
        command_count = np.log1p(np.count_nonzero(episode.command_times < event_start))
        predictions = np.empty(len(models), dtype=np.float64)
        actual = np.empty(len(models), dtype=np.float64)
        persistence = np.empty(len(models), dtype=np.float64)
        for channel, model in enumerate(models):
            context = episode.telemetry[context_mask, channel].astype(np.float64)
            future = episode.telemetry[future_mask, channel].astype(np.float64)
            row = np.asarray(
                [
                    [
                        context[-1],
                        context.mean(),
                        context.std(),
                        context[-1] - context[0],
                        np.log1p(len(context)),
                        command_count,
                    ]
                ]
            )
            predictions[channel] = model.predict(row)[0]
            actual[channel] = future.mean()
            persistence[channel] = context[-1]
        model_residual = actual - predictions
        persistence_residual = actual - persistence
        model_rows.append(np.concatenate((model_residual, np.abs(model_residual))))
        persistence_rows.append(
            np.concatenate((persistence_residual, np.abs(persistence_residual)))
        )
        model_errors.extend(np.abs(model_residual).tolist())
        persistence_errors.extend(np.abs(persistence_residual).tolist())
    return (
        np.asarray(model_rows),
        np.asarray(persistence_rows),
        {
            "forecast_mae": float(np.mean(model_errors)),
            "persistence_mae": float(np.mean(persistence_errors)),
        },
    )


def _raw_event_forecast_features(
    episodes: tuple[EventEpisode, ...],
    models: list[Ridge],
    source: Mission1Source,
    normalizer: ChannelNormalizer,
    *,
    context_hours: float,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    dict[str, float],
]:
    """Score exact post-protected ESA observations at the training horizon."""

    count = len(episodes)
    channels = len(models)
    predicted = np.full((count, channels), np.nan, dtype=np.float64)
    actual = np.full_like(predicted, np.nan)
    persistence = np.full_like(predicted, np.nan)
    context_features = np.zeros((count, channels, len(_FEATURES)), dtype=np.float64)
    starts_ns = np.asarray(
        [np.datetime64(episode.start_time, "ns").astype(np.int64) for episode in episodes],
        dtype=np.int64,
    )
    commands = source.load_command_events()
    command_ns = commands["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    command_counts = np.asarray(
        [
            np.log1p(
                np.count_nonzero(
                    (command_ns >= start - int(context_hours * _NS_PER_HOUR)) & (command_ns < start)
                )
            )
            for start in starts_ns
        ],
        dtype=np.float64,
    )
    for channel, name in enumerate(source.channel_names):
        frame = source.load_channel(name)
        timestamps = frame.index.asi8.astype(np.int64, copy=False)
        values = normalizer.transform_channel(
            name, frame.iloc[:, 0].to_numpy(dtype=np.float32)
        ).astype(np.float64)
        for row, start in enumerate(starts_ns):
            context_start = start - int(context_hours * _NS_PER_HOUR)
            future_end = start + _NS_PER_HOUR
            left = int(np.searchsorted(timestamps, context_start, side="left"))
            middle = int(np.searchsorted(timestamps, start, side="left"))
            right = int(np.searchsorted(timestamps, future_end, side="right"))
            context = values[left:middle]
            future = values[middle:right]
            if len(context) == 0 or len(future) == 0:
                continue
            features = np.asarray(
                [
                    [
                        context[-1],
                        context.mean(),
                        context.std(),
                        context[-1] - context[0],
                        np.log1p(len(context)),
                        command_counts[row],
                    ]
                ]
            )
            context_features[row, channel] = features[0]
            predicted[row, channel] = models[channel].predict(features)[0]
            actual[row, channel] = future.mean()
            persistence[row, channel] = context[-1]
    valid = np.isfinite(predicted) & np.isfinite(actual) & np.isfinite(persistence)
    if np.any(valid.sum(axis=1) < channels // 2):
        raise ValueError("raw event forecast has insufficient observed channels")
    model_residual = np.where(valid, actual - predicted, 0.0)
    persistence_residual = np.where(valid, actual - persistence, 0.0)
    pre_event_forecast = np.column_stack(
        (context_features.reshape(count, -1), np.where(valid, predicted, 0.0), ~valid)
    ).astype(np.float64)
    context_only = np.column_stack((context_features.reshape(count, -1), ~valid)).astype(np.float64)
    return (
        np.column_stack((model_residual, np.abs(model_residual), ~valid)).astype(np.float64),
        np.column_stack((persistence_residual, np.abs(persistence_residual), ~valid)).astype(
            np.float64
        ),
        pre_event_forecast,
        context_only,
        {
            "forecast_mae": float(np.abs(model_residual)[valid].mean()),
            "persistence_mae": float(np.abs(persistence_residual)[valid].mean()),
            "observed_fraction": float(valid.mean()),
        },
    )


def _anomaly_pipeline(*, c: float, selected_features: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("nonconstant", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=selected_features)),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    class_weight="balanced",
                    max_iter=5_000,
                    random_state=seed,
                ),
            ),
        ]
    )


@torch.no_grad()
def _pre_event_encoder_features(
    episodes: tuple[EventEpisode, ...],
    config: MirageConfig,
    source: Mission1Source,
    *,
    event_start_hours: float = 6.0,
) -> NDArray[np.float64]:
    """Encode only information timestamped before the labelled event begins."""

    encoder = NeuralJumpCDEEncoder(
        telemetry_channels=len(source.channel_names),
        num_commands=len(source.command_names),
        hidden_dim=int(config.model["hidden_dim"]),
        representation_dim=int(config.model["hidden_dim"]),
        command_embedding_dim=int(config.model["command_embedding_dim"]),
        affected_embedding_dim=int(config.model["affected_embedding_dim"]),
    )
    checkpoint = Path(config.outputs["pretraining"]) / "encoder.pt"
    encoder.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    encoder.eval()
    batch = collate_episodes(episodes)
    sequence_mask = batch.sequence_mask & (batch.timestamps < event_start_hours)
    command_mask = batch.command_mask & (batch.command_times < event_start_hours)
    representations = encoder.encode_pretraining(
        batch.telemetry,
        batch.timestamps,
        batch.observation_mask,
        batch.time_deltas,
        batch.command_ids,
        batch.command_times,
        command_mask,
        sequence_mask,
    )
    return representations.joint.cpu().numpy().astype(np.float64)


def _fit_quiet(model: Pipeline, x: NDArray[np.float64], y: NDArray[np.int64]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        model.fit(x, y)


def _select_anomaly_model(
    features: NDArray[np.float64],
    labels: NDArray[np.int64],
    episodes: tuple[EventEpisode, ...],
    protocol: TemporalProtocol,
    *,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    lookup = {episode.event_id: index for index, episode in enumerate(episodes)}
    folds = []
    for fold in rolling_origin_folds(protocol.outer_train_event_ids, fold_count=3):
        fit = np.asarray([lookup[x] for x in fold.fit_event_ids if x in lookup])
        validation = np.asarray([lookup[x] for x in fold.validation_event_ids if x in lookup])
        if len(fit) and len(validation) and len(np.unique(labels[fit])) == 2:
            folds.append((fit, validation))
    trials: list[dict[str, Any]] = []
    for c in (0.001, 0.01, 0.1, 1.0):
        for selected in (16, 32, 64, 128):
            scores = []
            for fit, validation in folds:
                model = _anomaly_pipeline(c=c, selected_features=selected, seed=seed)
                _fit_quiet(model, features[fit], labels[fit])
                scores.append(
                    float(f1_score(labels[validation], model.predict(features[validation])))
                )
            trials.append(
                {
                    "c": c,
                    "selected_features": selected,
                    "fold_f1": scores,
                    "mean_f1": float(np.mean(scores)),
                    "std_f1": float(np.std(scores)),
                }
            )
    chosen = max(
        trials,
        key=lambda x: (x["mean_f1"], -x["std_f1"], -x["selected_features"], -x["c"]),
    )
    model = _anomaly_pipeline(
        c=float(chosen["c"]), selected_features=int(chosen["selected_features"]), seed=seed
    )
    _fit_quiet(model, features, labels)
    return model, {"selected": chosen, "trials": trials, "test_not_used": True}


def _calibrate_threshold(labels: NDArray[np.int64], probabilities: NDArray[np.float64]) -> float:
    candidates = np.unique(probabilities)
    return float(
        max(
            candidates,
            key=lambda threshold: (
                balanced_accuracy_score(labels, probabilities >= threshold),
                threshold,
            ),
        )
    )


def _anomaly_metrics(
    model: Pipeline,
    calibration_x: NDArray[np.float64],
    calibration_y: NDArray[np.int64],
    test_x: NDArray[np.float64],
    test_y: NDArray[np.int64],
) -> dict[str, float]:
    threshold = _calibrate_threshold(calibration_y, model.predict_proba(calibration_x)[:, 1])
    probability = model.predict_proba(test_x)[:, 1]
    prediction = probability >= threshold
    anomaly = test_y == 1
    nominal = ~anomaly
    return {
        "threshold": threshold,
        "auroc": float(roc_auc_score(test_y, probability)),
        "average_precision": float(average_precision_score(test_y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, prediction)),
        "f1": float(f1_score(test_y, prediction)),
        "rare_nominal_false_positive_rate": float(np.mean(prediction[nominal])),
        "anomaly_miss_rate": float(np.mean(~prediction[anomaly])),
    }


def run_full_data_forecasting(
    config: MirageConfig,
    protocol: TemporalProtocol,
    partitions: dict[str, tuple[EventEpisode, ...]],
) -> FullDataForecastResult:
    """Train, confirm, refit, and evaluate the primary full-corpus forecasting task."""

    store = SSLStore.open(config.ssl_store_dir)
    normalizer = ChannelNormalizer.load(config.ssl_store_dir / "scaler.json")
    context_hours = float(config.pretraining["block_hours"]) - float(
        config.pretraining["forecast_hours"]
    )
    features, targets, persistence, coverage = _block_features(
        store, normalizer, context_hours=context_hours
    )
    if coverage["total_observations"] != store.pretraining_points_total:
        raise AssertionError("forecasting coverage does not equal the full SSL observation count")
    blocks = len(store.block_ids)
    fit_end = int(blocks * 0.80)
    selection_end = int(blocks * 0.90)
    fit = np.arange(fit_end, dtype=np.int64)
    selection = np.arange(fit_end, selection_end, dtype=np.int64)
    confirmation = np.arange(selection_end, blocks, dtype=np.int64)
    alpha_trials = []
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        models = _fit_channel_models(features, targets, fit, alpha=alpha)
        prediction = _predict_blocks(models, features)
        alpha_trials.append(
            {"alpha": alpha, "selection_mae": float(_errors(prediction, targets, selection).mean())}
        )
    selected_alpha = float(min(alpha_trials, key=lambda x: x["selection_mae"])["alpha"])
    development = np.arange(selection_end, dtype=np.int64)
    confirmation_models = _fit_channel_models(features, targets, development, alpha=selected_alpha)
    confirmation_prediction = _predict_blocks(confirmation_models, features)
    model_errors = _errors(confirmation_prediction, targets, confirmation)
    persistence_errors = _errors(persistence, targets, confirmation)
    data_scale = []
    for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 1.0):
        count = max(2, int(len(development) * fraction))
        models = _fit_channel_models(features, targets, development[:count], alpha=selected_alpha)
        errors = _errors(_predict_blocks(models, features), targets, confirmation)
        data_scale.append(
            {
                "training_fraction": fraction,
                "blocks": count,
                "confirmation_mae": float(errors.mean()),
            }
        )
    final_models = _fit_channel_models(
        features, targets, np.arange(blocks, dtype=np.int64), alpha=selected_alpha
    )
    event_features: dict[str, NDArray[np.float64]] = {}
    persistence_features: dict[str, NDArray[np.float64]] = {}
    pre_event_features: dict[str, NDArray[np.float64]] = {}
    context_only_features: dict[str, NDArray[np.float64]] = {}
    event_forecast: dict[str, dict[str, float]] = {}
    event_labels: dict[str, NDArray[np.int64]] = {}
    source = Mission1Source(config.raw_dir)
    for name, episodes in partitions.items():
        (
            event_features[name],
            persistence_features[name],
            pre_event_features[name],
            context_only_features[name],
            event_forecast[name],
        ) = _raw_event_forecast_features(
            episodes,
            final_models,
            source,
            normalizer,
            context_hours=context_hours,
        )
        event_labels[name] = np.asarray(
            [episode.event_type == "Anomaly" for episode in episodes], dtype=np.int64
        )
        pre_event_features[name] = np.column_stack(
            (
                pre_event_features[name],
                _pre_event_encoder_features(episodes, config, source),
            )
        )
    anomaly_model, anomaly_selection = _select_anomaly_model(
        event_features["train"],
        event_labels["train"],
        partitions["train"],
        protocol,
        seed=config.seed,
    )
    persistence_model, persistence_selection = _select_anomaly_model(
        persistence_features["train"],
        event_labels["train"],
        partitions["train"],
        protocol,
        seed=config.seed,
    )
    risk_model, risk_selection = _select_anomaly_model(
        pre_event_features["train"],
        event_labels["train"],
        partitions["train"],
        protocol,
        seed=config.seed,
    )
    context_risk_model, context_risk_selection = _select_anomaly_model(
        context_only_features["train"],
        event_labels["train"],
        partitions["train"],
        protocol,
        seed=config.seed,
    )
    anomaly = _anomaly_metrics(
        anomaly_model,
        event_features["calibration"],
        event_labels["calibration"],
        event_features["test"],
        event_labels["test"],
    )
    anomaly_baseline = _anomaly_metrics(
        persistence_model,
        persistence_features["calibration"],
        event_labels["calibration"],
        persistence_features["test"],
        event_labels["test"],
    )
    pre_event_risk = _anomaly_metrics(
        risk_model,
        pre_event_features["calibration"],
        event_labels["calibration"],
        pre_event_features["test"],
        event_labels["test"],
    )
    context_risk_baseline = _anomaly_metrics(
        context_risk_model,
        context_only_features["calibration"],
        event_labels["calibration"],
        context_only_features["test"],
        event_labels["test"],
    )
    output = Path(config.outputs["root"]) / "forecasting"
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "full_data_forecaster.joblib"
    joblib.dump(
        {
            "channel_models": final_models,
            "pre_event_risk_model": risk_model,
            "pre_event_risk_threshold": pre_event_risk["threshold"],
            "residual_detection_model": anomaly_model,
            "residual_detection_threshold": anomaly["threshold"],
            "feature_order": _FEATURES,
        },
        model_path,
    )
    metrics: dict[str, Any] = {
        "schema_version": "mirage-full-data-forecasting/1.0",
        "status": "primary result",
        "protocol_hash": protocol.protocol_hash,
        "scaler_hash": normalizer.scaler_hash,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "pretrained_encoder_sha256": hashlib.sha256(
            (Path(config.outputs["pretraining"]) / "encoder.pt").read_bytes()
        ).hexdigest(),
        "coverage": coverage | {"blocks": blocks, "coverage_fraction": 1.0},
        "forecast_task": "predict each channel's final-hour mean from the preceding five hours",
        "model_selection": {
            "alpha_trials": alpha_trials,
            "selected_alpha": selected_alpha,
            "fit_blocks": len(fit),
            "selection_blocks": len(selection),
        },
        "confirmation": {
            "blocks": len(confirmation),
            "forecast_mae": float(model_errors.mean()),
            "persistence_mae": float(persistence_errors.mean()),
            "relative_mae_reduction": float(1.0 - model_errors.mean() / persistence_errors.mean()),
            "relative_mae_reduction_bootstrap_95": _bootstrap_improvement(
                model_errors,
                persistence_errors,
                seed=config.seed,
                samples=2_000,
            ),
        },
        "data_scale_ablation": data_scale,
        "pre_event_anomaly_forecasting": pre_event_risk
        | {
            "horizon": "next hour from event start",
            "information_available": (
                "five-hour pre-event context and full-data channel forecasts only"
            ),
            "future_observations_used": False,
            "selection": risk_selection,
            "context_only_baseline": context_risk_baseline | {"selection": context_risk_selection},
            "test_events": len(partitions["test"]),
        },
        "residual_anomaly_detection": anomaly
        | {
            "features": "signed/absolute full-forecast residuals and missingness indicators",
            "selection": anomaly_selection,
            "test_events": len(partitions["test"]),
            "event_forecast_mae": event_forecast["test"],
        },
        "persistence_residual_detection_baseline": anomaly_baseline
        | {"selection": persistence_selection},
    }
    metrics_path = output / "forecasting_results.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return FullDataForecastResult(metrics=metrics, metrics_path=metrics_path, model_path=model_path)
