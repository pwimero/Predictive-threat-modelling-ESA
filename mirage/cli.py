"""Minimal Bash-friendly MIRAGE command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from mirage.config import MirageConfig, load_config
from mirage.pipeline import (
    audit_full_data_pipeline,
    build_temporal_protocol,
    event_cache_dir,
    prepare_event_episodes,
    prepare_ssl_store,
    pretrain_mirage_encoder,
    run_forecasting,
)

app = typer.Typer(
    name="mirage",
    help="Full-corpus ESA telemetry forecasting and anomaly-risk evaluation.",
    no_args_is_help=True,
)
console = Console()


def _ensure_ready(config: MirageConfig, *, device: str = "cpu") -> None:
    protocol = build_temporal_protocol(config)
    if not (config.ssl_store_dir / "manifest.json").exists():
        console.print("Building full-data telemetry store...")
        prepare_ssl_store(config, protocol)
    if not (config.pretraining_dir / "encoder.pt").exists():
        console.print("Pretraining on every eligible telemetry observation...")
        pretrain_mirage_encoder(config, protocol, device=device)
    if not (event_cache_dir(config) / "episodes.json").exists():
        console.print("Preparing chronological event windows...")
        prepare_event_episodes(config, protocol)


@app.command("run")
def run_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config.yaml"),
    device: Annotated[str, typer.Option("--device", help="cpu, mps, or cuda")] = "cpu",
) -> None:
    """Run the complete pipeline and write the primary forecasting report."""

    config = load_config(config_path)
    _ensure_ready(config, device=device)
    result = run_forecasting(config)
    _print_result(result.metrics_path)


@app.command("forecast")
def forecast_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config.yaml"),
) -> None:
    """Run forecasting using existing prepared artifacts."""

    config = load_config(config_path)
    result = run_forecasting(config)
    _print_result(result.metrics_path)


@app.command("audit")
def audit_command(
    config_path: Annotated[Path, typer.Option("--config")] = Path("config.yaml"),
) -> None:
    """Verify full observation coverage and temporal isolation."""

    report = audit_full_data_pipeline(load_config(config_path))
    console.print(f"Audit passed: {report}")


def _print_result(path: Path) -> None:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    confirmation = metrics["confirmation"]
    anomaly = metrics["pre_event_anomaly_forecasting"]
    console.print(f"Results: {path}")
    console.print(
        f"Forecast MAE {confirmation['forecast_mae']:.4f} vs persistence "
        f"{confirmation['persistence_mae']:.4f} "
        f"({confirmation['relative_mae_reduction']:.1%} reduction)"
    )
    console.print(
        f"Pre-event anomaly AUROC {anomaly['auroc']:.3f}; "
        f"average precision {anomaly['average_precision']:.3f}"
    )
    console.print(
        f"Observations used: {metrics['coverage']['total_observations']:,} "
        f"({metrics['coverage']['blocks']:,} blocks)"
    )


if __name__ == "__main__":
    app()
