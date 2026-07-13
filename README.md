
## Run it

Put `ESA-Mission1.zip` in the project root, then run:

```bash
bash scripts/setup.sh
bash scripts/run_mirage.sh run
```

The script creates `.venv` if needed, extracts the data, trains the models, and
writes the results. There is no environment activation step.

Useful follow-up commands:

```bash
bash scripts/run_mirage.sh forecast  # rerun the forecast report
bash scripts/run_mirage.sh audit     # check coverage and time splits
bash scripts/verify.sh               # lint, type-check, and test
```

## What it does

1. Reads all usable safe-period telemetry.
2. Pretrains a small command-aware Neural Jump CDE with self-supervised tasks.
3. Fits one Ridge forecaster per telemetry channel.
4. Uses the forecasts and the frozen encoder to rank risk before labelled events.

The current run covers 353,471,368 observations across 11,351 time blocks.
The channel forecaster has MAE `0.0235`, compared with `0.0995` for persistence
(a 76.4% reduction). The pre-event risk ranking has AUROC `0.810` and average
precision `0.863` on the held-out event period.

## Results

The useful files are in `outputs/`:

```text
outputs/
├── forecasting/forecasting_results.json
├── forecasting/full_data_forecaster.joblib
├── pretraining/encoder.pt
├── pretraining/coverage_report.json
├── pretraining/pretraining_manifest.json
└── audit/audit_report.json
```

`forecasting_results.json` is the main report. It includes coverage, the
chronological test result, a data-size check, and the pre-event anomaly scores.

## Data

via the official [ESA Anomaly Dataset](https://zenodo.org/records/12528696)
and save the Mission 1 archive as `ESA-Mission1.zip` in this folder.

## Layout

```text
mirage/       Python pipeline
scripts/      Bash entry points
outputs/      saved results and model files
data/         local dataset (not committed)
```