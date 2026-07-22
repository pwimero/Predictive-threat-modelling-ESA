
## Run it

Put `ESA-Mission1.zip` in the project root, then run:

```bash
bash scripts/setup.sh
bash scripts/run_mirage.sh run
```

The script creates `.venv` if needed, extracts the data, trains the models, and
writes the results. There is no environment activation step. macOS users can
store the archive and/or the `data` directory with a `.nosync` suffix (for
example `ESA-Mission1.zip.nosync.zip` and `data.nosync/`); MIRAGE detects these
local-only alternatives automatically while retaining portable repository paths.

Useful follow-up commands:

```bash
bash scripts/run_mirage.sh forecast  # rerun the forecast report
bash scripts/run_mirage.sh audit     # check coverage and time splits
bash scripts/verify.sh               # compile, smoke-check, and test
```

## What it does

1. Reads all usable safe-period telemetry.
2. Pretrains a small command-aware Neural Jump CDE with self-supervised tasks.
3. Fits one Ridge forecaster per telemetry channel.
4. Uses the forecasts and the frozen encoder to rank risk before labelled events.

The current run covers 353,471,368 observations across 11,351 time blocks.
The channel forecaster has MAE `0.0235`, compared with `0.0995` for persistence
(a 76.4% reduction). The pre-event risk ranking has AUROC `0.810` and average
precision `0.863` on 33 held-out event windows. These are prototype results:
the selected threshold misses 77.3% of anomalies, so the system is not yet an
operational alerting tool.

## Purpose and relation to the PhD proposal

MIRAGE-M1 is a temporal prototype for the predictive
perception layer of the proposed Executable Threat Twin. It shows how
spacecraft telemetry, command history and uncertainty-aware evaluation can be
used to establish an evidence base before a disruptive event.

It is narrower than the proposed PhD system. It does not yet
model a causal constellation graph, generate multi-stage cyber/physical threat
paths, simulate mission-loss propagation, or co-design autonomous recovery
policies. Those capabilities are the intended research extension: MIRAGE-M1
provides the telemetry forecasting and pre-event risk component on which they
can be built.

## What the current results mean

- The forecasting result is strong: the model reduces next-hour telemetry error
  by 76.4% relative to a persistence baseline on a chronological confirmation
  period. This supports its use as a compact predictive model of normal
  spacecraft behaviour.
- The pre-event risk model shows useful ranking ability (AUROC 0.810; average
  precision 0.863) on 33 event windows. This means that it can
  often place anomalous windows above nominal ones before the event begins.
- As it currently exists the results arent good enough to be a functional syste, At its calibrated threshold it
  misses 77.3% of anomalous windows.
- The audit records complete data coverage and a chronology-respecting split,
  so the reported results have explicit provenance and leakgage robustness

## Saved results

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
