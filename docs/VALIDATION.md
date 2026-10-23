# Validation protocol

## Prediction task and cutoff

At a measured exact UTC-hour origin, predict the same station's temperature
24 hours later. IST displays consequently fall on half-hours. Labels are never
imputed. A score requires both measured endpoints, after 168h feature warm-up.

The final test is the calendar week starting Monday **19 October 2026 00:00 IST**.
Training **target timestamps**, not merely training input timestamps, must precede
that boundary. Test origins must be on/after the boundary. This prevents training
on Monday answers and prevents using a Monday-fitted model for Sunday forecasts.

First observed test target: **Tuesday 20 October 00:30 IST**.
Last observed test target: **Wednesday 21 October 18:30 IST**.
Unknown future outcomes, missing answers, and Monday targets requiring pre-fit
Sunday origins are excluded. The final weights remain frozen during the test.

## Earlier cross-validation

| Fold | Validation origins start, IST | Target cutoff, exclusive, IST |
|---|---|---|
| 1 | 7 September 00:00 | 21 September 00:00 |
| 2 | 21 September 00:00 | 5 October 00:00 |
| 3 | 5 October 00:00 | 19 October 00:00 |

Each fold expands its training history and refits its imputer/model using labels
strictly before the first validation origin. The +24h lead means eligible target
times start one day after the origin boundary. All eligible fold scores are
included in `reports/metrics/week_2026-10-19/cross_validation.csv`.

## Metrics and baseline

Persistence forecast: measured origin temperature. Model and persistence are scored
on exactly the same pairs. Record MAE, RMSE, mean signed bias, R² and fraction
within ±2°C. R² is not percentage accuracy; tolerance coverage must name its units
and threshold. There is no defensible universal conversion from RMSE to “95% accuracy.”

| City | Training pairs | Test pairs | ML MAE °C | Baseline MAE °C | ML RMSE °C |
|---|---:|---:|---:|---:|---:|
| Delhi | 3,596 | 39 | 0.727 | **0.467** | 0.841 |
| Mumbai | 6,717 | 43 | 0.686 | **0.628** | 0.854 |
| Bengaluru | 6,332 | 43 | 1.035 | **1.000** | 1.261 |

Persistence wins on MAE in all three cities. ML has lower RMSE in Mumbai and
Bengaluru: 0.854 vs 0.902°C and 1.261 vs 1.422°C respectively. In Delhi ML RMSE
is 0.841°C versus persistence's 0.801°C. Retain this distinction when presenting
the results; a lower MAE and a lower RMSE are different criteria.

## Scientific checks

Offline tests cover source-specific QC meanings, rejection of unknown flags,
future-input perturbation invariance, actual-endpoint deltas, wind vectors/calm
handling, explicit missing hourly rows, subhourly lookahead prevention, duplicate
conflict rejection, chronological CV purging, IST week/year boundaries, exclusion
of missing/future outcomes, saved result/metric consistency, artifact checksums,
and prediction replay against saved rows.

GitHub Actions is configured to run offline tests and regenerate figures on push
and pull request. A configured workflow is not evidence that a hosted run passed;
inspect the Actions result after publication.

## Limits on interpretation

This is a small retrospective test with serially dependent pairs, incomplete
station coverage, less than one annual cycle and unknown original report arrival
times. Archive QC/revisions may use later information. No independently verified
real-time replay or calibrated interval-coverage experiment has been completed.

The test is exposed and must not become a tuning target. Use earlier validation
to define the next experiment, then reserve fresh test periods across seasons and
weather regimes. Retain failed improvements and compare against the same baseline.
