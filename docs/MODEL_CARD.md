# Model card

## Intended use

Educational and personal research on 24-hour station temperature forecasting for
three Indian locations. This initial model has higher MAE than persistence in all
three cities on the exposed held-out week. It is not a validated public weather
service, city-wide forecast or emergency-warning model.

## Algorithm and fitted objects

One scikit-learn 1.7.2 `HistGradientBoostingRegressor` per city, inside a pipeline
with `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)`.
Imputation statistics are learned inside each fold/final fit. Entirely missing
features receive the imputer's constant empty-column placeholder, not a measured
weather value. Tree inputs do not require standardization or robust scaling.

Parameters: 150 boosting iterations, at most 15 leaves, learning rate 0.05,
L2 regularization 10, early stopping disabled, random seed 42. The default
squared-error objective learns temperature change. Parameters were fixed for
this starting experiment; no test-driven hyperparameter search is claimed.

```text
label = measured temperature(t+24h) − measured temperature(t)
forecast = measured temperature(t) + pipeline.predict(features(t))
```

## Features

- Current scalar temperature, dew point, humidity, wind speed, station pressure,
  sea-level pressure; missingness indicators; at most 2h past-only carry-forward.
- Actual temperature lags: 1, 3, 6, 24, 48, 72, 168h.
- Temperature means, sample standard deviations and observed counts: 3, 6, 24, 72h.
- Temperature endpoint changes and average rates: 1, 3, 24h. Missing endpoints
  yield missing features, not invented changes.
- Meteorological wind components: `u=-speed*sin(direction)`, `v=-speed*cos(direction)`.
  Calm zero-speed vectors are zero even when direction is unavailable.
- Wind-component, humidity and pressure means/std over 6 and 24h.
- Sine/cosine of IST hour and fractional year, accounting for leap years.

No input explicitly includes rain, gusts, cyclone state, ENSO or numerical forecast
output. The first 168 hours are excluded from supervised training/evaluation as
feature warm-up. Origin and target temperatures must both be measured.

## Artifacts and inference

`models/week_2026-10-19/{Delhi,Mumbai,Bengaluru}.joblib` contain the fitted pipeline,
feature order, experiment report and prediction rule. `manifest.json` records
checksums, environment versions, estimator settings and training-input fingerprints.
`python -m india_weather.predict` verifies artifact checksum and scikit-learn
version, rejects origins before simulated model availability, and builds features
from local data through the requested origin only. It requires measured origins
with at least a week of history. No live API integration or intervals are provided.

Only load artifacts from a trusted source. SHA-256 detects byte changes but is not
proof of a publisher's identity or artifact safety. See the
[scikit-learn persistence guidance](https://scikit-learn.org/1.7/model_persistence.html).

## Evaluation and limitations

See [validation](VALIDATION.md) and committed metrics. All three models trail
persistence on MAE; ML has lower RMSE in Mumbai and Bengaluru on the short test.
Delhi data are particularly incomplete. Errors
are temporally dependent, and no statistical superiority claim is supported.
The retrospective archive lacks verified original arrival times and may include
later QC/revisions. There is no calibrated uncertainty, spatial holdout study,
multi-season evaluation, drift monitoring or latency validation yet.

Adding standard deviations or more predictors does not itself establish better
forecast skill. Ablations and parameter choices need separate validation, then
a fresh held-out period after this exposed test.
