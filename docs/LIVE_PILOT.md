# Live pilot: forecasts, original correction training and deployment

## What is working

- Live ECMWF IFS 0.25° forecasts through Open-Meteo for three supported stations.
- Browser dashboard: location/date selection, temperature/rain/wind charts, CSV
  download, source/retrieval labels, validation and method views.
- Original correction models: Delhi boosted trees; Mumbai and Bengaluru ridge.
- A separate nominal 24h temperature panel using the provider-aligned day1 stream,
  with exported model inference verified against Python.
- FastAPI `/forecast`, `/predict`, `/healthz`; Streamlit frontend; Docker Compose.
- 10-minute per-process cache, one bounded retry, connection/read timeouts,
  labelled stale fallback up to 6h, and unavailable state beyond that limit.

The public browser frontend calls public weather APIs and executes exported
numerical model coefficients/trees. The Python backend is a separately runnable
application; publishing the browser frontend does not host a Python API endpoint.

## Original training and evaluation

Training observations are NOAA GHCNh station reports for 2024, 2025 and 2026, after
source-aware QC and exact-hour sampling. Missing labels are never filled. Inputs
are archived ECMWF `ecmwf_ifs025` forecasts from Open-Meteo's Previous Runs API,
using `_previous_day1`. Source URLs, retrieval times and input fingerprints are
preserved in `data/provenance/`.

The target is `observed temperature − raw forecast temperature`. Each station's
correction is selected among identity/raw, median bias, ridge regression and
histogram gradient boosting. Three earlier expanding-history folds compare MAE;
the best correction needs a fixed 3% improvement over raw validation MAE to be used.
No choice is made using the exposed final-week scores.

Final model training labels end before **5 October 2026 00:00 IST**. A separate
calibration segment has origins on/after 5 October and targets before 19 October.
Weights remain frozen. Absolute calibration errors estimate an empirical 90%
band with a finite-sample quantile adjustment. Weather dependence/drift means
coverage is not guaranteed; the small final test is descriptive only.

Test origins begin **19 October 00:00 IST**, with +24h valid times. The downloaded
station archive ends 21 October. Scoring requires measured temperature at both
origin and target so persistence and corrections use identical pairs.

| City | Training rows | Calibration rows | Test pairs | Raw MAE | Corrected MAE | Persistence MAE |
|---|---:|---:|---:|---:|---:|---:|
| Delhi | 8,942 | 303 | 39 | 1.451°C | 1.097°C | **0.467°C** |
| Mumbai | 22,855 | 312 | 43 | 2.200°C | 0.922°C | **0.628°C** |
| Bengaluru | 22,184 | 309 | 43 | 1.505°C | **0.624°C** | 1.000°C |

The original October test was already exposed during earlier project work. The
new experiment preserves it for continuity but is exploratory; a fresh prospective
holdout is needed before a confirmatory claim. Baselines, failures and the limited
weather regimes remain visible.

## Feature semantics and availability

Inputs include forecast temperature, humidity, dew point, pressure, wind vectors,
precipitation, cloud cover and shortwave radiation. Rolling means/std, temperature
lags and deltas describe the **archived forecast trajectory**, not station-measured
history. Daily/yearly cycles use sine/cosine in IST.

The provider-aligned day1 endpoint does not supply original run initialization
or dissemination timestamps. Earlier archive rows are assumed available according
to the provider's offset semantics; exact historical arrival times are unverified.
For issuance-exact work use individually archived runs and record actual receipt
times. NOAA observations and historical QC may also be revised later.

The hourly chart uses the latest provider stream. The correction panel uses its
matching day1 stream. They can differ because of lead time and run selection.
Do not apply the correction to arbitrary leads, another provider model or measured
history substituted for forecast predictors. The nominal target is 24h after the
current UTC-hour boundary, so the remaining lead at fetch time can be slightly less.

No dedicated ENSO index, Indian Ocean Dipole index, cyclone advisory, radar or
satellite feed is integrated into our correction. Weather-model inputs can reflect
some broad atmospheric influences; this is not equivalent to an explicit, verified
feature integration. Provider rain/wind displays are not independently trained
local rain/wind models. Three stations do not establish India-wide performance.

## Reproduce

```bash
python -m pip install -r requirements-app.lock.txt
python -m pip install -e .
python -m india_weather.correction --download --week 2026-10-19
python scripts/export_dashboard.py --snapshot
python -m unittest discover -s tests -v
node dashboard/scripts/test-portable.mjs
```

Snapshots are a failure fallback only: they expire after 6h and are visibly
labelled when used. No synthetic weather is used in the training data or live UI.
Test fixtures use synthetic inputs only for bounded error-handling checks.

## Deployment and remaining validation

`docker compose up --build` launches one FastAPI worker and a separate Streamlit
container. This pilot uses a per-process cache; multiple replicas need a shared
cache, quotas and concurrent load testing. Model JSON avoids joblib execution in
the public browser; the Python training artifacts remain trusted-input-only.
The model root can be set with `INDIA_WEATHER_ROOT`, as configured in Docker.

Open-Meteo free access is for non-commercial use and has rate limits. This project
remains a personal research pilot. A commercial launch needs suitable service
access, operational monitoring and a demonstrated reliability target. Data licence,
API-service terms and the project's still-undecided code licence are distinct.

Docker configuration is supplied; report a Docker build as verified only after
running it on a machine with Docker installed. Passing local tests is not proof
of a hosted GitHub Actions run, load-test result or multi-season forecast skill.
