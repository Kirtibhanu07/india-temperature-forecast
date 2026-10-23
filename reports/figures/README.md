# Figures for GitHub and research discussions

Regenerate with `python -m india_weather.plots`. Figures use committed result and
audit tables, require no network access, and do not train a model. Both PNG and
SVG versions are exported under `week_2026-10-19/`.

1. `01_data_quality`: missing hourly measurements by station/variable.
2. `02_validation_timeline`: earlier training labels, +24h lead and target windows.
3. `03_cv_mae`: every pre-test validation fold versus persistence.
4. `04_test_mae_rmse`: held-out matched errors for ML and persistence.
5. `05_forecast_vs_observed`: all available held-out measured/predicted pairs.
6. `06_error_distribution`: signed errors, exposing warm/cold bias.

Use PNG for GitHub README embeds and SVG for scalable graphics. Captions retain
the partial-week sample size, missingness and retrospective limitations. These
figures do not establish confidence intervals, significance or nationwide skill.
