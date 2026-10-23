# Saved research models

`week_2026-10-19/` contains three locally trained scikit-learn pipelines, one per
station, plus a machine-readable manifest. Each file is approximately 360 KB.
See `docs/MODEL_CARD.md` and the main README for the prediction rule and replay CLI.

The model trails persistence on MAE in the exposed partial-week test. It is included to
make the experiment inspectable, not as an operationally validated weather model.
Only load trusted joblib artifacts, using the recorded environment versions.
