# Deployment status

The browser dashboard was deployed successfully on
24 October 2026 IST:

https://india-weather-kirti-lab.sid880088.chatgpt.site

**Current audience: public.** The owner explicitly approved public dashboard access
after reviewing the separate website-sharing request. Anyone with the URL can
use the dashboard and inspect its published numerical model artifacts.

The published browser source is the sibling `weather-dashboard/` project, at
commit `c7b69ccbb3d4b31950545a8b88765ebac1b68a24`. A sanitized copy of the browser
source is included in `dashboard/` in this Python repository. The hosted frontend
calls public weather APIs and applies the exported, original Python-trained
temperature corrections in the browser. It does not require a hosted Python backend.

FastAPI and Streamlit have been implemented and tested locally. Their Docker
configuration is supplied, but Docker is not installed on the development host,
so no successful image-build or container-start claim is made.

## Verification

- 24 Python checks pass, covering chronological evaluation, artifact consistency,
  corrupt inputs, missing values, bounded stale-cache fallback and API errors.
- Twelve real-data fixtures verify 552 browser/Python feature comparisons and
  every exported correction prediction.
- All three city live service forecasts were verified.
- Browser checks covered live values, city switching, validation tables, future
  date selection, rainfall/wind charts and CSV download.
- HTTP smoke checks are recorded in `reports/live_http_check.json`. These are
  individual local requests, not an operational load test.

The public-data provider has non-commercial service terms/rate limits. The pilot
does not establish company-level reliability, India-wide forecast skill or
guaranteed uncertainty coverage. See `LIVE_PILOT.md` for scope and limitations.
