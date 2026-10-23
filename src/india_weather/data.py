"""Lesson 2: download, audit and clean actual NOAA GHCNh observations.

Run from the project directory: python -m india_weather.data
Raw PSV reports and provenance are preserved separately from Open-Meteo data.
The hourly product uses exact UTC-hour reports only, without averaging future
half-hour reports into an earlier forecast origin. Missing labels stay missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(os.environ.get('INDIA_WEATHER_ROOT', Path(__file__).resolve().parents[2])).resolve()
BASE = 'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/'
STATIONS = {'Delhi': 'INI0000VIDD', 'Mumbai': 'INI0000VABB', 'Bengaluru': 'INM00043296'}
# Values in these PSV files are already decimal Celsius, m/s, degrees, hPa, mm.
# Never divide temperature by 10 merely because the documentation says tenths.
BOUNDS = {
    'temperature': (-90, 60), 'dew_point_temperature': (-100, 60),
    'relative_humidity': (0, 100), 'wind_speed': (0, 120),
    'wind_direction': (0, 360), 'wind_gust': (0, 150),
    'sea_level_pressure': (850, 1100), 'station_level_pressure': (500, 1100),
    'precipitation': (0, 1000), 'precipitation_3_hour': (0, 2000),
    'precipitation_24_hour': (0, 3000),
}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def download(station, year, refresh=False):
    """Cache public files; record the exact bytes used in this experiment."""
    path = ROOT / 'data/raw' / f'GHCNh_{station}_{year}.psv'
    if path.exists() and not refresh:
        return path
    url = BASE + f'hourly/access/by-year/{year}/psv/{path.name}'
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retry))
    with session:
        response = session.get(url, timeout=(10, 60))
        response.raise_for_status()
    if not response.content.startswith(b'STATION|'):
        raise ValueError(f'Unexpected NOAA schema: {url}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(response.content)
    temporary.replace(path)
    write_json(path.with_suffix('.metadata.json'), {
        'url': url, 'retrieved_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'sha256': hashlib.sha256(response.content).hexdigest(),
        'bytes': len(response.content), 'last_modified': response.headers.get('Last-Modified'),
    })
    return path


def read_raw(path):
    """Keep flag strings and blanks intact, including zero-valued QC codes."""
    frame = pd.read_csv(path, sep='|', dtype=str, keep_default_na=False)
    required = ['STATION', 'DATE']
    for variable in BOUNDS:
        required.extend([variable, variable+'_Quality_Code', variable+'_Source_Code',
                         variable+'_Measurement_Code'])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing NOAA columns: {sorted(missing)}')
    return frame


def qc_accepted(flags, sources):
    """Explicit source-aware conservative policy (GHCNh manual section VI).

    Blank means no flag supplied, NOT certified correct. Keep it but audit it.
    Modern alphabetic flags are excluded. Source 223 uses 5=removed, while
    legacy ISD sources use 5=passed. Unknown nonblank codes are quarantined.
    """
    flags, sources = flags.str.strip(), sources.str.strip()
    accepted = flags.eq('')
    isd = sources.isin(['313', '314', '315', '322', '335', '343', '344', '346'])
    legacy = sources.isin(['220', '221', '222', '223', '347', '348'])
    accepted |= isd & flags.isin(['0', '1', '4', '5', '9'])
    accepted |= legacy & flags.isin(['0', '1', '4'])
    # 412/413 are newer BUFR/METAR streams. Nonblank unknown flags are rejected.
    return accepted


def clean_reports(raw, station):
    """Return unfilled hourly measurements plus an auditable cleaning report."""
    if set(raw.STATION) != {station}:
        raise ValueError('A file must contain exactly the selected station')
    times = pd.to_datetime(raw.DATE, utc=True, errors='coerce')
    valid_time = times.notna()
    selected = raw.loc[valid_time].copy()
    selected.index = pd.DatetimeIndex(times[valid_time], name='time_utc')
    duplicates = int(selected.index.duplicated().sum())
    # Conflicting duplicate reports need a documented source priority, not an
    # arbitrary keep-first/keep-last decision. Stop and inspect if encountered.
    if duplicates:
        raise ValueError(f'{duplicates} duplicate timestamps; inspect before continuing')
    selected = selected.sort_index()
    cleaned = pd.DataFrame(index=selected.index)
    audit = []
    for variable, (low, high) in BOUNDS.items():
        numeric = pd.to_numeric(selected[variable], errors='coerce').replace([np.inf, -np.inf], np.nan)
        good_qc = qc_accepted(selected[variable+'_Quality_Code'], selected[variable+'_Source_Code'])
        in_range = numeric.between(low, high)
        measurement = selected[variable+'_Measurement_Code'].str.strip()
        good_measurement = pd.Series(True, index=selected.index)
        if variable.startswith('precipitation'):
            # Trace is not exactly zero; accumulated/uncertain codes are retained
            # only in the raw data. Do not manufacture hourly rain totals.
            good_measurement = measurement.eq('')
        elif variable == 'wind_direction':
            good_measurement = ~measurement.eq('V')  # variable direction has no fixed bearing
        cleaned[variable] = numeric.where(good_qc & in_range & good_measurement)
        audit.append({'variable': variable, 'raw_rows': len(selected),
            'raw_missing': int(selected[variable].eq('').sum()),
            'invalid_numeric': int((selected[variable].ne('') & numeric.isna()).sum()),
            'qc_rejected_nonmissing': int((numeric.notna() & ~good_qc).sum()),
            'range_rejected_nonmissing': int((numeric.notna() & ~in_range).sum()),
            'measurement_rejected_nonmissing': int((numeric.notna() & ~good_measurement).sum()),
            'qc_counts': selected.loc[numeric.notna(), variable+'_Quality_Code'].value_counts().to_dict(),
            'source_counts': selected.loc[numeric.notna(), variable+'_Source_Code'].value_counts().to_dict()})
    # Exact hour samples make shift(24) mean precisely 24 hours, even with gaps.
    exact = cleaned.index == cleaned.index.floor('h')
    hourly = cleaned.loc[exact]
    if hourly.empty:
        raise ValueError('No exact-hour observations; choose a sampling policy explicitly')
    grid = pd.date_range(cleaned.index.min().ceil('h'), cleaned.index.max().floor('h'), freq='h', name='time_utc')
    hourly = hourly.reindex(grid)
    for row in audit:
        row['hourly_missing_percent'] = round(100 * float(hourly[row['variable']].isna().mean()), 2)
    metadata = {'station_id': station, 'station_name': raw.Station_name.iloc[0],
        'raw_rows': len(raw), 'raw_columns': len(raw.columns),
        'invalid_timestamps': int((~valid_time).sum()), 'duplicate_timestamps': duplicates,
        'subhourly_reports_excluded': int((~exact).sum()), 'hourly_rows': len(hourly),
        'first_utc': hourly.index.min().isoformat(), 'last_utc': hourly.index.max().isoformat(),
        'last_report_utc': cleaned.index.max().isoformat(),
        'observed_temperature_hours': int(hourly.temperature.notna().sum()),
        'temperature_missing_percent': round(100*float(hourly.temperature.isna().mean()), 2),
        'sampling': 'Exact UTC-hour reports only; no target imputation or subhourly binning',
        'variables': audit}
    return hourly, metadata


def prepare(year=2026, refresh=False):
    summaries = []
    out = ROOT / 'reports/data_quality'
    out.mkdir(parents=True, exist_ok=True)
    for city, station in STATIONS.items():
        path = download(station, year, refresh)
        hourly, audit = clean_reports(read_raw(path), station)
        audit['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        audit['source_url'] = BASE+f'hourly/access/by-year/{year}/psv/{path.name}'
        dest = ROOT / 'data/processed' / f'{city}_{year}_hourly.csv'
        dest.parent.mkdir(parents=True, exist_ok=True)
        hourly.to_csv(dest)
        write_json(out / f'{city}_{year}_audit.json', audit)
        pd.DataFrame(audit['variables']).to_csv(out / f'{city}_{year}_missingness.csv', index=False)
        summaries.append({'city': city, **{k: audit[k] for k in ['station_id', 'station_name',
            'raw_rows', 'hourly_rows', 'observed_temperature_hours', 'temperature_missing_percent', 'last_utc']}})
        print(f'{city}: {len(hourly):,} hourly slots, '
              f'{audit["temperature_missing_percent"]}% missing temperature', flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / f'coverage_{year}.csv', index=False)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    print(prepare(args.year, args.refresh).to_string(index=False))
