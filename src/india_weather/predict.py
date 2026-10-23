"""Replay a saved model using a measured origin from the local hourly dataset.

This is offline inference, not a current-weather API. Only load artifacts from
sources you trust: joblib uses pickle. A checksum verifies integrity, not safety.
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from .artifacts import sha256
from .data import ROOT, STATIONS
from .model import load_hourly, make_features, predict


def forecast(city, origin, week='2026-10-19', year=2026):
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        raise ValueError('Specify the origin timezone, for example 2026-10-20T12:00:00Z')
    origin = origin.tz_convert('UTC')
    folder = ROOT/'models'/f'week_{week}'
    path = folder/f'{city}.joblib'
    manifest = json.loads((folder/'manifest.json').read_text())
    if sha256(path) != manifest['models'][path.name]['sha256']:
        raise ValueError('Model checksum does not match its manifest')
    if sklearn.__version__ != manifest['environment']['scikit_learn']:
        raise ValueError('Use the scikit-learn version recorded in the model manifest')
    artifact = joblib.load(path)
    if origin < pd.Timestamp(artifact['report']['train_labels_before_ist']):
        raise ValueError('Origin precedes the model\'s simulated availability')
    hourly = load_hourly(city, year).loc[:origin]
    if len(hourly)<169 or hourly.index[-1] != origin or pd.isna(hourly.temperature.iloc[-1]):
        raise ValueError('Need an observed exact-hour origin and at least one week of history')
    X = make_features(hourly).loc[[origin], artifact['feature_names']]
    value = float(predict(artifact['model'], X, hourly.temperature.loc[[origin]])[0])
    return {'city': city, 'station_id': STATIONS[city], 'origin_utc': origin.isoformat(),
            'valid_utc': (origin+pd.Timedelta(hours=24)).isoformat(), 'temperature_c': value,
            'model_sha256': manifest['models'][path.name]['sha256'],
            'mode': 'retrospective_station_replay', 'prediction_interval': None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', choices=STATIONS, required=True)
    parser.add_argument('--origin', required=True, help='Timezone-aware exact UTC-hour timestamp')
    parser.add_argument('--week', default='2026-10-19')
    parser.add_argument('--year', type=int, default=2026)
    args = parser.parse_args()
    print(json.dumps(forecast(args.city, args.origin, args.week, args.year), indent=2))
