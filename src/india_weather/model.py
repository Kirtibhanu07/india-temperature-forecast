"""Lesson 2: a 24-hour station temperature model with chronological validation.

We learn tomorrow's change from today's temperature (persistence + correction).
All labels are actual, unfilled station reports. Test-week scores are descriptive
and must not be used to tune this same model. Archive QC/revisions can use later
information: this is retrospective evaluation, not proof of live performance.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .data import ROOT, STATIONS, write_json
from .evaluation import split_week, score, week_bounds
from .artifacts import save_manifest


def load_hourly(city, year=2026):
    path = ROOT / 'data/processed' / f'{city}_{year}_hourly.csv'
    data = pd.read_csv(path, index_col='time_utc', parse_dates=True)
    data.index = pd.to_datetime(data.index, utc=True)
    if data.index.has_duplicates or not data.index.equals(pd.date_range(data.index.min(), data.index.max(), freq='h', name=data.index.name)):
        raise ValueError('Expected unique, sorted, complete hourly grid including missing rows')
    return data


def make_features(hourly):
    """Causal predictors; no backward filling and no imputation of outcomes.

    Temperature differences require both actual endpoint measurements. Rolling
    summaries skip missing measurements and include counts, so a sparse window
    is distinguishable from an entirely observed one. Scalar predictors may use
    up to two past hours; training medians handle remaining gaps within each fit.
    """
    names = ['temperature', 'dew_point_temperature', 'relative_humidity',
             'wind_speed', 'sea_level_pressure', 'station_level_pressure']
    X = pd.DataFrame(index=hourly.index)
    for name in names:
        value = hourly[name]
        X[name+'_missing'] = value.isna().astype(int)
        X['now_'+name] = value.ffill(limit=2)
    temperature = hourly.temperature
    for lag in [1, 3, 6, 24, 48, 72, 168]:
        X[f'temp_lag_{lag}h'] = temperature.shift(lag)
    for window in [3, 6, 24, 72]:
        roll = temperature.rolling(window, min_periods=2)
        X[f'temp_mean_{window}h'] = roll.mean()
        X[f'temp_std_{window}h'] = roll.std()
        X[f'temp_count_{window}h'] = temperature.rolling(window, min_periods=1).count()
    for delta in [1, 3, 24]:
        X[f'temp_delta_{delta}h'] = temperature-temperature.shift(delta)
        X[f'temp_rate_{delta}h'] = X[f'temp_delta_{delta}h']/delta
    # Meteorological bearing is where the wind comes FROM. Never average raw
    # directions: 359 and 1 degrees average to north, not south.
    speed, radians = hourly.wind_speed, np.deg2rad(hourly.wind_direction)
    X['wind_u_ms'] = (-speed*np.sin(radians)).mask(speed.eq(0), 0)
    X['wind_v_ms'] = (-speed*np.cos(radians)).mask(speed.eq(0), 0)
    for name in ['wind_u_ms', 'wind_v_ms', 'now_relative_humidity', 'now_sea_level_pressure', 'now_station_level_pressure']:
        for window in [6, 24]:
            X[f'{name}_mean_{window}h'] = X[name].rolling(window, min_periods=2).mean()
            X[f'{name}_std_{window}h'] = X[name].rolling(window, min_periods=2).std()
    local = hourly.index.tz_convert('Asia/Kolkata')
    hour = local.hour+local.minute/60
    year_fraction = (local.dayofyear-1+hour/24)/np.where(local.is_leap_year, 366, 365)
    X['hour_sin'], X['hour_cos'] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    X['year_sin'], X['year_cos'] = np.sin(2*np.pi*year_fraction), np.cos(2*np.pi*year_fraction)
    # Sparse precipitation and gust reports remain in the clean dataset for EDA.
    # Their accumulation/reporting semantics need audit before adding predictors.
    return X.replace([np.inf, -np.inf], np.nan)


def new_model():
    # Tree splits do not need scaling. Each fit learns its own imputation values.
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
        HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15,
            learning_rate=.05, l2_regularization=10, early_stopping=False, random_state=42))


def predict(model, X, current):
    return current.to_numpy()+model.predict(X)


def validation_folds(index, known, test_start):
    """Three pre-test, fortnight-long folds with label-time purge at every fit."""
    target_times = index+pd.Timedelta(hours=24)
    for fold in range(3):
        start = test_start-pd.Timedelta(days=42-14*fold)
        end = start+pd.Timedelta(days=14)
        train = known & (target_times<start)
        valid = known & (index>=start) & (target_times<end)
        yield fold+1, train, valid, start, end


def run(year=2026, week_start=None):
    as_of = pd.Timestamp.now(tz='UTC')
    start, _ = week_bounds(as_of, week_start)
    out = ROOT / 'reports/metrics' / f'week_{start.date()}'
    out.mkdir(parents=True, exist_ok=True)
    model_out = ROOT / 'models' / f'week_{start.date()}'
    model_out.mkdir(parents=True, exist_ok=True)
    summaries, all_predictions, cv_rows = [], [], []
    for city, station in STATIONS.items():
        hourly = load_hourly(city, year).loc[:as_of.floor('h')]
        X = make_features(hourly)
        y = hourly.temperature.shift(-24)  # unfilled measured target, precisely +24h
        known = y.notna() & hourly.temperature.notna()
        known.iloc[:168] = False
        target_times = hourly.index+pd.Timedelta(hours=24)
        train, test, _, _ = split_week(hourly.index, known, as_of, week_start)
        if train.sum()<1000 or test.sum()<2:
            raise ValueError(f'{city}: insufficient observed train/test pairs for this week')
        for fold, fit, valid, fold_start, fold_end in validation_folds(hourly.index, known, start):
            if fit.sum()<500 or valid.sum()<2:
                continue
            model = new_model()
            model.fit(X.loc[fit], y.loc[fit]-hourly.temperature.loc[fit])
            pred = predict(model, X.loc[valid], hourly.temperature.loc[valid])
            cv_rows.append({'city': city, 'fold': fold, 'validation_origin_start_ist': fold_start.isoformat(),
                'validation_target_end_exclusive_ist': fold_end.isoformat(),
                'fit_pairs': int(fit.sum()), 'validation_pairs': int(valid.sum()),
                **score(y.loc[valid].to_numpy(), pred),
                'persistence_mae_c': score(y.loc[valid].to_numpy(), hourly.temperature.loc[valid].to_numpy())['mae_c']})
        model = new_model()
        model.fit(X.loc[train], y.loc[train]-hourly.temperature.loc[train])
        pred = predict(model, X.loc[test], hourly.temperature.loc[test])
        actual, baseline = y.loc[test].to_numpy(), hourly.temperature.loc[test].to_numpy()
        model_metrics, baseline_metrics = score(actual, pred), score(actual, baseline)
        table = pd.DataFrame({'city': city, 'station_id': station,
            'origin_utc': hourly.index[test], 'valid_utc': target_times[test],
            'actual_station_c': actual, 'prediction_c': pred, 'persistence_c': baseline,
            'error_c': pred-actual, 'absolute_error_c': abs(pred-actual)})
        table.to_csv(out / f'{city}_predictions.csv', index=False)
        audit = json.loads((ROOT/'reports/data_quality'/f'{city}_{year}_audit.json').read_text())
        report = {'city': city, 'station_id': station, 'horizon_hours': 24,
            'target_type': 'actual_unfilled_station_temperature',
            'input_sha256': audit['input_sha256'], 'run_at_utc': as_of.isoformat(),
            'train_labels_before_ist': start.isoformat(),
            'last_training_target_utc': target_times[train][-1].isoformat(),
            'first_test_origin_utc': hourly.index[test][0].isoformat(),
            'first_test_target_utc': target_times[test][0].isoformat(),
            'last_test_target_utc': target_times[test][-1].isoformat(),
            'train_pairs': int(train.sum()), 'test_pairs': int(test.sum()),
            'model_metrics': model_metrics, 'persistence_metrics': baseline_metrics,
            'limitations': ['2026 only: less than a complete annual cycle.',
                'Archive retrieval/QC/revisions do not establish real-time availability at historical origins.',
                'Test is partial-week; missing target hours are excluded, not filled.',
                'Sparse stations can bias evaluation toward available reporting hours.',
                'Rain/cyclones/ENSO and calibrated prediction intervals are not yet model inputs.',
                'Weights frozen before this week; test inputs update as observed. First target is Tuesday.',
                'No hyperparameter tuning on this test week.']}
        write_json(out / f'{city}_report.json', report)
        joblib.dump({'model': model, 'feature_names': list(X.columns), 'report': report,
                     'prediction_rule': 'origin_temperature_plus_predicted_change'}, model_out/f'{city}.joblib')
        summaries.append({'city': city, 'train_pairs': int(train.sum()), 'test_pairs': int(test.sum()),
            'first_test_target_ist': target_times[test][0].tz_convert('Asia/Kolkata').isoformat(),
            'last_test_target_ist': target_times[test][-1].tz_convert('Asia/Kolkata').isoformat(),
            **model_metrics, 'persistence_mae_c': baseline_metrics['mae_c']})
        all_predictions.append(table)
        X.tail(200).to_csv(out/f'{city}_feature_preview.csv')
        print(f'{city}: {test.sum()} observed test pairs; MAE {model_metrics["mae_c"]:.3f} C; '
              f'persistence {baseline_metrics["mae_c"]:.3f} C', flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(out/'cross_validation.csv', index=False)
    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(out/'all_predictions.csv', index=False)
    save_manifest(model_out)
    return summary



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, default=2026)
    parser.add_argument('--week-start', help='Monday YYYY-MM-DD in IST; default current week')
    args = parser.parse_args()
    print(run(args.year, args.week_start).to_string(index=False))
