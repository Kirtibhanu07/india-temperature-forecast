"""Offline checks for live-feed integrity, caching, correction exports and API errors."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

from india_weather.data import ROOT
from india_weather.weather import VARIABLES,WeatherClient,WeatherUnavailable,parse_hourly,forecast_features
from india_weather.correction import portable_predict
import numpy as np
import pandas as pd
import requests


def payload(previous=False):
    """Synthetic weather is confined to tests; never used for training or UI data."""
    idx=pd.date_range('2026-10-15',periods=240,freq='h',tz='UTC');s='_previous_day1' if previous else ''
    values={v:[1.0]*len(idx) for v in VARIABLES}
    values['temperature_2m']=(25+np.sin(np.arange(len(idx))/6)).tolist()
    values['wind_direction_10m']=[90.0]*len(idx);values['pressure_msl']=[1010.0]*len(idx)
    return {'hourly':{'time':[t.strftime('%Y-%m-%dT%H:%M') for t in idx],**{k+s:v for k,v in values.items()}},
            'hourly_units':{'temperature_2m'+s:'°C','wind_speed_10m'+s:'m/s'}}


class FeedTests(unittest.TestCase):
    def test_corrupt_shape_and_units_are_rejected(self):
        with self.assertRaises(WeatherUnavailable):parse_hourly([])
        with self.assertRaises(WeatherUnavailable):parse_hourly({'hourly':None})
        d=payload();d['hourly']['precipitation'].pop()
        with self.assertRaises(WeatherUnavailable):parse_hourly(d)
        d=payload();d['hourly_units']['wind_speed_10m']='km/h'
        with self.assertRaises(WeatherUnavailable):parse_hourly(d)

    def test_missing_rain_and_impossible_humidity_stay_unknown(self):
        d=payload();d['hourly']['precipitation'][0]=None;d['hourly']['relative_humidity_2m'][0]=150
        clean=parse_hourly(d)
        self.assertTrue(pd.isna(clean.precipitation.iloc[0]));self.assertTrue(pd.isna(clean.relative_humidity_2m.iloc[0]))

    def test_duplicate_or_irregular_provider_times_are_rejected(self):
        d=payload();d['hourly']['time'][1]=d['hourly']['time'][0]
        with self.assertRaises(WeatherUnavailable):parse_hourly(d)
        d=payload();d['hourly']['time'][1]='2026-10-15T01:30'
        with self.assertRaises(WeatherUnavailable):parse_hourly(d)

    def test_future_forecast_rows_cannot_change_earlier_feature_rows(self):
        frame=parse_hourly(payload(True),True);a=forecast_features(frame)
        altered=frame.copy();altered.iloc[150:]=999
        pd.testing.assert_frame_equal(a.iloc[:150],forecast_features(altered).iloc[:150])

    def test_cache_fallback_is_bounded_and_labelled(self):
        with tempfile.TemporaryDirectory() as directory:
            client=WeatherClient(directory)
            response=Mock();response.json.return_value=payload();response.url='https://api.open-meteo.com/v1/forecast'
            with patch('india_weather.weather.requests.Session') as factory:
                factory.return_value.__enter__.return_value.get.return_value=response
                packet=client.fetch('Mumbai');self.assertFalse(packet['stale'])
            with patch('india_weather.weather.requests.Session') as factory:
                self.assertFalse(client.fetch('Mumbai')['stale']);factory.assert_not_called()
            path=next(Path(directory).glob('*.json'));cached=json.loads(path.read_text())
            cached['retrieved_at_utc']=(pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=2)).isoformat();path.write_text(json.dumps(cached))
            with patch('india_weather.weather.requests.Session') as factory:
                factory.return_value.__enter__.return_value.get.side_effect=requests.Timeout()
                self.assertTrue(client.fetch('Mumbai')['stale'])
            cached['retrieved_at_utc']=(pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=7)).isoformat();path.write_text(json.dumps(cached))
            with patch('india_weather.weather.requests.Session') as factory:
                factory.return_value.__enter__.return_value.get.side_effect=requests.Timeout()
                with self.assertRaises(WeatherUnavailable):client.fetch('Mumbai')


class CorrectionReleaseTests(unittest.TestCase):
    def test_browser_release_assets_are_present(self):
        for name in ['index.html','app.js','styles.css','inference.mjs','assets/portable.json','assets/report.json','assets/snapshot.json']:
            self.assertTrue((ROOT/'dashboard/dist'/name).is_file(),f'Missing browser release file: {name}')

    def test_calibration_precedes_test_and_metrics_use_matching_pairs(self):
        from india_weather.evaluation import score
        folder=ROOT/'reports/correction/week_2026-10-19'
        report=json.loads((folder/'report.json').read_text())
        data=pd.read_csv(folder/'all_predictions.csv')
        for city,r in report['cities'].items():
            self.assertLess(pd.Timestamp(r['fit_labels_before_ist']),pd.Timestamp(r['calibration_labels_before_ist']))
            d=data[data.city.eq(city)];origins=pd.to_datetime(d.origin_utc,utc=True);valid=pd.to_datetime(d.valid_utc,utc=True)
            self.assertTrue((origins>=pd.Timestamp(r['calibration_labels_before_ist'])).all())
            self.assertTrue(((valid-origins)==pd.Timedelta(hours=24)).all())
            for column,field in [('selected_forecast_c','selected_metrics'),('raw_forecast_c','raw_metrics'),('persistence_c','persistence_metrics')]:
                self.assertFalse(d[column].isna().any())
                metrics=score(d.observed_c.to_numpy(),d[column].to_numpy())
                self.assertAlmostEqual(metrics['mae_c'],r[field]['mae_c'],places=10)
            self.assertEqual(len(d),r['test_rows'])

    def test_correction_checksums_and_cv_selection(self):
        import hashlib
        folder=ROOT/'models/correction/week_2026-10-19'
        manifest=json.loads((folder/'manifest.json').read_text())
        for name,expected in manifest['files'].items():self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),expected)
        model=json.loads((folder/'portable.json').read_text())
        for city,m in model['cities'].items():
            scores=m['report']['cv_mae_by_candidate'];best=min(scores,key=scores.get)
            expected=best if scores[best]<=scores['raw']*.97 else 'raw'
            self.assertEqual(m['kind'],expected)


class ApiTests(unittest.TestCase):
    def test_invalid_requests_and_provider_failure(self):
        from fastapi.testclient import TestClient
        import app as api
        with TestClient(api.app) as client:
            self.assertEqual(client.get('/healthz').status_code,200)
            self.assertEqual(client.post('/predict',json={'city':'Unknown'}).status_code,422)
            self.assertEqual(client.post('/predict',json={'city':'Delhi','horizon_hours':48}).status_code,422)
            self.assertEqual(client.post('/predict',json={'city':'Delhi','extra':'ignored?'}).status_code,422)
            self.assertEqual(client.get('/forecast?days=100').status_code,422)
            with patch.object(api.app.state.service,'forecast',side_effect=WeatherUnavailable('Provider unavailable')):
                result=client.get('/forecast?city=Delhi')
                self.assertEqual(result.status_code,503);self.assertEqual(result.headers['retry-after'],'60')

    def test_predict_returns_the_service_forecast(self):
        from fastapi.testclient import TestClient
        import app as api
        expected={'available':True,'city':'Mumbai','temperature_c':25.3,'source_kind':'test_service_fixture'}
        with TestClient(api.app) as client:
            with patch.object(api.app.state.service,'prediction',return_value=expected) as call:
                result=client.post('/predict',json={'city':'Mumbai'})
                self.assertEqual(result.status_code,200);self.assertEqual(result.json(),expected);call.assert_called_once_with('Mumbai')


if __name__=='__main__':unittest.main()
