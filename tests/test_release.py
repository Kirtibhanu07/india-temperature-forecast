"""Offline checks of published results and locally produced trusted models."""
import hashlib
import json
from pathlib import Path
import unittest

from india_weather.data import ROOT, STATIONS
from india_weather.evaluation import score
import joblib
import numpy as np
import pandas as pd


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.results = ROOT/'reports/metrics/week_2026-10-19'
        self.models = ROOT/'models/week_2026-10-19'

    def test_metrics_match_published_predictions(self):
        summary = pd.read_csv(self.results/'summary.csv').set_index('city')
        all_rows = pd.read_csv(self.results/'all_predictions.csv')
        for city in STATIONS:
            d = all_rows[all_rows.city.eq(city)]
            metrics = score(d.actual_station_c.to_numpy(), d.prediction_c.to_numpy())
            for name,value in metrics.items():
                self.assertAlmostEqual(value, summary.loc[city,name], places=10)
            baseline = score(d.actual_station_c.to_numpy(),d.persistence_c.to_numpy())
            self.assertAlmostEqual(baseline['mae_c'], summary.loc[city,'persistence_mae_c'],places=10)
            origin = pd.to_datetime(d.origin_utc,utc=True)
            valid = pd.to_datetime(d.valid_utc,utc=True)
            self.assertTrue(((valid-origin)==pd.Timedelta(hours=24)).all())
            self.assertEqual(len(d), int(summary.loc[city,'test_pairs']))

    def test_model_manifest_and_training_cutoff(self):
        manifest = json.loads((self.models/'manifest.json').read_text())
        for city in STATIONS:
            p = self.models/f'{city}.joblib'
            entry = manifest['models'][p.name]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),entry['sha256'])
            artifact = joblib.load(p)  # Trusted project artifacts only.
            self.assertEqual(artifact['feature_names'],entry['feature_names'])
            self.assertEqual(artifact['report']['input_sha256'],entry['training_data_sha256'])
            self.assertEqual(artifact['model'].steps[-1][1].get_params(),entry['estimator_parameters'])
            report = artifact['report']
            self.assertLess(pd.Timestamp(report['last_training_target_utc']),
                            pd.Timestamp(report['first_test_origin_utc']))
            self.assertEqual(report['target_type'],'actual_unfilled_station_temperature')

    def test_saved_model_reproduces_published_forecast(self):
        fixture = json.loads((ROOT/'tests/fixtures/Mumbai_feature_row.json').read_text())
        artifact = joblib.load(self.models/'Mumbai.joblib')
        features = pd.DataFrame([fixture['features']], dtype=float)
        features = features[artifact['feature_names']]
        pred = fixture['current_temperature_c']+artifact['model'].predict(features)[0]
        self.assertAlmostEqual(pred,fixture['expected_prediction_c'],places=8)

    def test_document_links_and_notebook_syntax(self):
        import re
        for path in [ROOT/'README.md',*(ROOT/'docs').glob('*.md')]:
            for link in re.findall(r'\]\(([^\s)]+)\)',path.read_text()):
                if '://' in link or link.startswith('#'):
                    continue
                self.assertTrue((path.parent/link.split('#')[0]).exists(),f'{path}: missing {link}')
        nb = json.loads((ROOT/'notebooks/01_observations_to_forecast.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                compile(''.join(cell['source']),'<notebook>','exec')


if __name__=='__main__':
    unittest.main()
