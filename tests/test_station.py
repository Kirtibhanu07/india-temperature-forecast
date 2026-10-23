"""Scientific invariants: timestamps, quality flags and information cutoffs."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from india_weather.data import BOUNDS, clean_reports, qc_accepted
from india_weather.model import make_features, validation_folds


class StationTests(unittest.TestCase):
    def frame(self, n=300):
        idx = pd.date_range('2026-01-01', periods=n, freq='h', tz='UTC')
        d = pd.DataFrame(1., index=idx, columns=list(BOUNDS))
        d.temperature = 20+np.sin(np.arange(n)/4)
        return d

    def test_qc_is_source_specific(self):
        result = qc_accepted(pd.Series(['5','5','s','','1','2','f','1']),
                             pd.Series(['223','313','413','413','223','223','413','999']))
        self.assertEqual(result.tolist(), [False,True,False,True,True,False,False,False])

    def test_features_cannot_see_future(self):
        d = self.frame()
        original = make_features(d)
        changed = d.copy()
        changed.iloc[200:] = 999
        pd.testing.assert_frame_equal(original.iloc[:200], make_features(changed).iloc[:200])

    def test_delta_requires_actual_endpoint_and_missing_target_stays_missing(self):
        d = self.frame()
        d.iloc[200, d.columns.get_loc('temperature')] = np.nan
        X = make_features(d)
        self.assertTrue(np.isnan(X.iloc[200].temp_delta_1h))
        self.assertTrue(np.isnan(d.temperature.shift(-24).iloc[176]))
        self.assertAlmostEqual(X.iloc[100].temp_delta_24h, d.temperature.iloc[100]-d.temperature.iloc[76])

    def test_wind_vectors_and_calm(self):
        d = self.frame()
        d.wind_speed = 10.
        d.wind_direction = 90.
        X = make_features(d)
        self.assertAlmostEqual(X.iloc[0].wind_u_ms, -10.)
        self.assertAlmostEqual(X.iloc[0].wind_v_ms, 0.)
        d.iloc[0, d.columns.get_loc('wind_speed')] = 0
        d.iloc[0, d.columns.get_loc('wind_direction')] = np.nan
        self.assertEqual(make_features(d).iloc[0].wind_u_ms, 0.)

    def test_cv_label_cutoff_precedes_validation_origin(self):
        idx = pd.date_range('2026-01-01', periods=24*290, freq='h', tz='UTC')
        for _, fit, valid, start, end in validation_folds(idx, np.ones(len(idx), bool), pd.Timestamp('2026-10-19', tz='Asia/Kolkata')):
            self.assertLess((idx[fit]+pd.Timedelta(hours=24)).max(), idx[valid].min())
            self.assertTrue(((idx[valid]+pd.Timedelta(hours=24))<end).all())

    def test_no_subhour_lookahead_no_target_fill(self):
        raw = pd.DataFrame({'STATION':['test']*3, 'Station_name':['Test']*3,
            'DATE':['2026-01-01T00:00:00','2026-01-01T00:30:00','2026-01-01T02:00:00']})
        for name in BOUNDS:
            raw[name] = ['20','50','22'] if name=='temperature' else ['','','']
            raw[name+'_Quality_Code'] = ''
            raw[name+'_Source_Code'] = '413'
            raw[name+'_Measurement_Code'] = ''
        h, audit = clean_reports(raw, 'test')
        self.assertEqual(h.temperature.iloc[0], 20)
        self.assertTrue(np.isnan(h.temperature.iloc[1]))
        self.assertEqual(h.temperature.iloc[2], 22)
        self.assertEqual(audit['subhourly_reports_excluded'], 1)
        with self.assertRaises(ValueError):
            clean_reports(pd.concat([raw, raw.iloc[:1]]), 'test')


if __name__ == '__main__':
    unittest.main()
