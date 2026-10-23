import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import pandas as pd
from india_weather.evaluation import week_bounds,split_week


class WeeklySplitTests(unittest.TestCase):
    def test_ist_calendar_week_and_year_boundary(self):
        start,end=week_bounds(pd.Timestamp('2026-10-23T16:00:00Z'))
        self.assertEqual(str(start),'2026-10-19 00:00:00+05:30')
        self.assertEqual(str(end),'2026-10-26 00:00:00+05:30')
        start,_=week_bounds(pd.Timestamp('2027-01-01T12:00:00Z'))
        self.assertEqual(str(start.date()),'2026-12-28')

    def test_cutoff_applies_to_labels_not_just_inputs(self):
        origins=pd.date_range('2026-10-16',periods=24*12,freq='h',tz='UTC')
        as_of=pd.Timestamp('2026-10-23T16:00:00Z')
        train,test,start,end=split_week(origins,np.ones(len(origins),dtype=bool),as_of)
        targets=origins+pd.Timedelta(hours=24)
        self.assertTrue((targets[train]<start).all())
        self.assertTrue((targets[test]>=start).all())
        self.assertTrue((targets[test]<=as_of).all())
        self.assertFalse((train & test).any())
        sunday=origins.get_loc(pd.Timestamp('2026-10-18T12:00:00Z'))
        self.assertFalse(train[sunday])
        self.assertFalse(test[sunday])
        self.assertTrue((origins[test]>=start).all())
        self.assertTrue((targets[test]>=start+pd.Timedelta(hours=24)).all())
        self.assertLess(targets[train].max(),origins[test].min())

    def test_unknown_targets_never_become_test_outcomes(self):
        origins=pd.date_range('2026-10-16',periods=240,freq='h',tz='UTC')
        known=np.ones(len(origins),dtype=bool)
        known[100]=False
        train,test,_,_=split_week(origins,known,pd.Timestamp('2026-10-23T12:00:00Z'))
        self.assertFalse(train[100])
        self.assertFalse(test[100])

    def test_explicit_week_requires_a_monday(self):
        with self.assertRaises(ValueError):
            week_bounds(pd.Timestamp('2026-10-23T12:00:00Z'),'2026-10-20')
        with self.assertRaises(ValueError):
            week_bounds(pd.Timestamp('2026-10-23T12:00:00Z'),'2026-10-26')
        with self.assertRaises(ValueError):
            week_bounds(pd.Timestamp('2026-10-23T12:00:00'))

