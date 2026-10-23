"""Chronological splits and regression metrics shared by training and tests."""
from datetime import date
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
TZ = "Asia/Kolkata"


def week_bounds(as_of,week_start=None):
    as_of=pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError('as_of must include a timezone')
    local=as_of.tz_convert(TZ)
    if week_start is None:
        start=local.normalize()-pd.Timedelta(days=local.weekday())
    else:
        selected=date.fromisoformat(week_start)
        if selected.weekday()!=0:
            raise ValueError('--week-start must be a Monday in YYYY-MM-DD format')
        start=pd.Timestamp(selected,tz=TZ)
    if start>local:
        raise ValueError('The selected test week has not started yet')
    return start,start+pd.Timedelta(days=7)


def split_week(origins,known,as_of,week_start=None):
    """Split by TARGET timestamps: no Monday labels in training Sunday origins."""
    origins=pd.DatetimeIndex(origins)
    if origins.tz is None:
        raise ValueError('Forecast origins require a timezone')
    start,end=week_bounds(as_of,week_start)
    valid_at=origins+pd.Timedelta(hours=24)
    known=np.asarray(known,dtype=bool)
    train=known & (valid_at<start) & (valid_at<=as_of)
    # A model fitted at Monday 00:00 cannot be used for Sunday-origin forecasts.
    # Thus its first 24h target is Tuesday, even though the test origin week starts Monday.
    test=known & (origins>=start) & (valid_at<end) & (valid_at<=as_of)
    return train,test,start,end


def score(actual,predicted):
    return {'mae_c':float(mean_absolute_error(actual,predicted)),
            'rmse_c':float(np.sqrt(mean_squared_error(actual,predicted))),
            'bias_c':float(np.mean(predicted-actual)),
            'within_2c':float(np.mean(np.abs(predicted-actual)<=2)),
            'r2':float(r2_score(actual,predicted)) if len(actual)>1 else None}

