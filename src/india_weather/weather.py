"""Bounded, cached Open-Meteo access for live forecasts and day-one archives."""
from __future__ import annotations
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data import ROOT, write_json

CITIES = {
    'Delhi': {'latitude':28.5845,'longitude':77.2058,'station':'Safdarjung'},
    'Mumbai': {'latitude':19.0887,'longitude':72.8679,'station':'Mumbai airport'},
    'Bengaluru': {'latitude':12.95,'longitude':77.6333,'station':'HAL airport'},
}
MODEL = 'ecmwf_ifs025'
VARIABLES = ['temperature_2m','relative_humidity_2m','dew_point_2m','pressure_msl',
             'wind_speed_10m','wind_direction_10m','precipitation','cloud_cover','shortwave_radiation']
SUFFIX = '_previous_day1'


class WeatherUnavailable(RuntimeError):
    pass


def parse_hourly(payload, previous=False):
    """Reject corrupt response shapes, units, duplicate times and broken grids."""
    if not isinstance(payload,dict):raise WeatherUnavailable('Expected a provider JSON object')
    hourly, units = payload.get('hourly',{}), payload.get('hourly_units',{})
    if not isinstance(hourly,dict) or not isinstance(units,dict):
        raise WeatherUnavailable('Malformed provider hourly data')
    suffix = SUFFIX if previous else ''
    if not hourly.get('time'):
        raise WeatherUnavailable('Provider returned no hourly timestamps')
    if units.get('temperature_2m'+suffix) != '°C' or units.get('wind_speed_10m'+suffix) != 'm/s':
        raise WeatherUnavailable('Unexpected provider units')
    index = pd.DatetimeIndex(pd.to_datetime(hourly['time'], utc=True, errors='raise'))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise WeatherUnavailable('Invalid provider timestamp order')
    if len(index)>1 and not ((index[1:]-index[:-1])==pd.Timedelta(hours=1)).all():
        raise WeatherUnavailable('Expected an hourly provider grid')
    result = pd.DataFrame(index=index)
    for name in VARIABLES:
        values = hourly.get(name+suffix)
        if values is None or len(values)!=len(index):
            raise WeatherUnavailable(f'Missing or inconsistent variable: {name}')
        result[name] = pd.to_numeric(pd.Series(values,index=index),errors='coerce')
    result = result.replace([np.inf,-np.inf],np.nan)
    limits = {'temperature_2m':(-90,60),'relative_humidity_2m':(0,100),
              'wind_speed_10m':(0,120),'wind_direction_10m':(0,360),
              'precipitation':(0,1000),'cloud_cover':(0,100),'pressure_msl':(850,1100)}
    for name,(low,high) in limits.items():
        result[name] = result[name].where(result[name].between(low,high))
    if result.temperature_2m.notna().sum()<24:
        raise WeatherUnavailable('Too few valid provider temperatures')
    return result


class WeatherClient:
    """10 minute cache, bounded retries and explicitly labelled stale fallback.

    A per-process lock avoids concurrent cold-start requests to the same cache.
    Deploy one worker for this pilot; use a shared cache for multiple replicas.
    """
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(cache_dir or ROOT/'data/live_cache')
        self.lock = threading.Lock()

    def fetch(self,city,previous=False):
        if city not in CITIES:
            raise ValueError('Unsupported city')
        suffix = SUFFIX if previous else ''
        params = {k:CITIES[city][k] for k in ['latitude','longitude']}
        params.update({'models':MODEL,'timezone':'UTC','wind_speed_unit':'ms',
            'hourly':','.join(v+suffix for v in VARIABLES),'past_days':4,'forecast_days':7 if not previous else 3})
        endpoint = 'https://previous-runs-api.open-meteo.com/v1/forecast' if previous else 'https://api.open-meteo.com/v1/forecast'
        key = hashlib.sha256(json.dumps([endpoint,params],sort_keys=True).encode()).hexdigest()
        path = self.cache_dir/f'{key}.json'
        with self.lock:
            cached = None
            if path.exists():
                try:
                    cached = json.loads(path.read_text())
                    parse_hourly(cached['payload'],previous)
                    age = (pd.Timestamp.now(tz='UTC')-pd.Timestamp(cached['retrieved_at_utc'])).total_seconds()
                    if 0<=age<600:
                        return {**cached,'stale':False,'cache_age_seconds':int(age)}
                except (ValueError,KeyError,TypeError,WeatherUnavailable):
                    cached = None
            try:
                with requests.Session() as session:
                    session.mount('https://',HTTPAdapter(max_retries=Retry(total=1,backoff_factor=.3,status_forcelist=[429,502,503,504])))
                    response = session.get(endpoint,params=params,timeout=(3,8))
                    response.raise_for_status()
                payload = response.json()
                parse_hourly(payload,previous)
                item = {'payload':payload,'retrieved_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
                        'source':'Open-Meteo / ECMWF IFS 0.25°','provider_model':MODEL,
                        'kind':'aligned_day1_forecast' if previous else 'latest_forecast',
                        'request_url':response.url,'stale':False,'cache_age_seconds':0}
                # Atomic replace; weather data are public and contain no credentials.
                self.cache_dir.mkdir(parents=True,exist_ok=True)
                temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(item));temporary.replace(path)
                return item
            except (requests.RequestException,ValueError,WeatherUnavailable) as exc:
                if cached:
                    age=(pd.Timestamp.now(tz='UTC')-pd.Timestamp(cached['retrieved_at_utc'])).total_seconds()
                    if 0<=age<=6*3600:
                        return {**cached,'stale':True,'cache_age_seconds':int(age)}
                raise WeatherUnavailable('Weather provider unavailable; no recent cache') from exc


def forecast_features(frame):
    """Inputs are forecast values, not later observed weather.

    For the provider-aligned day1 archive, earlier valid rows correspond to
    earlier forecast offsets. No future rows of this archive enter a feature.
    Original dissemination times remain unverified and are disclosed in reports.
    """
    X = frame.copy()
    angle = np.deg2rad(frame.wind_direction_10m)
    X['wind_u']=-frame.wind_speed_10m*np.sin(angle)
    X['wind_v']=-frame.wind_speed_10m*np.cos(angle)
    X = X.drop(columns='wind_direction_10m')
    for variable in ['temperature_2m','wind_speed_10m','pressure_msl','relative_humidity_2m']:
        for window in [6,24,72]:
            rolling = frame[variable].rolling(window,min_periods=max(2,window//2))
            X[f'{variable}_mean_{window}h']=rolling.mean()
            X[f'{variable}_std_{window}h']=rolling.std()
    for hours in [1,3,24]:
        X[f'temp_delta_{hours}h']=frame.temperature_2m-frame.temperature_2m.shift(hours)
    for lag in [1,3,24,48,72]:
        X[f'temp_lag_{lag}h']=frame.temperature_2m.shift(lag)
    local=frame.index.tz_convert('Asia/Kolkata')
    hour=local.hour+local.minute/60
    phase=(local.dayofyear-1+hour/24)/np.where(local.is_leap_year,366,365)
    X['hour_sin']=np.sin(2*np.pi*hour/24);X['hour_cos']=np.cos(2*np.pi*hour/24)
    X['year_sin']=np.sin(2*np.pi*phase);X['year_cos']=np.cos(2*np.pi*phase)
    return X.replace([np.inf,-np.inf],np.nan)
