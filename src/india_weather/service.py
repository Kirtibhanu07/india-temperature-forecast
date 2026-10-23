"""Shared forecast service; upstream weather and experimental corrections stay distinct."""
import json
from pathlib import Path

import pandas as pd

from .data import ROOT
from .weather import WeatherClient,WeatherUnavailable,parse_hourly,forecast_features,CITIES
from .correction import portable_predict


class ForecastService:
    def __init__(self,client=None,model_path=None):
        self.client=client or WeatherClient()
        self.model_path=Path(model_path or ROOT/'models/correction/week_2026-10-19/portable.json')
        self.models=json.loads(self.model_path.read_text()) if self.model_path.exists() else None

    def prediction(self,city,now=None):
        if not self.models:return {'available':False,'reason':'No evaluated correction artifact installed'}
        now=pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz='UTC')
        if now.tzinfo is None:raise ValueError('now must include a timezone')
        origin=now.floor('h');target=origin+pd.Timedelta(hours=24)
        packet=self.client.fetch(city,previous=True)
        data=parse_hourly(packet['payload'],previous=True)
        if target not in data.index or pd.isna(data.loc[target,'temperature_2m']):
            return {'available':False,'reason':'Provider day-one forecast missing for the requested target'}
        row=forecast_features(data.loc[:target]).loc[target]
        artifact=self.models['cities'][city]
        raw=float(data.loc[target,'temperature_2m']);value=raw+portable_predict(artifact,row)
        radius=artifact['radius_c']
        return {'available':True,'city':city,'origin_utc':origin.isoformat(),'valid_utc':target.isoformat(),
            'raw_day1_temperature_c':raw,'temperature_c':value,'lower_c':value-radius,'upper_c':value+radius,
            'selected_candidate':artifact['kind'],'band_kind':'empirical_time_split_90_percent_band_not_guaranteed',
            'source_kind':'provider_aligned_day1_forecast_plus_selected_correction',
            'retrieved_at_utc':packet['retrieved_at_utc'],'stale':packet['stale'],
            'cache_age_seconds':packet['cache_age_seconds'],'validation':artifact['report'],
            'limitation':'Nominal 24h target from the current UTC-hour boundary. Original run dissemination times are unverified.'}

    def forecast(self,city,days=7):
        packet=self.client.fetch(city)
        data=parse_hourly(packet['payload'])
        now=pd.Timestamp.now(tz='UTC');first=now.floor('h')
        data=data.loc[(data.index>=first)&(data.index<first+pd.Timedelta(days=days))]
        if data.empty:raise WeatherUnavailable('Cached forecast no longer covers the requested period')
        # JSON round-trip maps missing values to null rather than invalid NaN.
        hourly=json.loads(data.reset_index(names='time_utc').to_json(orient='records',date_format='iso'))
        try:correction=self.prediction(city,now)
        except WeatherUnavailable:correction={'available':False,'reason':'Day-one correction feed unavailable; latest upstream forecast remains available'}
        return {'city':city,'location':CITIES[city],'source':packet['source'],'provider_model':packet['provider_model'],
            'retrieved_at_utc':packet['retrieved_at_utc'],'stale':packet['stale'],'cache_age_seconds':packet['cache_age_seconds'],
            'attribution':'Weather data by Open-Meteo.com / ECMWF','hourly':hourly,'prediction':correction,
            'notice':'Weather values are upstream model forecasts. Only the separate day-one temperature panel applies our evaluated local correction.'}
