"""FastAPI pilot. Run: uvicorn app:app --host 0.0.0.0 --port 8000"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI,HTTPException,Query
from pydantic import BaseModel,ConfigDict

from india_weather.service import ForecastService
from india_weather.weather import WeatherUnavailable

City=Literal['Delhi','Mumbai','Bengaluru']


@asynccontextmanager
async def lifespan(app):
    app.state.service=ForecastService()
    yield


app=FastAPI(title='India Weather Lab',version='0.2.0',lifespan=lifespan,
    description='Upstream weather forecasts and separately labelled, station-trained day-one temperature corrections. Research pilot.')


class PredictRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    city:City
    horizon_hours:Literal[24]=24


@app.get('/healthz')
def health():
    return {'status':'ok','correction_model_loaded':app.state.service.models is not None}


@app.get('/forecast')
def forecast(city:City='Mumbai',days:int=Query(7,ge=1,le=7)):
    try:return app.state.service.forecast(city,days)
    except WeatherUnavailable as exc:raise HTTPException(503,str(exc),headers={'Retry-After':'60'}) from exc


@app.post('/predict')
def predict(request:PredictRequest):
    try:
        result=app.state.service.prediction(request.city)
        if not result['available']:raise HTTPException(503,result['reason'])
        return result
    except WeatherUnavailable as exc:raise HTTPException(503,str(exc),headers={'Retry-After':'60'}) from exc
