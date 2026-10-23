"""Python dashboard counterpart to the publicly hosted browser dashboard."""
import os
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title='India Weather Lab',page_icon='🌤️',layout='wide')
st.title('India Weather Lab')
st.caption('Live upstream weather • station-trained temperature correction • transparent validation')
city=st.selectbox('Station location',['Mumbai','Delhi','Bengaluru'])
backend=os.getenv('WEATHER_BACKEND_URL','http://127.0.0.1:8000').rstrip('/')


@st.cache_data(ttl=600,show_spinner=False)
def fetch(city):
    response=requests.get(backend+'/forecast',params={'city':city,'days':7},timeout=(4,45))
    response.raise_for_status()
    return response.json()


try:
    with st.spinner('Loading the forecast and evaluated correction…'):data=fetch(city)
except requests.RequestException:
    st.error('The forecast backend is unavailable. Start FastAPI and check the provider connection.')
    st.stop()

if data['stale']:st.warning('Provider delayed: displaying a labelled cached forecast.')
st.caption(f"{data['source']} · retrieved {data['retrieved_at_utc']} · these are weather-model forecasts")
frame=pd.DataFrame(data['hourly']);frame['time_utc']=pd.to_datetime(frame.time_utc,utc=True).dt.tz_convert('Asia/Kolkata')
frame=frame.set_index('time_utc')
temperature,rain,wind=st.tabs(['Temperature','Rainfall','Wind'])
with temperature:st.line_chart(frame[['temperature_2m']]);st.caption('Latest upstream forecast, °C')
with rain:st.bar_chart(frame[['precipitation']]);st.caption('Upstream hourly precipitation, mm; missing values are not zero rain.')
with wind:st.line_chart(frame[['wind_speed_10m']]);st.caption('Upstream wind speed, m/s')
pred=data['prediction']
st.subheader('Nominal 24-hour temperature research forecast')
if pred['available']:
    cols=st.columns(3)
    cols[0].metric('Selected temperature',f"{pred['temperature_c']:.1f} °C")
    cols[1].metric('Unchanged day-one forecast',f"{pred['raw_day1_temperature_c']:.1f} °C")
    cols[2].metric('Empirical error band',f"{pred['lower_c']:.1f}–{pred['upper_c']:.1f} °C")
    st.caption(f"Target: {pred['valid_utc']} · selected method: {pred['selected_candidate']} · band coverage is not guaranteed")
    with st.expander('Validation and limitations'):st.json(pred['validation'])
else:st.info(pred['reason'])
st.info('Rain and wind panels show provider forecasts. ENSO indices and cyclone advisories are not integrated into our correction model.')
st.markdown('[Weather data by Open-Meteo.com](https://open-meteo.com/) · [Official India weather warnings](https://mausam.imd.gov.in/)')
