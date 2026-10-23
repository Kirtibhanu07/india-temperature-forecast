"""Export verified artifacts, validation figures and a short-lived fallback snapshot."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'india-weather-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

from india_weather.data import ROOT,write_json
from india_weather.weather import CITIES,WeatherClient,parse_hourly,forecast_features
from india_weather.correction import load_archive,portable_predict


def export(snapshot=False):
    source=ROOT/'reports/correction/week_2026-10-19'
    site=ROOT.parent/'weather-dashboard'
    if not (site/'.openai/hosting.json').exists():site=ROOT/'dashboard'
    assets=site/'dist/assets';assets.mkdir(parents=True,exist_ok=True)
    models=json.loads((ROOT/'models/correction/week_2026-10-19/portable.json').read_text())
    shutil.copy2(ROOT/'models/correction/week_2026-10-19/portable.json',assets/'portable.json')
    shutil.copy2(source/'report.json',assets/'report.json')
    summary=pd.read_csv(source/'summary.csv');predictions=pd.read_csv(source/'all_predictions.csv')
    colors=['#9caaa2','#547c39','#bd7446']
    fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(3)
    for shift,col,label,color in zip([-.25,0,.25],['raw_mae_c','selected_mae_c','persistence_mae_c'],
            ['Raw ECMWF day-one forecast','Selected local correction','Persistence'],colors):
        bars=ax.bar(x+shift,summary[col],.24,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
    ax.set(xticks=x,xticklabels=summary.city,ylabel='MAE (°C) · lower is better',ylim=(0,2.85))
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=9)
    fig.suptitle('Local corrections versus two baselines',fontsize=16,fontweight='bold')
    fig.text(.02,.02,'39 Delhi / 43 Mumbai / 43 Bengaluru matched pairs. Partial-week retrospective test, 20–21 Oct 2026. No broad skill claim.',fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(assets/'correction_scores.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8))
    for ax,city in zip(axes,['Delhi','Mumbai','Bengaluru']):
        d=predictions[predictions.city.eq(city)];dates=pd.to_datetime(d.valid_utc,utc=True)
        for col,label,color,style in [('observed_c','Measured station','#176b87','-'),
            ('selected_forecast_c','Selected correction','#547c39','-'),
            ('raw_forecast_c','Raw day-one forecast','#9caaa2',':')]:
            ax.plot(dates,d[col],style,marker='.',ms=3,color=color,label=label)
        ax.set(title=city,ylabel='°C');ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M',tz=ZoneInfo('Asia/Kolkata')))
    fig.suptitle('Temperature forecasts and measured outcomes · IST',fontsize=15,fontweight='bold')
    fig.text(.02,.01,'Frozen fit and earlier separate calibration. Lines join available outcomes across gaps. Original run availability is unverified.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(assets/'correction_forecasts.png',dpi=160);plt.close(fig)
    for name in ['correction_scores.png','correction_forecasts.png']:shutil.copy2(assets/name,source/name)
    # Fixtures prove raw time-series feature parity as well as model prediction parity.
    archive=load_archive(ROOT/'data/forecast_archive/previous_day1_2024_2026.json')
    fixtures=[]
    for city,frame in archive.items():
        features=forecast_features(frame)
        for index in [1000,5000,12000,len(frame)-48]:
            window=frame.iloc[max(0,index-72):index+1]
            row=features.iloc[index]
            hourly={'time':[v.strftime('%Y-%m-%dT%H:%M') for v in window.index],
                **{k+'_previous_day1':[None if pd.isna(v) else float(v) for v in window[k]] for k in window.columns}}
            fixtures.append({'city':city,'hourly':hourly,'index':len(window)-1,
                'expected_features':{k:None if pd.isna(v) else float(v) for k,v in row.items()},
                'expected_adjustment':portable_predict(models['cities'][city],row)})
    write_json(site/'scripts/parity-fixtures.json',fixtures)
    if snapshot:
        client=WeatherClient();cached={}
        for city in CITIES:
            cached[city]={'latest':client.fetch(city),'previous':client.fetch(city,True)}
            print('Verified live feeds:',city,flush=True)
        write_json(assets/'snapshot.json',cached)
    for p in (ROOT/'data/raw').glob('*.metadata.json'):shutil.copy2(p,ROOT/'data/provenance'/p.name)
    provenance=ROOT/'data/forecast_archive/previous_day1_2024_2026.metadata.json'
    shutil.copy2(provenance,ROOT/'data/provenance'/provenance.name)
    write_json(ROOT/'models/correction/week_2026-10-19/manifest.json',{
        'schema_version':1,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'models/correction/week_2026-10-19').iterdir() if p.name!='manifest.json'}})
    if site!=ROOT/'dashboard':
        # Publish browser source with the Python repository, without Site identity,
        # hosting credentials, Git internals or unrelated local state.
        for folder in ['dist','scripts']:
            shutil.copytree(site/folder,ROOT/'dashboard'/folder,dirs_exist_ok=True)
        shutil.copy2(site/'package.json',ROOT/'dashboard/package.json')
    print('Exported browser artifacts and validation plots:',assets)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',action='store_true')
    export(parser.parse_args().snapshot)
