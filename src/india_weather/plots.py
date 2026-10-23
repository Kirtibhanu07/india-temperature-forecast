"""Export GitHub-ready PNG/SVG figures from committed audits and result tables.

No observations are downloaded and no model is fitted. Every displayed metric
comes from the saved station experiment; persistence is shown alongside ML.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'india-weather-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .data import ROOT, STATIONS

BLUE, ORANGE, GRAY = '#176b87', '#c95d20', '#667785'
TZ = ZoneInfo('Asia/Kolkata')


def export(fig, out, name, caption):
    fig.text(.02, .012, caption, fontsize=9, color='#465564', va='bottom')
    fig.tight_layout(rect=(.01,.065,.99,.94))
    for suffix in ['png', 'svg']:
        fig.savefig(out/f'{name}.{suffix}', dpi=170, facecolor='white')
    plt.close(fig)


def generate(week='2026-10-19', year=2026):
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.titleweight':'bold', 'svg.fonttype':'none'})
    source = ROOT/'reports/metrics'/f'week_{week}'
    out = ROOT/'reports/figures'/f'week_{week}'
    out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(source/'summary.csv')
    cv = pd.read_csv(source/'cross_validation.csv')
    predictions = pd.read_csv(source/'all_predictions.csv')
    cities = list(STATIONS)

    # 1. Missingness is measured on the exact-hour grid after cleaning.
    variables = ['temperature','relative_humidity','wind_speed','wind_direction',
                 'sea_level_pressure','station_level_pressure','precipitation']
    values = []
    for city in cities:
        audit = json.loads((ROOT/'reports/data_quality'/f'{city}_{year}_audit.json').read_text())
        by_variable = {item['variable']:item for item in audit['variables']}
        values.append([by_variable[v]['hourly_missing_percent'] for v in variables])
    fig, ax = plt.subplots(figsize=(12,4.6))
    mat = np.array(values)
    heat = ax.imshow(mat, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
    ax.set_yticks(range(3), cities)
    ax.set_xticks(range(len(variables)), ['Temperature','Humidity','Wind speed','Wind bearing',
                     'Sea-level\npressure','Station\npressure','Precipitation'])
    for i in range(3):
        for j in range(len(variables)):
            ax.text(j,i,f'{mat[i,j]:.1f}%',ha='center',va='center',
                    color='white' if mat[i,j]>60 else '#1f2d3a',fontsize=11)
    fig.colorbar(heat,ax=ax,label='Missing measurements (%)',shrink=.8)
    fig.suptitle('Data quality | missing hourly measurements, January–21 October 2026', fontsize=15)
    export(fig,out,'01_data_quality','7,046 exact-hour slots per station. Missing targets stay missing; unreported rainfall is not zero rainfall.')

    # 2. Draw label-time intervals, including the 24-hour gap after each fit boundary.
    fig, ax = plt.subplots(figsize=(12,5.3))
    first = pd.Timestamp(f'{year}-01-01',tz=TZ)
    rows = cv[cv.city.eq(cities[0])]
    for i, row in enumerate(rows.itertuples()):
        start = pd.Timestamp(row.validation_origin_start_ist)
        end = pd.Timestamp(row.validation_target_end_exclusive_ist)
        left, boundary = mdates.date2num(first), mdates.date2num(start)
        ax.barh(i, boundary-left, left=left, color=BLUE, height=.5)
        ax.barh(i, 1, left=boundary, color='#dce3e8', height=.5)
        ax.barh(i, mdates.date2num(end)-boundary-1, left=boundary+1, color=ORANGE, height=.5)
    start = pd.Timestamp(week,tz=TZ)
    last = pd.to_datetime(summary.last_test_target_ist,utc=True).max().tz_convert(TZ)
    left, boundary = mdates.date2num(first), mdates.date2num(start)
    ax.barh(3,boundary-left,left=left,color=BLUE,height=.5)
    ax.barh(3,1,left=boundary,color='#dce3e8',height=.5)
    ax.barh(3,mdates.date2num(last)-boundary-1,left=boundary+1,color='#6b4598',height=.5)
    ax.set_yticks(range(4),['CV fold 1','CV fold 2','CV fold 3','Held-out week'])
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y',tz=TZ))
    ax.grid(axis='x',alpha=.15)
    ax.legend(handles=[Patch(color=BLUE,label='Earlier training label times'),
        Patch(color='#dce3e8',label='24h until first target'),Patch(color=ORANGE,label='Validation target window'),
        Patch(color='#6b4598',label='Available held-out targets')],loc='upper left',bbox_to_anchor=(0,-.11),ncol=2,fontsize=9)
    fig.suptitle('Chronological validation | every training answer precedes the next prediction origin',fontsize=14)
    export(fig,out,'02_validation_timeline','Schematic time windows, IST. Gaps within station records are not drawn. Held-out targets: 20–21 October only.')

    # 3. All folds are displayed, rather than selecting a winning fold.
    fig, axes = plt.subplots(1,3,figsize=(13,4.7),sharey=True)
    for ax, city in zip(axes,cities):
        d = cv[cv.city.eq(city)]
        x = np.arange(len(d))
        ax.bar(x-.18,d.mae_c,.36,color=ORANGE,label='Gradient boosting')
        ax.bar(x+.18,d.persistence_mae_c,.36,color=BLUE,label='Persistence')
        ax.set(title=city,xticks=x,xticklabels=['7 Sep','21 Sep','5 Oct'],xlabel='Validation origin-window start')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('MAE (°C) · lower is better')
    axes[-1].legend(fontsize=9)
    fig.suptitle('Three pre-test validation folds | model skill varies across periods',fontsize=15)
    export(fig,out,'03_cv_mae','Expanding training history; 14-day fold windows; +24h targets. These are validation periods, not independent trials.')

    # 4. Recompute both model and baseline errors from matched prediction rows.
    fig, axes = plt.subplots(1,2,figsize=(12,5.2))
    for ax, metric in zip(axes,['MAE','RMSE']):
        model, base = [], []
        for city in cities:
            d = predictions[predictions.city.eq(city)]
            for column, result in [('prediction_c',model),('persistence_c',base)]:
                err = (d[column]-d.actual_station_c).to_numpy()
                result.append(np.abs(err).mean() if metric=='MAE' else np.sqrt((err**2).mean()))
        x = np.arange(3)
        bars = ax.bar(x-.18,model,.36,color=ORANGE,label='Gradient boosting')
        baseline = ax.bar(x+.18,base,.36,color=BLUE,label='Persistence')
        ax.bar_label(bars,fmt='%.2f',padding=4,fontsize=10)
        ax.bar_label(baseline,fmt='%.2f',padding=4,fontsize=10)
        ax.set(xticks=x,xticklabels=cities,title=metric,ylabel='Error (°C) · lower is better',ylim=(0,max(model+base)*1.3))
        ax.grid(axis='y',alpha=.15)
    axes[1].legend(fontsize=9)
    fig.suptitle('Held-out errors | persistence has lower MAE in every city',fontsize=15)
    export(fig,out,'04_test_mae_rmse','39 Delhi / 43 Mumbai / 43 Bengaluru matched pairs. Partial-week retrospective test; no generic “95% accuracy” claim.')

    # 5. The complete small test is visible, with point markers exposing sample density.
    fig, axes = plt.subplots(3,1,figsize=(12,10))
    for ax, city in zip(axes,cities):
        d = predictions[predictions.city.eq(city)]
        t = pd.to_datetime(d.valid_utc,utc=True).dt.tz_convert(TZ)
        for column,label,color,style in [('actual_station_c','Measured',BLUE,'-'),
            ('prediction_c','Gradient boosting',ORANGE,'-'),('persistence_c','Persistence',GRAY,':')]:
            ax.plot(t,d[column],style,color=color,label=label,marker='.',ms=4,lw=1.7)
        ax.set(title=f'{city} · {len(d)} pairs',ylabel='Temperature (°C)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M',tz=TZ))
        ax.grid(alpha=.15)
        ax.legend(fontsize=9,ncol=3,loc='upper left')
    fig.suptitle('24-hour forecasts versus measured station temperatures | 20–21 October, IST',fontsize=15)
    export(fig,out,'05_forecast_vs_observed','Weights frozen before Monday 19 October. Lines connect available pairs across missing hours; no targets were imputed.')

    # 6. Signed errors reveal warm/cold bias. No independence-based CI is implied.
    fig, axes = plt.subplots(1,3,figsize=(13,4.8),sharex=True,sharey=True)
    bins = np.linspace(-4,4,17)
    for ax, city in zip(axes,cities):
        d = predictions[predictions.city.eq(city)]
        ax.hist(d.prediction_c-d.actual_station_c,bins=bins,alpha=.65,color=ORANGE,label='Gradient boosting')
        ax.hist(d.persistence_c-d.actual_station_c,bins=bins,histtype='step',lw=2,color=BLUE,label='Persistence')
        ax.axvline(0,color='#263744',lw=1)
        ax.set(title=city,xlabel='Prediction − measured (°C)')
        ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Number of matched pairs')
    axes[-1].legend(fontsize=9)
    fig.suptitle('Forecast errors | negative values are too cold, positive values too warm',fontsize=15)
    export(fig,out,'06_error_distribution','Small, serially dependent samples. Descriptive error distributions; not calibrated prediction intervals.')
    print(f'Exported six figures in PNG and SVG: {out}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--week',default='2026-10-19')
    parser.add_argument('--year',type=int,default=2026)
    args=parser.parse_args()
    generate(args.week,args.year)
