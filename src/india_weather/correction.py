"""Train original local corrections against real, unfilled station outcomes.

Candidate selection is confined to earlier CV windows. A separate chronological
calibration segment estimates an empirical 90% error band without refitting the
chosen model. The exposed October week is descriptive final evaluation only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import requests

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from .data import ROOT,STATIONS,write_json
from .evaluation import score,week_bounds
from .weather import parse_hourly,forecast_features,MODEL,VARIABLES


def download_archive(start='2024-01-01',end='2026-10-21'):
    """Reproduce the archived predictor download; preserve exact input bytes."""
    from .weather import CITIES,SUFFIX
    path=ROOT/'data/forecast_archive/previous_day1_2024_2026.json'
    if path.exists():return path
    params={'latitude':','.join(str(CITIES[c]['latitude']) for c in STATIONS),
            'longitude':','.join(str(CITIES[c]['longitude']) for c in STATIONS),
            'models':MODEL,'timezone':'UTC','wind_speed_unit':'ms','start_date':start,'end_date':end,
            'hourly':','.join(v+SUFFIX for v in VARIABLES)}
    response=requests.get('https://previous-runs-api.open-meteo.com/v1/forecast',params=params,timeout=(10,120))
    response.raise_for_status()
    payloads=response.json()
    if not isinstance(payloads,list) or len(payloads)!=3:raise ValueError('Expected three forecast locations')
    for payload in payloads:parse_hourly(payload,True)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_bytes(response.content);temporary.replace(path)
    write_json(path.with_suffix('.metadata.json'),{'url':response.url,'retrieved_at':pd.Timestamp.now(tz='UTC').isoformat(),
        'sha256':hashlib.sha256(response.content).hexdigest(),'bytes':len(response.content),
        'kind':'provider_aligned_previous_day1','availability_limit':'Original initialization and dissemination timestamps are not supplied.'})
    return path


def estimator(name):
    if name=='ridge':
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RobustScaler(),Ridge(alpha=100,solver='lsqr',tol=1e-8))
    if name=='boosting':
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
            HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,learning_rate=.05,
                l2_regularization=20,early_stopping=False,random_state=42))
    return None


def fit_candidate(name,X,y,baseline):
    if name=='raw':return {'name':name,'offset':0.0,'model':None}
    if name=='bias':return {'name':name,'offset':float(np.median(y-baseline)),'model':None}
    model=estimator(name);model.fit(X,y-baseline)
    return {'name':name,'offset':0.0,'model':model}


def corrected(bundle,X,baseline):
    values=np.asarray(baseline,dtype=float)
    adjustment=bundle['model'].predict(X) if bundle['model'] is not None else bundle['offset']
    return values+adjustment


def portable(bundle,feature_names):
    """Export the fitted numerical predictor to JSON for identical browser inference."""
    result={'kind':bundle['name'],'features':feature_names,'offset':bundle['offset']}
    model=bundle['model']
    if model is None:return result
    result['medians']=model.steps[0][1].statistics_.tolist()
    if bundle['name']=='ridge':
        scale=model.steps[1][1];reg=model.steps[2][1]
        weights=reg.coef_/scale.scale_
        result.update({'weights':weights.tolist(),'intercept':float(reg.intercept_-np.sum(weights*scale.center_))})
    else:
        reg=model.steps[-1][1]
        result['baseline']=float(reg._baseline_prediction[0,0])
        result['trees']=[[[float(n['value']),int(n['feature_idx']),float(n['num_threshold']),
                          int(n['left']),int(n['right']),int(n['is_leaf'])] for n in stage[0].nodes]
                         for stage in reg._predictors]
    return result


def portable_predict(model,row):
    """Reference interpreter also used to verify the JSON artifact."""
    if model['kind'] in ['raw','bias']:return model['offset']
    values=np.array([row[n] for n in model['features']],dtype=float)
    values=np.where(np.isnan(values),np.array(model['medians']),values)
    if model['kind']=='ridge':return float(model['intercept']+np.dot(values,model['weights']))
    value=model['baseline']
    for tree in model['trees']:
        idx=0
        while not tree[idx][5]:
            node=tree[idx];idx=node[3] if values[node[1]]<=node[2] else node[4]
        value+=tree[idx][0]
    return value


def load_archive(path):
    payloads=json.loads(path.read_text())
    if not isinstance(payloads,list) or len(payloads)!=3:raise ValueError('Expected exactly three city responses')
    frames={}
    for (city,station),payload in zip(STATIONS.items(),payloads):
        frames[city]=parse_hourly(payload,previous=True)
    return frames


def load_observed(city):
    chunks=[pd.read_csv(ROOT/'data/processed'/f'{city}_{year}_hourly.csv',index_col='time_utc',parse_dates=True)
            for year in [2024,2025,2026]]
    frame=pd.concat(chunks)
    frame.index=pd.to_datetime(frame.index,utc=True)
    if frame.index.has_duplicates:raise ValueError('Duplicate station measurements across years')
    return frame.sort_index().temperature


def train(week='2026-10-19',archive=None):
    archive=archive or ROOT/'data/forecast_archive/previous_day1_2024_2026.json'
    frames=load_archive(archive)
    as_of=pd.Timestamp.now(tz='UTC');start,end=week_bounds(as_of,week)
    fit_cutoff=start-pd.Timedelta(days=14)
    out=ROOT/'reports/correction'/f'week_{week}';out.mkdir(parents=True,exist_ok=True)
    models=ROOT/'models/correction'/f'week_{week}';models.mkdir(parents=True,exist_ok=True)
    candidates=['raw','bias','ridge','boosting']
    summary=[];cv_rows=[];all_predictions=[];public_models={};reports={}
    for city in STATIONS:
        raw=frames[city];X=forecast_features(raw);baseline=raw.temperature_2m
        observations=load_observed(city)
        y=observations.reindex(raw.index)
        known=y.notna() & baseline.notna();known.iloc[:72]=False
        origins=raw.index-pd.Timedelta(hours=24)
        persistence=pd.Series(observations.reindex(origins).to_numpy(),index=raw.index)
        for fold,days in enumerate([56,42,28],1):
            val_start=fit_cutoff-pd.Timedelta(days=days);val_end=val_start+pd.Timedelta(days=14)
            fit=known & (raw.index<val_start)
            valid=known & (origins>=val_start) & (raw.index<val_end)
            if fit.sum()<1000 or valid.sum()<50:raise ValueError(f'{city}: insufficient CV coverage')
            for name in candidates:
                bundle=fit_candidate(name,X.loc[fit],y.loc[fit],baseline.loc[fit])
                pred=corrected(bundle,X.loc[valid],baseline.loc[valid])
                cv_rows.append({'city':city,'fold':fold,'candidate':name,'rows':int(valid.sum()),
                    'validation_origin_start_ist':val_start.isoformat(),'validation_target_end_exclusive_ist':val_end.isoformat(),
                    **score(y.loc[valid].to_numpy(),pred)})
        city_cv=pd.DataFrame([r for r in cv_rows if r['city']==city])
        weighted=city_cv.assign(total=lambda d:d.mae_c*d.rows).groupby('candidate').agg(total=('total','sum'),rows=('rows','sum'))
        scores=(weighted.total/weighted.rows).to_dict()
        best=min(scores,key=scores.get)
        # A small improvement gate is fixed before looking at the final test.
        selected=best if scores[best]<=scores['raw']*.97 else 'raw'
        fit=known & (raw.index<fit_cutoff)
        cal=known & (origins>=fit_cutoff) & (raw.index<start)
        test=known & persistence.notna() & (origins>=start) & (raw.index<end) & (raw.index<=as_of)
        if cal.sum()<100 or test.sum()<2:raise ValueError(f'{city}: insufficient calibration/test observations')
        bundle=fit_candidate(selected,X.loc[fit],y.loc[fit],baseline.loc[fit])
        cal_pred=corrected(bundle,X.loc[cal],baseline.loc[cal])
        errors=abs(y.loc[cal].to_numpy()-cal_pred);n=len(errors)
        radius=float(np.quantile(errors,min(1,math.ceil((n+1)*.9)/n),method='higher'))
        prediction=corrected(bundle,X.loc[test],baseline.loc[test]);actual=y.loc[test].to_numpy()
        raw_metrics=score(actual,baseline.loc[test].to_numpy());metrics=score(actual,prediction)
        persistence_metrics=score(actual,persistence.loc[test].to_numpy())
        coverage=float(np.mean(abs(prediction-actual)<=radius))
        table=pd.DataFrame({'city':city,'origin_utc':origins[test],'valid_utc':raw.index[test],
            'observed_c':actual,'raw_forecast_c':baseline.loc[test].values,'selected_forecast_c':prediction,
            'persistence_c':persistence.loc[test].values,
            'lower_c':prediction-radius,'upper_c':prediction+radius,'error_c':prediction-actual})
        table.to_csv(out/f'{city}_predictions.csv',index=False);all_predictions.append(table)
        report={'city':city,'provider_model':MODEL,'selected_candidate':selected,'cv_mae_by_candidate':scores,
            'selection_rule':'Lowest pooled pre-test CV MAE; require at least 3% improvement over raw to use a correction.',
            'training_rows':int(fit.sum()),'calibration_rows':int(cal.sum()),'test_rows':int(test.sum()),
            'fit_labels_before_ist':fit_cutoff.isoformat(),'calibration_labels_before_ist':start.isoformat(),
            'first_test_target_utc':raw.index[test][0].isoformat(),'last_test_target_utc':raw.index[test][-1].isoformat(),
            'selected_metrics':metrics,'raw_metrics':raw_metrics,'persistence_metrics':persistence_metrics,
            'empirical_90_radius_c':radius,
            'test_interval_coverage':coverage,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'created_at_utc':as_of.isoformat(),
            'limitations':['Provider-aligned day1 forecasts: original initialization/dissemination times are not supplied.',
                'Retrospective archive and station QC/revisions; no independently verified live-performance claim.',
                'Rolling features describe forecast trajectories, not measured station history.',
                'No ENSO index or cyclone advisory feed is integrated. Rain/wind are provider forecast features.',
                'Three stations and a small exposed partial-week test do not establish India-wide skill.',
                'Empirical time-split error bands have no guaranteed coverage under weather drift or dependence.',
                'Correction applies only to matching provider/day1 feature streams, not every forecast lead.']}
        write_json(out/f'{city}_report.json',report);reports[city]=report
        portable_model=portable(bundle,list(X.columns))
        # Verify exported inference against the actual fitted predictor on many rows.
        checked=X.loc[test].head(100)
        expected=corrected(bundle,checked,np.zeros(len(checked)))
        portable_values=np.array([portable_predict(portable_model,row) for _,row in checked.iterrows()])
        if not np.allclose(expected,portable_values,atol=1e-8):raise AssertionError('Portable predictor mismatch')
        portable_model.update({'city':city,'radius_c':radius,'report':report})
        public_models[city]=portable_model
        joblib.dump({**bundle,'feature_names':list(X.columns),'radius_c':radius,'report':report},models/f'{city}.joblib')
        summary.append({'city':city,'selected_candidate':selected,'train_rows':int(fit.sum()),'calibration_rows':int(cal.sum()),
            'test_rows':int(test.sum()),'raw_mae_c':raw_metrics['mae_c'],'selected_mae_c':metrics['mae_c'],
            'persistence_mae_c':persistence_metrics['mae_c'],
            'raw_rmse_c':raw_metrics['rmse_c'],'selected_rmse_c':metrics['rmse_c'],
            'interval_radius_c':radius,'test_interval_coverage':coverage})
        print(f'{city}: selected {selected}; raw MAE {raw_metrics["mae_c"]:.3f} → selected {metrics["mae_c"]:.3f} C ({test.sum()} test pairs)',flush=True)
    pd.DataFrame(cv_rows).to_csv(out/'cross_validation.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    pd.concat(all_predictions).to_csv(out/'all_predictions.csv',index=False)
    write_json(models/'portable.json',{'schema_version':1,'provider_model':MODEL,'week':week,'cities':public_models})
    write_json(out/'report.json',{'schema_version':1,'week':week,'cities':reports,'summary':summary})
    print('Model and evaluation:',out,flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--week',default='2026-10-19')
    p.add_argument('--download',action='store_true',help='Download/prepare missing 2024–2026 observations and archived forecasts')
    args=p.parse_args()
    if args.download:
        from .data import prepare
        for year in [2024,2025,2026]:prepare(year)
        download_archive()
    train(args.week)
