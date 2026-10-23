// Portable inference for the original Python-trained local correction models.
// No fitting happens in the browser. Coefficients/trees come from portable.json.
export const variables=['temperature_2m','relative_humidity_2m','dew_point_2m','pressure_msl','wind_speed_10m','wind_direction_10m','precipitation','cloud_cover','shortwave_radiation'];
const number=x=>x==null?NaN:Number(x);
export function buildFeatures(hourly,index,suffix='_previous_day1'){
  const at=(name,i)=>i<0?NaN:number(hourly[name+suffix]?.[i]);
  const row={};for(const v of variables)row[v]=at(v,index);
  const angle=row.wind_direction_10m*Math.PI/180;
  row.wind_u=-row.wind_speed_10m*Math.sin(angle);row.wind_v=-row.wind_speed_10m*Math.cos(angle);delete row.wind_direction_10m;
  for(const v of ['temperature_2m','wind_speed_10m','pressure_msl','relative_humidity_2m'])for(const w of [6,24,72]){
    const vals=[];for(let i=Math.max(0,index-w+1);i<=index;i++){const x=at(v,i);if(Number.isFinite(x))vals.push(x)}
    const mean=vals.reduce((a,b)=>a+b,0)/vals.length;
    const enough=vals.length>=Math.max(2,Math.floor(w/2));
    row[`${v}_mean_${w}h`]=enough?mean:NaN;
    row[`${v}_std_${w}h`]=enough?Math.sqrt(vals.reduce((a,b)=>a+(b-mean)**2,0)/(vals.length-1)):NaN;
  }
  for(const h of [1,3,24])row[`temp_delta_${h}h`]=row.temperature_2m-at('temperature_2m',index-h);
  for(const h of [1,3,24,48,72])row[`temp_lag_${h}h`]=at('temperature_2m',index-h);
  const instant=new Date(hourly.time[index].endsWith('Z')?hourly.time[index]:hourly.time[index]+'Z');
  const local=new Date(instant.getTime()+5.5*3600000),year=local.getUTCFullYear(),hour=local.getUTCHours()+local.getUTCMinutes()/60;
  const day=Math.floor((Date.UTC(year,local.getUTCMonth(),local.getUTCDate())-Date.UTC(year,0,1))/86400000);
  const days=(year%4===0&&(year%100!==0||year%400===0))?366:365,phase=(day+hour/24)/days;
  row.hour_sin=Math.sin(2*Math.PI*hour/24);row.hour_cos=Math.cos(2*Math.PI*hour/24);
  row.year_sin=Math.sin(2*Math.PI*phase);row.year_cos=Math.cos(2*Math.PI*phase);
  return row;
}
export function adjustment(model,row){
  if(model.kind==='raw'||model.kind==='bias')return model.offset;
  const x=model.features.map((name,i)=>Number.isFinite(row[name])?row[name]:model.medians[i]);
  if(model.kind==='ridge')return model.intercept+x.reduce((sum,value,i)=>sum+value*model.weights[i],0);
  let result=model.baseline;
  for(const tree of model.trees){let i=0;while(!tree[i][5]){const node=tree[i];i=x[node[1]]<=node[2]?node[3]:node[4]}result+=tree[i][0]}
  return result;
}
