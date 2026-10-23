import {variables,buildFeatures,adjustment} from './inference.mjs';
const locations={Mumbai:{latitude:19.0887,longitude:72.8679,station:'Mumbai airport'},Delhi:{latitude:28.5845,longitude:77.2058,station:'Safdarjung'},Bengaluru:{latitude:12.95,longitude:77.6333,station:'HAL airport'}};
const $=id=>document.getElementById(id),modelName={raw:'Unchanged forecast',bias:'Bias adjustment',ridge:'Linear correction',boosting:'Boosted-tree correction'};
let models=null,evaluation=null,snapshots={},latest=null,requestId=0,variable='temperature_2m';
const fmt=(v,n=1)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(n);
const dateKey=t=>{const p=new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Kolkata',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(new Date(t));return ['year','month','day'].map(k=>p.find(x=>x.type===k).value).join('-')};
const localTime=t=>new Intl.DateTimeFormat('en-IN',{timeZone:'Asia/Kolkata',day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'}).format(new Date(t));
const instant=t=>new Date(t.endsWith('Z')?t:t+'Z');
function message(text){$('notice').hidden=!text;$('notice').textContent=text||''}
function show(view){document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!==view+'-view');document.querySelectorAll('.nav').forEach(el=>{const active=el.dataset.view===view;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active))});if(view==='validation')renderValidation()}
document.querySelectorAll('.nav').forEach(b=>b.addEventListener('click',()=>show(b.dataset.view)));
document.querySelectorAll('[data-open]').forEach(b=>b.addEventListener('click',()=>{show(b.dataset.open);$('validation-view').scrollIntoView({behavior:'smooth',block:'start'})}));
function validPayload(payload,previous){
  const suffix=previous?'_previous_day1':'',h=payload?.hourly,u=payload?.hourly_units;
  if(!h?.time?.length||u?.['temperature_2m'+suffix]!=='°C'||u?.['wind_speed_10m'+suffix]!=='m/s')return false;
  if(!variables.every(v=>Array.isArray(h[v+suffix])&&h[v+suffix].length===h.time.length))return false;
  return h.time.every((t,i)=>Number.isFinite(instant(t).getTime())&&(!i||instant(t)-instant(h.time[i-1])===3600000));
}
async function getWeather(city,previous=false){
  const suffix=previous?'_previous_day1':'',loc=locations[city],key=`iwl-v1-${city}-${previous}`;
  let cached=null;try{cached=JSON.parse(localStorage.getItem(key))}catch{}
  const usable=p=>p&&validPayload(p.payload,previous)&&Date.now()-Date.parse(p.retrieved_at_utc)>=0&&Date.now()-Date.parse(p.retrieved_at_utc)<6*3600000;
  if(usable(cached)&&Date.now()-Date.parse(cached.retrieved_at_utc)<600000)return {...cached,stale:false};
  const endpoint=previous?'https://previous-runs-api.open-meteo.com/v1/forecast':'https://api.open-meteo.com/v1/forecast';
  const params=new URLSearchParams({latitude:loc.latitude,longitude:loc.longitude,models:'ecmwf_ifs025',timezone:'UTC',wind_speed_unit:'ms',hourly:variables.map(v=>v+suffix).join(','),past_days:4,forecast_days:previous?3:7});
  const controller=new AbortController(),timeout=setTimeout(()=>controller.abort(),18000);
  try{
    const response=await fetch(`${endpoint}?${params}`,{signal:controller.signal});if(!response.ok)throw Error('Forecast provider unavailable');
    const payload=await response.json();if(!validPayload(payload,previous))throw Error('Invalid provider response');
    const value={payload,retrieved_at_utc:new Date().toISOString(),stale:false};
    try{localStorage.setItem(key,JSON.stringify(value))}catch{}
    return value;
  }catch(error){
    const fallback=usable(cached)?cached:snapshots[city]?.[previous?'previous':'latest'];
    if(usable(fallback))return {...fallback,stale:true};
    throw error;
  }finally{clearTimeout(timeout)}
}
function rows(packet){const h=packet.payload.hourly;return h.time.map((t,i)=>Object.fromEntries([['time',instant(t).toISOString()],...variables.map(v=>[v,h[v][i]])]))}
function populateDates(){
  const now=Math.floor(Date.now()/3600000)*3600000,available=[...new Set(latest.rows.filter(r=>Date.parse(r.time)>=now).map(r=>dateKey(r.time)))];
  if(!available.length)throw Error('Provider data no longer covers the current forecast window');
  $('date').min=available[0];$('date').max=available.at(-1);
  if(!available.includes($('date').value))$('date').value=available[0];
}
function renderCurrent(packet){
  const r=latest.rows.find(r=>Date.parse(r.time)>=Math.floor(Date.now()/3600000)*3600000);if(!r)throw Error('No current forecast hour');
  $('temperature').textContent=fmt(r.temperature_2m);$('humidity').textContent=fmt(r.relative_humidity_2m,0)+'%';$('wind').textContent=fmt(r.wind_speed_10m)+' m/s';$('rain').textContent=fmt(r.precipitation)+' mm';
  $('condition').textContent=r.precipitation==null||r.cloud_cover==null?'Weather state unavailable':r.precipitation>=.1?'Rain expected':r.cloud_cover>=70?'Mostly cloudy':r.cloud_cover>=25?'Partly cloudy':'Mostly clear';
  const localHour=Number(new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Kolkata',hour:'2-digit',hour12:false}).format(new Date(r.time)));
  document.querySelector('.weather-symbol').textContent=localHour<6||localHour>=18?'☾':r.cloud_cover>=70?'☁':'☀';
  $('current-time').textContent=localTime(r.time)+' IST · upstream hourly forecast';
  $('status').textContent=(packet.stale?'Cached · ':'Retrieved ')+localTime(packet.retrieved_at_utc)+' IST';
  $('download').disabled=false;
}
function renderCorrection(packet){
  if(!models)throw Error('Correction model unavailable');
  const target=Math.floor(Date.now()/3600000)*3600000+24*3600000,h=packet.payload.hourly;
  const index=h.time.findIndex(t=>instant(t).getTime()===target),artifact=models.cities[$('city').value];
  if(index<72||h.temperature_2m_previous_day1[index]==null)throw Error('No matching next-day forecast and history');
  const features=buildFeatures(h,index),raw=h.temperature_2m_previous_day1[index];
  if(!Number.isFinite(raw))throw Error('Invalid next-day temperature');
  const value=raw+adjustment(artifact,features);if(!Number.isFinite(value))throw Error('Correction could not be calculated');
  $('corrected').textContent=fmt(value);$('change').textContent=(value-raw>=0?'+':'')+fmt(value-raw)+'° vs day-one forecast';
  $('model-tag').textContent=modelName[artifact.kind].toUpperCase();
  $('target-time').textContent='Target '+localTime(new Date(target).toISOString())+' IST · nominal 24h';
  $('interval').textContent=fmt(value-artifact.radius_c)+'–'+fmt(value+artifact.radius_c)+' °C';
  $('raw-day1').textContent='Unchanged day-one forecast: '+fmt(raw)+' °C';
  $('coverage-note').textContent=`Earlier calibration; ${fmt(artifact.report.test_interval_coverage*100,0)}% held-out coverage on ${artifact.report.test_rows} pairs. Not guaranteed.`;
  if(packet.stale)message('The provider is delayed. A cached feed (under 6 hours old) is being used; check the forecast target and retrieval time.');
}
async function load(){
  const id=++requestId,city=$('city').value,loc=locations[city];latest=null;
  $('station').textContent=`${loc.station} · ${loc.latitude.toFixed(4)}° N, ${loc.longitude.toFixed(4)}° E`;
  $('refresh').disabled=true;$('download').disabled=true;$('status').textContent='Fetching weather forecasts…';message('');
  for(const k of ['temperature','humidity','wind','rain','corrected','interval'])$(k).textContent='—';
  $('model-tag').textContent='24H TARGET';$('change').textContent='Loading…';$('target-time').textContent='Finding the next-day target…';$('raw-day1').textContent='Unchanged day-one forecast: —';$('condition').textContent='Loading the weather';$('chart').innerHTML='<p class="muted">Loading the hourly forecast…</p>';
  const results=await Promise.allSettled([getWeather(city),getWeather(city,true)]);if(id!==requestId)return;
  const [live,previous]=results;
  try{
    if(live.status!=='fulfilled')throw live.reason;
    latest={packet:live.value,rows:rows(live.value)};populateDates();renderCurrent(live.value);renderChart();
    if(live.value.stale)message('Provider delayed: displaying a cached forecast retrieved '+localTime(live.value.retrieved_at_utc)+' IST.');
  }catch(error){$('status').textContent='Forecast connection unavailable';$('condition').textContent='Forecast unavailable';$('chart').innerHTML='<p class="muted">We could not retrieve a recent forecast. Please try Refresh.</p>';message('The latest forecast is temporarily unavailable. No old values are presented as current. Validation remains available.');}
  try{if(previous.status!=='fulfilled')throw previous.reason;renderCorrection(previous.value)}catch(error){$('change').textContent='Unavailable';$('target-time').textContent='The matching day-one feed is unavailable. Try Refresh.'}
  $('refresh').disabled=false;renderValidation();
}
function renderChart(){
  if(!latest)return;
  const date=$('date').value,d=latest.rows.filter(r=>dateKey(r.time)===date),unit=variable==='temperature_2m'?'°C':variable==='precipitation'?'mm':'m/s';
  if(!d.length){$('chart').innerHTML='<p class="muted">No forecast hours for this date. Choose another available date.</p>';return}
  const vals=d.map(r=>r[variable]).filter(v=>v!=null&&Number.isFinite(v));if(!vals.length){$('chart').textContent='No valid values for this variable.';return}
  const W=1060,H=245,L=47,R=18,T=20,B=36,bar=variable==='precipitation';
  let min=bar?0:Math.floor(Math.min(...vals)-1),max=Math.ceil(Math.max(...vals)+(bar?.2:1));if(max<=min)max=min+1;
  const x=i=>L+i*(W-L-R)/Math.max(1,d.length-1),y=v=>T+(max-v)*(H-T-B)/(max-min);
  let svg=`<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" aria-label="${variable} forecast on ${date}"><defs><linearGradient id="fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="#abc588" stop-opacity=".45"/><stop offset="100%" stop-color="#abc588" stop-opacity=".02"/></linearGradient></defs>`;
  for(let i=0;i<5;i++){const value=min+(max-min)*i/4,py=y(value);svg+=`<line x1="${L}" x2="${W-R}" y1="${py}" y2="${py}" stroke="#e8ede4"/><text x="${L-10}" y="${py+4}" text-anchor="end">${fmt(value,variable==='precipitation'?1:0)}</text>`}
  let path='',contiguous=false,points=[];
  d.forEach((r,i)=>{const value=r[variable],good=value!=null&&Number.isFinite(value);if(!good){contiguous=false;return}const px=x(i),py=y(value),time=new Intl.DateTimeFormat('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(r.time));
    if(bar)svg+=`<rect x="${px-8}" y="${py}" width="16" height="${Math.max(0,H-B-py)}" rx="3" fill="#7b9f5b"><title>${time} IST: ${fmt(value)} ${unit}</title></rect>`;
    else{path+=(contiguous?'L':'M')+px+','+py;contiguous=true;points.push([px,py,time,value])}
  });
  if(!bar){svg+=`<path d="${path}" fill="none" stroke="#547b38" stroke-width="2.5" stroke-linejoin="round"/>`;for(const [px,py,time,value]of points)svg+=`<circle cx="${px}" cy="${py}" r="3" fill="#547b38"><title>${time} IST: ${fmt(value)} ${unit}</title></circle>`}
  d.forEach((r,i)=>{if(i%3===0||i===d.length-1){const time=new Intl.DateTimeFormat('en-IN',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date(r.time));svg+=`<text x="${x(i)}" y="${H-10}" text-anchor="middle">${time}</text>`}});
  svg+='</svg>';$('chart').innerHTML=svg;$('chart').setAttribute('aria-label',`${variable.replaceAll('_',' ')} forecast for ${$('city').value}, ${date}, ${unit}`);
  $('chart-caption').textContent=`${unit} · IST · latest ECMWF forecast via Open-Meteo · correction shown separately above`;
}
function renderValidation(){
  if(!evaluation){$('validation-table').textContent='Evaluation report is loading.';return}
  const rows=evaluation.summary.map(r=>`<tr><td><strong>${r.city}</strong></td><td>${modelName[r.selected_candidate]}</td><td>${r.test_rows}</td><td>${fmt(r.raw_mae_c,3)}</td><td><strong>${fmt(r.selected_mae_c,3)}</strong></td><td>${fmt(r.persistence_mae_c,3)}</td><td>${fmt(r.test_interval_coverage*100,0)}%</td></tr>`).join('');
  $('validation-table').innerHTML=`<table><thead><tr><th>Station city</th><th>Selected method</th><th>Pairs</th><th>Raw MAE °C</th><th>Corrected MAE °C</th><th>Persistence MAE °C</th><th>Band coverage</th></tr></thead><tbody>${rows}</tbody></table>`;
}
$('city').addEventListener('change',load);$('refresh').addEventListener('click',load);$('date').addEventListener('change',renderChart);
document.querySelectorAll('.chart-tab').forEach(b=>b.addEventListener('click',()=>{variable=b.dataset.variable;document.querySelectorAll('.chart-tab').forEach(el=>{const active=el===b;el.classList.toggle('active',active);el.setAttribute('aria-pressed',String(active))});renderChart()}));
$('download').addEventListener('click',()=>{
  if(!latest)return;const data=latest.rows.filter(r=>dateKey(r.time)===$('date').value),columns=['time_utc','city',...variables,'source','retrieved_at_utc','cached_fallback'];
  const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
  const csv=[columns.join(','),...data.map(r=>[r.time,$('city').value,...variables.map(v=>r[v]),'ECMWF IFS via Open-Meteo',latest.packet.retrieved_at_utc,latest.packet.stale].map(quote).join(','))].join('\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download=`${$('city').value}-${$('date').value}-forecast.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function init(){
  const results=await Promise.allSettled([fetch('assets/portable.json').then(r=>{if(!r.ok)throw Error();return r.json()}),fetch('assets/report.json').then(r=>r.json()),fetch('assets/snapshot.json').then(r=>r.json())]);
  if(results[0].status==='fulfilled')models=results[0].value;
  if(results[1].status==='fulfilled')evaluation=results[1].value;
  if(results[2].status==='fulfilled')snapshots=results[2].value;
  await load();
}
init();
