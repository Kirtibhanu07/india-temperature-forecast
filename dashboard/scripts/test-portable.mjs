import fs from 'node:fs';
import assert from 'node:assert/strict';
import {buildFeatures,adjustment} from '../dist/inference.mjs';
const models=JSON.parse(fs.readFileSync(new URL('../dist/assets/portable.json',import.meta.url)));
const fixtures=JSON.parse(fs.readFileSync(new URL('./parity-fixtures.json',import.meta.url)));
let checks=0;
for(const f of fixtures){
  const row=buildFeatures(f.hourly,f.index);
  for(const [name,expected] of Object.entries(f.expected_features)){
    if(expected==null)assert.ok(!Number.isFinite(row[name]),name);
    else assert.ok(Math.abs(row[name]-expected)<1e-6,`${f.city} ${name}: ${row[name]} != ${expected}`);
    checks++;
  }
  assert.ok(Math.abs(adjustment(models.cities[f.city],row)-f.expected_adjustment)<1e-6,`${f.city}: Python/browser prediction mismatch`);
}
console.log(`${fixtures.length} real-data fixtures passed; ${checks} feature comparisons and every portable prediction match Python.`);
