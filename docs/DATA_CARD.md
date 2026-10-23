# Data card

## Source and scope

- Provider: NOAA National Centers for Environmental Information (NCEI).
- Dataset: Global Historical Climatology Network hourly (GHCNh), v1.1.0 documentation.
- [Product overview](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly).
- [Format manual](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf).
- [Source metadata and use conditions](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C01688/html).
- Inputs: 2026 station/year PSV files, retrieved 23 October 2026 UTC.
- Actual snapshot coverage: 1 January–21 October 2026; last exact-hour report 13:00 UTC / 18:30 IST.
- Granularity: individual station; not a city-wide average or India-wide grid.

| City | Station ID | Station environment | Raw reports | Exact-hour slots | Accepted temperature hours |
|---|---|---|---:|---:|---:|
| Delhi | INI0000VIDD | Safdarjung | 4,892 | 7,046 | 3,882 |
| Mumbai | INI0000VABB | Airport | 14,075 | 7,046 | 7,007 |
| Bengaluru | INM00043296 | HAL airport | 13,383 | 7,046 | 6,779 |

## Provenance and reproduction

Each `data/provenance/*.metadata.json` records the public source URL, retrieval
timestamp, last-modified header, byte count and SHA-256 of the original PSV.
Raw observations and local processed CSVs are ignored by Git. Derived missingness
audits and evaluation tables are included. Redownloads can contain revisions;
compare fingerprints before claiming bit-identical reproduction. The cache is
reused unless `--refresh` is passed.

## Units

Temperature and dew point: decimal °C. Wind speed: m/s. Direction: degrees from
true north, indicating where the wind comes from. Pressure: hPa. Precipitation: mm,
with accumulation duration/reporting semantics needing further review.
Relative humidity can be derived by the source from temperature/dew point.

## Cleaning policy

1. Preserve original blank strings and QC codes when reading.
2. Parse timestamps as UTC; reject invalid timestamps and stop on duplicates.
3. Coerce weather measurements to numbers; replace nonfinite/unusable values with NaN.
4. Apply source-specific QC rules. Blank QC is accepted but is not a certificate
   of correctness. Unknown nonblank flags are conservatively excluded.
5. Apply broad physical bounds rather than removing a percentile of extremes.
6. Keep exact UTC-hour reports. Exclude half-hour reports instead of averaging
   later measurements into an earlier origin. Reindex absent hours as NaN.
7. Never impute target temperatures. Current origin temperature must be measured.
8. Preserve rainfall/gusts for inspection, without using them in the first model.

## Known limitations

Delhi has 44.90% missing hourly temperatures and may require another station or
a different sampling study. Evaluations can overrepresent available reporting
hours. QC may reject real extremes as well as bad measurements; sensitivity to
this policy has not been quantified. Archived QC/revisions may use later data.
Exact report arrival times are unknown, so valid time is not proof of historical
availability. The data span less than one annual cycle. No station-homogenization,
urban representativeness or multi-year climate-baseline study has been completed.

## Data not yet integrated

NWP forecast runs, operational cyclone advisories, ENSO indices, reliable hourly
rainfall, terrain grids and spatial neighbouring-station features are future work.
Do not infer that a missing storm/rain report means the event did not occur.
