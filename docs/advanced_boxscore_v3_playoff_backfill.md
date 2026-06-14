# BoxScoreAdvancedV3 Playoff Backfill

Standalone historical data ingestion only. No model, prediction, or Streamlit behavior was changed.

## Coverage By Season

| Season | Games | Cache hits | Newly fetched | Team rows | Player rows | Retries |
|---|---:|---:|---:|---:|---:|---:|
| 2015-16 | 86 | 0 | 86 | 172 | 2236 | 0 |
| 2016-17 | 79 | 0 | 79 | 158 | 2046 | 0 |
| 2017-18 | 82 | 0 | 82 | 164 | 2110 | 0 |
| 2018-19 | 82 | 0 | 82 | 164 | 2120 | 0 |
| 2019-20 | 83 | 0 | 83 | 166 | 2144 | 0 |
| 2020-21 | 85 | 0 | 85 | 170 | 2518 | 0 |
| 2021-22 | 87 | 0 | 87 | 174 | 2372 | 0 |
| 2022-23 | 84 | 0 | 84 | 168 | 2321 | 0 |
| 2023-24 | 82 | 82 | 0 | 164 | 2270 | 0 |
| 2024-25 | 84 | 84 | 0 | 168 | 2413 | 0 |

## Totals

- Games discovered: 834
- Games already cached: 166
- Games newly fetched: 668
- Total successful games: 834
- Retries used: 0
- Team rows: 1668
- Player rows: 22550
- Duplicate team rows: 0
- Duplicate player rows: 0
- Average successful request time: 0.65 seconds
- Raw JSON cache size: 17.0 MB

## Failures By Type

- None.

## Missingness

- No missing values across available core team or player V3 fields.
- `plusMinus` is unavailable in BoxScoreAdvancedV3 and is explicitly marked `unavailable_in_v3`; it is not counted as a failed or missing field.

## Regular-Season Cost Estimate

- Historical regular-season games discovered: 11979
- Estimated serial runtime: 4.7 hours
- Estimate assumes the measured successful request time plus the configured courtesy delay and excludes retry overhead.

## Modeling Readiness

- Status: **ready for offline feature experiments**
- Reason: All discovered playoff games are cached with complete core V3 fields and unique team/player rows.
- Recommended next experiment: `team_advanced_recent_deltas_experiment.py` using strictly pregame rolling team-level V3 statistics.
- Player rotation strength should follow later because it requires minute/identity aggregation; V3 does not provide plus-minus.
