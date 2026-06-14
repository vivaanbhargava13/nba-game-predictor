# Advanced Box Score Cache Pilot

This pilot evaluates historical `BoxScoreAdvancedV2` availability only. It does not train or deploy a model.

## Scope

- Seasons: 2023-24 playoffs and 2024-25 playoffs
- Source game IDs: existing cached team playoff game logs
- Raw cache: one atomic endpoint JSON response per game
- Endpoint splits: TeamStats and PlayerStats

## Results

- Games discovered: 166
- Games attempted via API: 3
- Games fetched: 0
- Games already cached: 0
- Games served from valid stale cache: 0
- Failures: 3
- Games skipped after systemic V2 failure threshold: 163
- Average request time: 2.58 seconds

## Missing Fields

- 2023-24 team `OFF_RATING`: 100.00% missing
- 2023-24 team `DEF_RATING`: 100.00% missing
- 2023-24 team `NET_RATING`: 100.00% missing
- 2023-24 team `AST_PCT`: 100.00% missing
- 2023-24 team `AST_TOV`: 100.00% missing
- 2023-24 team `AST_RATIO`: 100.00% missing
- 2023-24 team `OREB_PCT`: 100.00% missing
- 2023-24 team `DREB_PCT`: 100.00% missing
- 2023-24 team `REB_PCT`: 100.00% missing
- 2023-24 team `TM_TOV_PCT`: 100.00% missing
- 2023-24 team `EFG_PCT`: 100.00% missing
- 2023-24 team `TS_PCT`: 100.00% missing
- 2023-24 team `PACE`: 100.00% missing
- 2023-24 team `POSS`: 100.00% missing
- 2023-24 team `PIE`: 100.00% missing
- 2023-24 player `OFF_RATING`: 100.00% missing
- 2023-24 player `DEF_RATING`: 100.00% missing
- 2023-24 player `NET_RATING`: 100.00% missing
- 2023-24 player `AST_PCT`: 100.00% missing
- 2023-24 player `AST_TOV`: 100.00% missing
- 2023-24 player `AST_RATIO`: 100.00% missing
- 2023-24 player `OREB_PCT`: 100.00% missing
- 2023-24 player `DREB_PCT`: 100.00% missing
- 2023-24 player `REB_PCT`: 100.00% missing
- 2023-24 player `TM_TOV_PCT`: 100.00% missing
- 2023-24 player `EFG_PCT`: 100.00% missing
- 2023-24 player `TS_PCT`: 100.00% missing
- 2023-24 player `USG_PCT`: 100.00% missing
- 2023-24 player `PACE`: 100.00% missing
- 2023-24 player `POSS`: 100.00% missing
- 2023-24 player `PIE`: 100.00% missing
- 2024-25 team `OFF_RATING`: 100.00% missing
- 2024-25 team `DEF_RATING`: 100.00% missing
- 2024-25 team `NET_RATING`: 100.00% missing
- 2024-25 team `AST_PCT`: 100.00% missing
- 2024-25 team `AST_TOV`: 100.00% missing
- 2024-25 team `AST_RATIO`: 100.00% missing
- 2024-25 team `OREB_PCT`: 100.00% missing
- 2024-25 team `DREB_PCT`: 100.00% missing
- 2024-25 team `REB_PCT`: 100.00% missing
- 2024-25 team `TM_TOV_PCT`: 100.00% missing
- 2024-25 team `EFG_PCT`: 100.00% missing
- 2024-25 team `TS_PCT`: 100.00% missing
- 2024-25 team `PACE`: 100.00% missing
- 2024-25 team `POSS`: 100.00% missing
- 2024-25 team `PIE`: 100.00% missing
- 2024-25 player `OFF_RATING`: 100.00% missing
- 2024-25 player `DEF_RATING`: 100.00% missing
- 2024-25 player `NET_RATING`: 100.00% missing
- 2024-25 player `AST_PCT`: 100.00% missing
- 2024-25 player `AST_TOV`: 100.00% missing
- 2024-25 player `AST_RATIO`: 100.00% missing
- 2024-25 player `OREB_PCT`: 100.00% missing
- 2024-25 player `DREB_PCT`: 100.00% missing
- 2024-25 player `REB_PCT`: 100.00% missing
- 2024-25 player `TM_TOV_PCT`: 100.00% missing
- 2024-25 player `EFG_PCT`: 100.00% missing
- 2024-25 player `TS_PCT`: 100.00% missing
- 2024-25 player `USG_PCT`: 100.00% missing
- 2024-25 player `PACE`: 100.00% missing
- 2024-25 player `POSS`: 100.00% missing
- 2024-25 player `PIE`: 100.00% missing

## Full Backfill Estimate

- Historical playoff games found for 2015-16 through 2024-25: 834
- Estimated successful endpoint calls required: 834
- Estimated serial runtime: 46.3 minutes
- Direct API fee: none; practical cost is runtime, retries, cache storage, and stats.nba.com rate-limit exposure.
- Estimate includes measured request time plus the configured courtesy delay; retries can increase it.

## Source Decision

- Worth full ingestion: not yet
- Rationale: Pilot coverage or requested team-field completeness is below 95%; investigate failures before a full backfill.
- Compatibility diagnostic: the installed nba_api 1.10.2 V2 endpoint returned HTTP 200 with an empty JSON object for known playoff games, while a one-game V3 diagnostic returned populated team and player tables.

This is a data-source ingestion decision, not a modeling or deployment recommendation.
