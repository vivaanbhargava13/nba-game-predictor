# BoxScoreAdvancedV3 Cache Pilot

This is a cache and schema reliability pilot only. It does not train or deploy a model.

## Results

- Games discovered: 166
- Games fetched: 166
- Games already cached: 0
- Failures: 0
- Circuit-breaker skips: 0
- Failure rate among attempted games: 0.00%
- Average successful request time: 0.70 seconds
- Team rows parsed: 332
- Player rows parsed: 4683

## Schema Notes

- Stable team fields audited: `offensiveRating`, `defensiveRating`, `netRating`, `pace`, `possessions`, `trueShootingPercentage`, `effectiveFieldGoalPercentage`, `assistRatio`, `offensiveReboundPercentage`, `defensiveReboundPercentage`, `reboundPercentage`, `turnoverRatio`, `PIE`
- Stable player fields audited: `minutes`, `usagePercentage`, `offensiveRating`, `defensiveRating`, `netRating`, `trueShootingPercentage`, `PIE`
- `plusMinus` is not present in BoxScoreAdvancedV3 and would require a separate traditional box-score join.

## Sample Parsed Team Row

```json
{
  "teamTricode": "PHI",
  "offensiveRating": 116.9,
  "defensiveRating": 124.7,
  "netRating": -7.9,
  "pace": 89.0,
  "possessions": 89.0,
  "trueShootingPercentage": 0.573,
  "effectiveFieldGoalPercentage": 0.519,
  "assistRatio": 14.3,
  "offensiveReboundPercentage": 0.298,
  "defensiveReboundPercentage": 0.482,
  "reboundPercentage": 0.398,
  "turnoverRatio": 12.4,
  "PIE": 0.466
}
```

## Sample Parsed Player Row

```json
{
  "teamTricode": "PHI",
  "personId": 202699,
  "firstName": "Tobias",
  "familyName": "Harris",
  "minutes": "31:25",
  "usagePercentage": 0.132,
  "offensiveRating": 131.6,
  "defensiveRating": 133.9,
  "netRating": -2.3,
  "trueShootingPercentage": 0.5,
  "PIE": 0.066
}
```

## Backfill Estimate

- Cached historical playoff games, 2015-16 through 2024-25: 834
- Cached regular-season plus playoff games: 12813
- Estimated playoff runtime: 20.1 minutes
- Estimated regular-season plus playoff runtime: 5.2 hours

## Recommendation

- Recommendation: **full playoff backfill**
- Rationale: V3 is highly reliable for playoff games and core advanced fields are stable; validate a regular-season sample before expanding further.
- Next experiment after a successful backfill: team advanced recent deltas first; player rotation strength second because it requires identity/minutes aggregation and a separate plus-minus source.

No production model or application behavior was changed.
