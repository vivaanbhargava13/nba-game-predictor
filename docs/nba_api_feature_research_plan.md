# NBA API Feature Research Plan

This is a source and coverage audit only. It does not train or deploy a model.

## Evidence Rules

- Local CSV coverage is measured before considering new API calls.
- Season aggregates do not count as game-level historical coverage.
- Every future experiment must build features from games strictly before the target game.
- Target-game DNP status, box scores, play-by-play, shots, and hustle data are leakage.

## Recommended Order

### 1. Schedule Rest Travel

- Cached seasons: 10/10
- Median game coverage: 100.0%
- API cost: low: already available in cached team logs
- Leakage risk: low for prior schedule; future schedule must be known pregame
- Deployability risk: low for rest/home; medium for travel due to arena/geocode maintenance
- Next experiment: rest days; back-to-back; home/away; travel-distance proxy; time-zone changes; Denver altitude indicator
- Rationale: Rest and venue features derive from complete local logs without new calls.

### 1. Team Game Logs

- Cached seasons: 10/10
- Median game coverage: 100.0%
- API cost: low-medium: one league call or 30 team calls per season/type
- Leakage risk: low when rows are filtered strictly before prediction_date
- Deployability risk: low
- Next experiment: rolling wins/margins; estimated possessions and ratings; rest; home-away splits; opponent-adjusted form
- Rationale: Complete local game-level history; cheapest leakage-safe experiment.

### 2. Team Advanced Box Scores

- Cached seasons: 10/10
- Median game coverage: 0.0%
- API cost: high for game boxes; low for season aggregates
- Leakage risk: medium: game box must enter rolling state only after that game; season aggregates need date-bounded queries
- Deployability risk: medium: game-by-game backfill and endpoint stability
- Next experiment: pregame rolling efficiency; shooting/turnover/rebounding form; style mismatch; PIE trend
- Rationale: High-value efficiency fields exist, but current caches are season aggregates; pilot a small game-box backfill before committing.

### 3. Player Availability Rotation

- Cached seasons: 10/10
- Median game coverage: 0.0%
- API cost: medium-high: league player logs plus optional game boxes
- Leakage risk: high: availability and rotation baselines must use only games before target; DNP comments for target game are forbidden
- Deployability risk: medium-high: roster identity and injury timing
- Next experiment: top-player minutes/usage present; top-7 minutes share; rotation stability; missing star production; bench compression
- Rationale: Player snapshots exist across seasons, but true game-level rotation and availability require a separate cached backfill.

### 4. Four Factors Misc Scoring Usage

- Cached seasons: 0/10
- Median game coverage: 0.0%
- API cost: very high: several calls per historical game
- Leakage risk: medium: all rolling summaries must stop before target game
- Deployability risk: high: multi-endpoint completeness and request volume
- Next experiment: four-factor matchup edges; paint/transition/second-chance style; assisted-shot and usage concentration
- Rationale: No local historical game cache; first run a small endpoint coverage pilot.

### 5. Shot Profile

- Cached seasons: 0/10
- Median game coverage: 0.0%
- API cost: high: player/team/date parameterized requests
- Leakage risk: medium-high: date windows must end before target game
- Deployability risk: high: expensive extraction and opponent attribution
- Next experiment: rim/midrange/three frequency; corner-three rate; shot-quality proxy; profile stability; opponent profile allowed
- Rationale: No local historical game cache; first run a small endpoint coverage pilot.

### 6. Play By Play Momentum

- Cached seasons: 0/10
- Median game coverage: 0.0%
- API cost: very high: one large call per game
- Leakage risk: high: target-game events cannot be used; only prior-game summaries may enter the next prediction
- Deployability risk: high: parsing/version drift and large backfill
- Next experiment: scoring runs; lead changes; quarter control; clutch margin; fourth-quarter control; timeout response where event sequence permits
- Rationale: No local historical game cache; first run a small endpoint coverage pilot.

### 8. Hustle Effort

- Cached seasons: 0/10
- Median game coverage: 0.0%
- API cost: very high: one specialized call per game
- Leakage risk: medium: use only prior-game rolling effort summaries
- Deployability risk: very high: uncertain older-season availability
- Next experiment: effort differential; contest rate; deflections; loose-ball recovery; screen-assist and box-out stability
- Rationale: No local historical game cache; first run a small endpoint coverage pilot.

## Suggested Pilots

1. Reuse cached team logs for rest, schedule, and opponent-adjusted rolling form.
2. Backfill a small stratified sample of BoxScoreAdvancedV2 games and verify field completeness by season.
3. Build rotation features from cached player snapshots, then separately test PlayerGameLogs coverage.
4. Pilot one playoff round for four factors, shot charts, play-by-play, and hustle before any full backfill.

No source should move to modeling until its cache is reproducible, date-bounded, and coverage-tested.
