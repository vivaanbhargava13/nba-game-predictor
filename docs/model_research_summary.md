# NBA Playoff Predictor Model Research Summary

## Executive decision

Production should remain the calibrated base game model plus the app's current safe behavior. The separate series-context work may remain as a contextual playoff view and for series-level probability reasoning, but it should not replace the base next-game probability.

Recent form, broad momentum, upset-response momentum, team advanced deltas, and player rotation features remain research-only. Their gains were either absent, worse than base, extremely small, unstable across seasons, or concentrated in a single fold.

## Metric conventions

The saved production artifact reports a held-out ROC-AUC of **0.7076**, Brier score of **0.2205**, and log loss of **0.6318** on 2023-24 and 2024-25. Several offline experiments use the project's order-symmetric evaluation flow, where the same base benchmark is ROC-AUC **0.7003**, Brier **0.2229**, and log loss **0.6376**. Comparisons below use the matching base from each experiment rather than mixing evaluation paths.

Lower Brier score and log loss are better. Higher ROC-AUC is better.

## Research results

| Experiment | Best measured result versus matching base | Recommendation |
|---|---|---|
| Base model | RF isotonic: ROC 0.7076, Brier 0.2205, log loss 0.6318 | Keep in production |
| Series context | Blend: ROC 0.6644, Brier 0.2275, log loss 0.6469 versus base 0.7003 / 0.2229 / 0.6376 | Contextual view only |
| Recent form | Base + series + recent: 0.6630 / 0.2294 / 0.6514 | Research only |
| Momentum state | Base + momentum: 0.6607 / 0.2284 / 0.6482 | Research only |
| Momentum subgroups | Isolated benefit after a lower-probability team won; not robust across related groups | Research only |
| Upset response | Best all-games gated policy: 0.6763 / 0.2340 / 0.6601 versus base 0.7028 / 0.2261 / 0.6442 | Do not deploy |
| NBA API source audit | Identified low-risk sources and justified an advanced-box pilot | Data planning only |
| Advanced V3 backfill | 834/834 games, 1,668 team rows, 22,550 player rows, no quality failures | Data source ready |
| Team advanced deltas | 0.6932 / 0.2225 / 0.6366 versus base 0.7003 / 0.2229 / 0.6376 | Research only |
| Player rotation | 0.7012 / 0.2227 / 0.6370 versus base 0.7003 / 0.2229 / 0.6376 | Research only |

### 1. Base model benchmark

**Purpose:** Predict the selected team's next-game win probability from stable team strength, home-court, and seed information.

**Data and validation:** The production artifact uses 11 features and a calibrated Random Forest with isotonic calibration. It was trained on 2015-16 through 2022-23 and tested on 2023-24 and 2024-25.

**Leakage protection:** Inputs are pregame snapshots. Series score, game number, and elimination status do not enter the production game probability.

**Decision:** Ship and retain. It remains the strongest validated production benchmark.

### 2. Series context model

**Purpose:** Learn whether pregame playoff-series state adds next-game information beyond the frozen base probability.

**Data and validation:** Historical playoff states included game number, pregame wins, elimination pressure, previous-game result, series road wins, and related context. A logistic context model and learned blend were evaluated on 2023-24 and 2024-25.

**Best result:** Base-only scored ROC 0.7003, Brier 0.2229, and log loss 0.6376. Series-only scored 0.6056 / 0.2414 / 0.6768; the learned blend scored 0.6644 / 0.2275 / 0.6469.

**Leakage protection:** Every state is reconstructed before the target game. The target result and postgame series state are excluded.

**Decision:** Keep only as a contextual playoff view, graceful separate layer, and series-level reasoning aid. It should not replace the base next-game model.

### 3. Recent form experiment

**Purpose:** Test last-3, last-5, and series-to-date team form.

**Data and validation:** Historical team game logs, rolling wins, margins, ratings, and pace; final holdout was 2023-24 and 2024-25.

**Best result:** Base + series + recent form scored ROC 0.6630, Brier 0.2294, and log loss 0.6514, worse than base on every primary metric.

**Leakage protection:** Rolling windows stop before the target game and exclude postgame series state.

**Decision:** Do not ship. Raw recent form did not add reliable predictive value.

### 4. Momentum state experiment

**Purpose:** Replace simplistic streaks with surprise-adjusted momentum: actual performance versus expected performance, decayed persistence, pressure, and series-win-probability swing.

**Data and validation:** Historical playoff games with cross-fitted expected probabilities; season-held-out final test on 2023-24 and 2024-25.

**Best result:** Base + momentum scored ROC 0.6607, Brier 0.2284, and log loss 0.6482. Momentum-only was substantially weaker.

**Leakage protection:** Expected probabilities were cross-fitted, and all momentum states used only completed prior games.

**Decision:** Research only. The more principled definition of momentum still reduced predictive quality.

### 5. Momentum subgroup audit

**Purpose:** Determine whether momentum helps in narrow playoff contexts despite poor aggregate results.

**Data and validation:** Final held-out predictions from 2023-24 and 2024-25 across 17 predefined subgroups.

**Best result:** The clearest isolated signal appeared in 45 games after the lower-probability team won the previous game. Momentum variants improved both Brier and log loss in only a small minority of model-subgroup comparisons.

**Leakage protection:** The audit used held-out predictions only; subgroup membership came from information known before the target game.

**Decision:** At most, useful for explanation and hypothesis generation. The evidence is not robust enough for a gated production model.

### 6. Upset-response momentum experiment

**Purpose:** Directly test the subgroup hypothesis that an expectation-breaking prior win creates useful next-game information.

**Data and validation:** Prior-game residual, margin, venue, seed, elimination, and series-swing features at upset thresholds below 50%, 45%, and 40%, evaluated on 2023-24 and 2024-25.

**Best result:** The best all-games gated policy, using the below-40% threshold, scored ROC 0.6763, Brier 0.2340, and log loss 0.6601 versus base 0.7028 / 0.2261 / 0.6442.

**Leakage protection:** The gate and features use the previous game's pregame probability and completed result only.

**Decision:** Do not ship. The purpose-built model failed to reproduce the exploratory subgroup benefit.

### 7. NBA API feature-source audit

**Purpose:** Map reliable historical endpoints and caches to leakage-safe feature families before spending API time.

**Data and validation:** Coverage, missingness, cost, leakage risk, and deployability were reviewed for team logs, advanced boxes, player data, play-by-play, shots, hustle, and schedule information from 2015-16 through 2024-25.

**Best result:** Team logs and schedule/rest data were complete and low-risk. Advanced box scores and player rotations were recommended for a small pilot before backfill. Play-by-play, shots, and hustle remained higher-cost research sources.

**Leakage protection:** Proposed features were required to use timestamped pregame snapshots and cached historical data.

**Decision:** This was a research-priority map, not a model. Its V3 pilot recommendation proved worthwhile.

### 8. BoxScoreAdvancedV3 pilot and playoff backfill

**Purpose:** Replace the failed V2 source with a reliable advanced team/player game cache.

**Data and validation:** The pilot fetched 166/166 games from 2023-24 and 2024-25. The full backfill covered 834/834 playoff games from 2015-16 through 2024-25.

**Best result:** The final cache contains 1,668 team rows and 22,550 player rows, with no failed games, duplicate rows, malformed responses, or missing available core fields. V3 `plusMinus` is unavailable and was not invented.

**Leakage protection:** Raw completed-game records are immutable; downstream builders add a target game's rows to history only after constructing its pregame features.

**Decision:** Successful data infrastructure, not a production model. It supports controlled offline experiments.

### 9. Team advanced recent deltas experiment

**Purpose:** Test rolling offensive, defensive, shooting, pace, turnover, rebounding, assist, and PIE changes from the V3 cache.

**Data and validation:** Final 2023-24 and 2024-25 holdout, seven expanding-window folds, and ten leave-one-season-out folds.

**Best result:** Base + advanced scored ROC 0.6932, Brier 0.2225, and log loss 0.6366 versus base 0.7003 / 0.2229 / 0.6376. The probability gains were only 0.00045 Brier and 0.0010 log loss while discrimination worsened.

**Stability:** Expanding-window improvements split 4 of 7 folds; leave-one-season-out split 5 of 10. Most positive Brier gain was concentrated in 2023-24, and 2024-25 worsened.

**Leakage protection:** Each target row is created before adding that game's V3 statistics to team history.

**Decision:** Research only. The tiny aggregate gain is not stable enough to justify production complexity.

### 10. Player rotation strength experiment

**Purpose:** Test rotation concentration, usage concentration, player-quality weighting, continuity, entropy, bench share, and minutes-load stress.

**Data and validation:** V3 player rows with final holdout, expanding-window, and leave-one-season-out validation.

**Best result:** Base + rotation scored ROC 0.7012, Brier 0.2227, and log loss 0.6370 versus base 0.7003 / 0.2229 / 0.6376.

**Stability:** Only 3 of 7 expanding folds improved Brier and log loss. Expanding-window average Brier, log loss, and ROC-AUC all worsened; leave-one-season-out gains were effectively split.

**Leakage protection:** Target-game player rows enter history only after the target feature row is built.

**Decision:** Research only. The held-out lift is too small and broader validation is not dependable.

## Production policy

1. Keep the calibrated base model and its current safe app behavior.
2. Keep series context separate and clearly labeled as contextual or series-level information.
3. Do not add recent form, momentum, upset-response, team advanced, or rotation features to production.
4. Retain the V3 cache and experiment outputs so future hypotheses can be tested offline without changing the deployed predictor.
5. Require repeatable Brier or log-loss improvement across most historical folds, without consistent ROC-AUC damage, before reconsidering deployment.

