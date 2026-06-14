# Team Advanced Recent Deltas Experiment

Offline held-out experiment only. No production model or application behavior changed.

## Held-Out Metrics

| model | roc_auc | brier_score | log_loss | accuracy | f1 | train_seasons | blend_seasons | test_seasons |
|---|---|---|---|---|---|---|---|---|
| base_only | 0.7003 | 0.2229 | 0.6376 | 0.6446 | 0.6845 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24,2024-25 |
| series_context_only | 0.6029 | 0.2441 | 0.6838 | 0.5843 | 0.6387 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24,2024-25 |
| team_advanced_only | 0.5999 | 0.2474 | 0.6958 | 0.5964 | 0.6564 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24,2024-25 |
| base_plus_team_advanced | 0.6932 | 0.2225 | 0.6366 | 0.6566 | 0.6885 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24,2024-25 |
| base_plus_series | 0.6612 | 0.2281 | 0.6480 | 0.6265 | 0.6771 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24,2024-25 |
| base_plus_series_plus_team_advanced | 0.6626 | 0.2279 | 0.6478 | 0.6325 | 0.6806 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24,2024-25 |

## Strongest Advanced Features

| feature | standardized_coefficient | absolute_coefficient |
|---|---|---|
| last_3_pie_diff | -0.8319 | 0.8319 |
| series_to_date_pie_diff | 0.7555 | 0.7555 |
| last_2_effectiveFieldGoalPercentage_diff | 0.6572 | 0.6572 |
| last_1_offensiveRating_diff | -0.6244 | 0.6244 |
| last_3_offensiveRating_diff | 0.6234 | 0.6234 |
| last_3_turnoverPercentage_diff | 0.5726 | 0.5726 |
| last_3_netRating_diff | 0.5423 | 0.5423 |
| last_3_effectiveFieldGoalPercentage_diff | -0.5112 | 0.5112 |
| last_2_possessions_diff | -0.4707 | 0.4707 |
| last_3_defensiveRating_diff | -0.4496 | 0.4496 |

## Leakage Audit

| SEASON | GAME_ID | GAME_DATE | game_number_audit | TEAM_A | TEAM_B | v3_games_used_in_rolling_window | latest_v3_game_date_used | target_game_excluded | postgame_state_excluded |
|---|---|---|---|---|---|---|---|---|---|
| 2015-16 | 0041500102 | 2016-04-20 | 2.0000 | CLE | DET | 0041500101 | 2016-04-17 | True | True |
| 2015-16 | 0041500103 | 2016-04-22 | 3.0000 | CLE | DET | 0041500101,0041500102 | 2016-04-20 | True | True |
| 2015-16 | 0041500104 | 2016-04-24 | 4.0000 | CLE | DET | 0041500101,0041500102,0041500103 | 2016-04-22 | True | True |
| 2015-16 | 0041500112 | 2016-04-18 | 2.0000 | IND | TOR | 0041500111 | 2016-04-16 | True | True |
| 2015-16 | 0041500113 | 2016-04-21 | 3.0000 | IND | TOR | 0041500111,0041500112 | 2016-04-18 | True | True |

## Decision

DEPLOY CANDIDATE: an advanced-feature combination improved held-out Brier score or log loss over base only. Require stability review before any production change.
