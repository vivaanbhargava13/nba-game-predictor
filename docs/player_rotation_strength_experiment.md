# Player Rotation Strength Experiment

Offline research experiment only. No deployed models or application behavior changed.

Usage concentration is based on minutes multiplied by V3 usage percentage. Player-quality features use minutes or usage-load weighting. Every target row is built before its player rows enter history.

## Final Held-Out Metrics

| evaluation_scope | validation_design | fold | test_seasons | model | roc_auc | brier_score | log_loss | accuracy | f1 | expected_calibration_error |
|---|---|---|---|---|---|---|---|---|---|---|
| final_held_out | held_out_2023_24_2024_25 | final_test | 2023-24,2024-25 | base_only | 0.7003 | 0.2229 | 0.6376 | 0.6446 | 0.6845 | 0.0638 |
| final_held_out | held_out_2023_24_2024_25 | final_test | 2023-24,2024-25 | rotation_only | 0.4929 | 0.2602 | 0.7157 | 0.5181 | 0.5876 | 0.1009 |
| final_held_out | held_out_2023_24_2024_25 | final_test | 2023-24,2024-25 | base_plus_rotation | 0.7012 | 0.2227 | 0.6370 | 0.6506 | 0.6848 | 0.0475 |
| final_held_out | held_out_2023_24_2024_25 | final_test | 2023-24,2024-25 | base_plus_series | 0.6612 | 0.2281 | 0.6480 | 0.6265 | 0.6771 | 0.0523 |
| final_held_out | held_out_2023_24_2024_25 | final_test | 2023-24,2024-25 | base_plus_series_plus_rotation | 0.6655 | 0.2276 | 0.6471 | 0.6325 | 0.6806 | 0.0595 |

## Broader Fold Metrics

| evaluation_scope | validation_design | fold | train_seasons | test_seasons | train_games | test_games | average_probability_movement | model | roc_auc | brier_score | log_loss | accuracy | f1 | expected_calibration_error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| broader_validation | expanding_window | through_2017-18_test_2018-19 | 2015-16,2016-17,2017-18 | 2018-19 | 247.0000 | 82.0000 | 0.0162 | base_only | 0.7357 | 0.2120 | 0.6149 | 0.6951 | 0.7059 | 0.0830 |
| broader_validation | expanding_window | through_2017-18_test_2018-19 | 2015-16,2016-17,2017-18 | 2018-19 | 247.0000 | 82.0000 | 0.0162 | base_plus_rotation | 0.7387 | 0.2108 | 0.6127 | 0.7073 | 0.7273 | 0.1035 |
| broader_validation | expanding_window | through_2018-19_test_2019-20 | 2015-16,2016-17,2017-18,2018-19 | 2019-20 | 329.0000 | 83.0000 | 0.0227 | base_only | 0.6187 | 0.2440 | 0.6807 | 0.5422 | 0.5870 | 0.1159 |
| broader_validation | expanding_window | through_2018-19_test_2019-20 | 2015-16,2016-17,2017-18,2018-19 | 2019-20 | 329.0000 | 83.0000 | 0.0227 | base_plus_rotation | 0.5958 | 0.2485 | 0.6899 | 0.5422 | 0.5778 | 0.1066 |
| broader_validation | expanding_window | through_2019-20_test_2020-21 | 2015-16,2016-17,2017-18,2018-19,2019-20 | 2020-21 | 412.0000 | 85.0000 | 0.0047 | base_only | 0.7578 | 0.2080 | 0.6052 | 0.6941 | 0.6750 | 0.1095 |
| broader_validation | expanding_window | through_2019-20_test_2020-21 | 2015-16,2016-17,2017-18,2018-19,2019-20 | 2020-21 | 412.0000 | 85.0000 | 0.0047 | base_plus_rotation | 0.7530 | 0.2084 | 0.6061 | 0.6941 | 0.6750 | 0.1046 |
| broader_validation | expanding_window | through_2020-21_test_2021-22 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21 | 2021-22 | 497.0000 | 87.0000 | 0.0096 | base_only | 0.6172 | 0.2415 | 0.6778 | 0.5977 | 0.5455 | 0.0986 |
| broader_validation | expanding_window | through_2020-21_test_2021-22 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21 | 2021-22 | 497.0000 | 87.0000 | 0.0096 | base_plus_rotation | 0.6138 | 0.2428 | 0.6809 | 0.6322 | 0.5897 | 0.1104 |
| broader_validation | expanding_window | through_2021-22_test_2022-23 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22 | 2022-23 | 584.0000 | 84.0000 | 0.0021 | base_only | 0.6103 | 0.2404 | 0.6748 | 0.6190 | 0.6444 | 0.0456 |
| broader_validation | expanding_window | through_2021-22_test_2022-23 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22 | 2022-23 | 584.0000 | 84.0000 | 0.0021 | base_plus_rotation | 0.6137 | 0.2404 | 0.6747 | 0.6190 | 0.6444 | 0.0459 |
| broader_validation | expanding_window | through_2022-23_test_2023-24 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24 | 668.0000 | 82.0000 | 0.0158 | base_only | 0.7244 | 0.2107 | 0.6123 | 0.6951 | 0.7525 | 0.1268 |
| broader_validation | expanding_window | through_2022-23_test_2023-24 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23 | 2023-24 | 668.0000 | 82.0000 | 0.0158 | base_plus_rotation | 0.7224 | 0.2073 | 0.6047 | 0.6829 | 0.7347 | 0.0957 |
| broader_validation | expanding_window | through_2023-24_test_2024-25 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24 | 2024-25 | 750.0000 | 84.0000 | 0.0159 | base_only | 0.6658 | 0.2340 | 0.6600 | 0.6071 | 0.6024 | 0.1230 |
| broader_validation | expanding_window | through_2023-24_test_2024-25 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24 | 2024-25 | 750.0000 | 84.0000 | 0.0159 | base_plus_rotation | 0.6524 | 0.2379 | 0.6686 | 0.6190 | 0.6098 | 0.0835 |
| broader_validation | leave_one_season_out | leave_out_2015-16 | 2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2015-16 | 748.0000 | 86.0000 | 0.0194 | base_only | 0.7496 | 0.2056 | 0.6007 | 0.7209 | 0.7273 | 0.0825 |
| broader_validation | leave_one_season_out | leave_out_2015-16 | 2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2015-16 | 748.0000 | 86.0000 | 0.0194 | base_plus_rotation | 0.7474 | 0.2034 | 0.5955 | 0.7209 | 0.7273 | 0.0690 |
| broader_validation | leave_one_season_out | leave_out_2016-17 | 2015-16,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2016-17 | 755.0000 | 79.0000 | 0.0218 | base_only | 0.6755 | 0.2147 | 0.6210 | 0.6962 | 0.7500 | 0.0754 |
| broader_validation | leave_one_season_out | leave_out_2016-17 | 2015-16,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2016-17 | 755.0000 | 79.0000 | 0.0218 | base_plus_rotation | 0.6765 | 0.2146 | 0.6209 | 0.6962 | 0.7500 | 0.0755 |
| broader_validation | leave_one_season_out | leave_out_2017-18 | 2015-16,2016-17,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2017-18 | 752.0000 | 82.0000 | 0.0189 | base_only | 0.6766 | 0.2186 | 0.6288 | 0.6951 | 0.7253 | 0.0974 |
| broader_validation | leave_one_season_out | leave_out_2017-18 | 2015-16,2016-17,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2017-18 | 752.0000 | 82.0000 | 0.0189 | base_plus_rotation | 0.6906 | 0.2173 | 0.6260 | 0.6951 | 0.7253 | 0.0987 |
| broader_validation | leave_one_season_out | leave_out_2018-19 | 2015-16,2016-17,2017-18,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2018-19 | 752.0000 | 82.0000 | 0.0228 | base_only | 0.7090 | 0.2144 | 0.6190 | 0.6585 | 0.6818 | 0.0675 |
| broader_validation | leave_one_season_out | leave_out_2018-19 | 2015-16,2016-17,2017-18,2019-20,2020-21,2021-22,2022-23,2023-24,2024-25 | 2018-19 | 752.0000 | 82.0000 | 0.0228 | base_plus_rotation | 0.7108 | 0.2153 | 0.6204 | 0.6585 | 0.6818 | 0.1368 |
| broader_validation | leave_one_season_out | leave_out_2019-20 | 2015-16,2016-17,2017-18,2018-19,2020-21,2021-22,2022-23,2023-24,2024-25 | 2019-20 | 751.0000 | 83.0000 | 0.0115 | base_only | 0.6442 | 0.2356 | 0.6645 | 0.5904 | 0.5854 | 0.1061 |
| broader_validation | leave_one_season_out | leave_out_2019-20 | 2015-16,2016-17,2017-18,2018-19,2020-21,2021-22,2022-23,2023-24,2024-25 | 2019-20 | 751.0000 | 83.0000 | 0.0115 | base_plus_rotation | 0.6410 | 0.2365 | 0.6670 | 0.5904 | 0.5854 | 0.1221 |
| broader_validation | leave_one_season_out | leave_out_2020-21 | 2015-16,2016-17,2017-18,2018-19,2019-20,2021-22,2022-23,2023-24,2024-25 | 2020-21 | 749.0000 | 85.0000 | 0.0148 | base_only | 0.7096 | 0.2164 | 0.6242 | 0.6824 | 0.6494 | 0.0597 |
| broader_validation | leave_one_season_out | leave_out_2020-21 | 2015-16,2016-17,2017-18,2018-19,2019-20,2021-22,2022-23,2023-24,2024-25 | 2020-21 | 749.0000 | 85.0000 | 0.0148 | base_plus_rotation | 0.7126 | 0.2153 | 0.6216 | 0.6941 | 0.6667 | 0.0818 |
| broader_validation | leave_one_season_out | leave_out_2021-22 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2022-23,2023-24,2024-25 | 2021-22 | 747.0000 | 87.0000 | 0.0117 | base_only | 0.6548 | 0.2352 | 0.6637 | 0.6322 | 0.6000 | 0.0763 |
| broader_validation | leave_one_season_out | leave_out_2021-22 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2022-23,2023-24,2024-25 | 2021-22 | 747.0000 | 87.0000 | 0.0117 | base_plus_rotation | 0.6481 | 0.2355 | 0.6647 | 0.6322 | 0.6000 | 0.0694 |
| broader_validation | leave_one_season_out | leave_out_2022-23 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2023-24,2024-25 | 2022-23 | 750.0000 | 84.0000 | 0.0179 | base_only | 0.6199 | 0.2447 | 0.6835 | 0.6071 | 0.6207 | 0.1148 |
| broader_validation | leave_one_season_out | leave_out_2022-23 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2023-24,2024-25 | 2022-23 | 750.0000 | 84.0000 | 0.0179 | base_plus_rotation | 0.6222 | 0.2475 | 0.6902 | 0.6071 | 0.6207 | 0.1218 |
| broader_validation | leave_one_season_out | leave_out_2023-24 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2024-25 | 2023-24 | 752.0000 | 82.0000 | 0.0351 | base_only | 0.7321 | 0.2151 | 0.6220 | 0.7073 | 0.7692 | 0.1477 |
| broader_validation | leave_one_season_out | leave_out_2023-24 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2024-25 | 2023-24 | 752.0000 | 82.0000 | 0.0351 | base_plus_rotation | 0.7308 | 0.2087 | 0.6075 | 0.6829 | 0.7451 | 0.0938 |
| broader_validation | leave_one_season_out | leave_out_2024-25 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24 | 2024-25 | 750.0000 | 84.0000 | 0.0159 | base_only | 0.6658 | 0.2340 | 0.6600 | 0.6071 | 0.6024 | 0.1230 |
| broader_validation | leave_one_season_out | leave_out_2024-25 | 2015-16,2016-17,2017-18,2018-19,2019-20,2020-21,2021-22,2022-23,2023-24 | 2024-25 | 750.0000 | 84.0000 | 0.0159 | base_plus_rotation | 0.6524 | 0.2379 | 0.6686 | 0.6190 | 0.6098 | 0.0835 |

## Broader Summary

| validation_design | folds | brier_improved_folds | log_loss_improved_folds | roc_auc_improved_folds | average_roc_auc_delta | average_brier_delta | average_log_loss_delta | average_accuracy_delta | average_f1_delta | average_ece_delta | average_probability_movement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expanding_window | 7.0000 | 3.0000 | 3.0000 | 2.0000 | -0.0057 | 0.0008 | 0.0017 | 0.0066 | 0.0066 | -0.0075 | 0.0124 |
| leave_one_season_out | 10.0000 | 5.0000 | 5.0000 | 5.0000 | -0.0005 | -0.0002 | -0.0005 | -0.0001 | 0.0001 | 0.0002 | 0.0190 |

## Strongest Rotation Coefficients

| feature | standardized_coefficient | absolute_coefficient |
|---|---|---|
| last_3_top_7_minutes_share_diff | -0.6033 | 0.6033 |
| last_3_top_8_minutes_share_diff | 0.3929 | 0.3929 |
| last_1_top_8_minutes_share_diff | -0.3758 | 0.3758 |
| series_to_date_top_3_usage_share_diff | 0.2650 | 0.2650 |
| minutes_weighted_off_rating_diff | 0.2072 | 0.2072 |
| series_to_date_top_7_minutes_share_diff | 0.2041 | 0.2041 |
| top_5_minutes_load_last_3_diff | -0.2025 | 0.2025 |
| series_to_date_top_5_usage_share_diff | -0.1884 | 0.1884 |
| usage_weighted_true_shooting_diff | -0.1767 | 0.1767 |
| last_1_top_7_minutes_share_diff | 0.1683 | 0.1683 |
| last_3_top_5_minutes_share_diff | 0.1661 | 0.1661 |
| bench_minutes_share_diff | -0.1661 | 0.1661 |

## Leakage Audit

| SEASON | GAME_ID | GAME_DATE | game_number_audit | TEAM_A | TEAM_B | player_games_used_in_rolling_window | latest_player_game_date_used | target_game_excluded | postgame_state_excluded |
|---|---|---|---|---|---|---|---|---|---|
| 2015-16 | 0041500102 | 2016-04-20 | 2.0000 | CLE | DET | 0041500101 | 2016-04-17 | True | True |
| 2015-16 | 0041500103 | 2016-04-22 | 3.0000 | CLE | DET | 0041500101,0041500102 | 2016-04-20 | True | True |
| 2015-16 | 0041500104 | 2016-04-24 | 4.0000 | CLE | DET | 0041500101,0041500102,0041500103 | 2016-04-22 | True | True |
| 2015-16 | 0041500112 | 2016-04-18 | 2.0000 | IND | TOR | 0041500111 | 2016-04-16 | True | True |
| 2015-16 | 0041500113 | 2016-04-21 | 3.0000 | IND | TOR | 0041500111,0041500112 | 2016-04-18 | True | True |

## Decision

RESEARCH-ONLY: rotation strength does not improve probability quality across most broader-validation folds without recurring ROC-AUC damage.
