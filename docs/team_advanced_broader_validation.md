# Team Advanced Broader Validation

Offline validation audit only. Production models and application behavior were unchanged.

Base + Advanced uses the existing leakage-safe V3 feature builder. Every fold fits model clones and blend coefficients only from that fold's training seasons.

## Fold Metrics

| validation_design | test_season | model | games | roc_auc | brier_score | log_loss | accuracy | f1 | expected_calibration_error | average_probability_movement |
|---|---|---|---|---|---|---|---|---|---|---|
| expanding_window | 2018-19 | base_only | 82.0000 | 0.7357 | 0.2120 | 0.6149 | 0.6951 | 0.7059 | 0.0830 | 0.0023 |
| expanding_window | 2018-19 | base_plus_team_advanced | 82.0000 | 0.7405 | 0.2117 | 0.6143 | 0.6951 | 0.7059 | 0.0743 | 0.0023 |
| expanding_window | 2019-20 | base_only | 83.0000 | 0.6187 | 0.2440 | 0.6807 | 0.5422 | 0.5870 | 0.1159 | 0.0095 |
| expanding_window | 2019-20 | base_plus_team_advanced | 83.0000 | 0.6069 | 0.2441 | 0.6812 | 0.5422 | 0.5870 | 0.1307 | 0.0095 |
| expanding_window | 2020-21 | base_only | 85.0000 | 0.7578 | 0.2080 | 0.6052 | 0.6941 | 0.6750 | 0.1095 | 0.0065 |
| expanding_window | 2020-21 | base_plus_team_advanced | 85.0000 | 0.7602 | 0.2078 | 0.6051 | 0.6941 | 0.6750 | 0.1021 | 0.0065 |
| expanding_window | 2021-22 | base_only | 87.0000 | 0.6172 | 0.2415 | 0.6778 | 0.5977 | 0.5455 | 0.0986 | 0.0061 |
| expanding_window | 2021-22 | base_plus_team_advanced | 87.0000 | 0.6148 | 0.2421 | 0.6795 | 0.6207 | 0.5823 | 0.0949 | 0.0061 |
| expanding_window | 2022-23 | base_only | 84.0000 | 0.6103 | 0.2404 | 0.6748 | 0.6190 | 0.6444 | 0.0456 | 0.0051 |
| expanding_window | 2022-23 | base_plus_team_advanced | 84.0000 | 0.6177 | 0.2400 | 0.6740 | 0.6190 | 0.6444 | 0.0749 | 0.0051 |
| expanding_window | 2023-24 | base_only | 82.0000 | 0.7244 | 0.2107 | 0.6123 | 0.6951 | 0.7525 | 0.1268 | 0.0174 |
| expanding_window | 2023-24 | base_plus_team_advanced | 82.0000 | 0.7295 | 0.2069 | 0.6038 | 0.6951 | 0.7525 | 0.0881 | 0.0174 |
| expanding_window | 2024-25 | base_only | 84.0000 | 0.6658 | 0.2340 | 0.6600 | 0.6071 | 0.6024 | 0.1230 | 0.0157 |
| expanding_window | 2024-25 | base_plus_team_advanced | 84.0000 | 0.6547 | 0.2361 | 0.6645 | 0.6190 | 0.6098 | 0.0914 | 0.0157 |
| leave_one_season_out | 2015-16 | base_only | 86.0000 | 0.7496 | 0.2056 | 0.6007 | 0.7209 | 0.7273 | 0.0825 | 0.0210 |
| leave_one_season_out | 2015-16 | base_plus_team_advanced | 86.0000 | 0.7409 | 0.2040 | 0.5966 | 0.7209 | 0.7273 | 0.0831 | 0.0210 |
| leave_one_season_out | 2016-17 | base_only | 79.0000 | 0.6755 | 0.2147 | 0.6210 | 0.6962 | 0.7500 | 0.0754 | 0.0195 |
| leave_one_season_out | 2016-17 | base_plus_team_advanced | 79.0000 | 0.6785 | 0.2139 | 0.6192 | 0.6962 | 0.7500 | 0.0559 | 0.0195 |
| leave_one_season_out | 2017-18 | base_only | 82.0000 | 0.6766 | 0.2186 | 0.6288 | 0.6951 | 0.7253 | 0.0974 | 0.0153 |
| leave_one_season_out | 2017-18 | base_plus_team_advanced | 82.0000 | 0.6784 | 0.2181 | 0.6278 | 0.6951 | 0.7253 | 0.1083 | 0.0153 |
| leave_one_season_out | 2018-19 | base_only | 82.0000 | 0.7090 | 0.2144 | 0.6190 | 0.6585 | 0.6818 | 0.0675 | 0.0197 |
| leave_one_season_out | 2018-19 | base_plus_team_advanced | 82.0000 | 0.7084 | 0.2134 | 0.6164 | 0.6585 | 0.6818 | 0.0809 | 0.0197 |
| leave_one_season_out | 2019-20 | base_only | 83.0000 | 0.6442 | 0.2356 | 0.6645 | 0.5904 | 0.5854 | 0.1061 | 0.0117 |
| leave_one_season_out | 2019-20 | base_plus_team_advanced | 83.0000 | 0.6486 | 0.2366 | 0.6672 | 0.5904 | 0.5854 | 0.1222 | 0.0117 |
| leave_one_season_out | 2020-21 | base_only | 85.0000 | 0.7096 | 0.2164 | 0.6242 | 0.6824 | 0.6494 | 0.0597 | 0.0160 |
| leave_one_season_out | 2020-21 | base_plus_team_advanced | 85.0000 | 0.7065 | 0.2165 | 0.6244 | 0.6824 | 0.6494 | 0.0826 | 0.0160 |
| leave_one_season_out | 2021-22 | base_only | 87.0000 | 0.6548 | 0.2352 | 0.6637 | 0.6322 | 0.6000 | 0.0763 | 0.0112 |
| leave_one_season_out | 2021-22 | base_plus_team_advanced | 87.0000 | 0.6497 | 0.2357 | 0.6650 | 0.6322 | 0.6000 | 0.0560 | 0.0112 |
| leave_one_season_out | 2022-23 | base_only | 84.0000 | 0.6199 | 0.2447 | 0.6835 | 0.6071 | 0.6207 | 0.1148 | 0.0162 |
| leave_one_season_out | 2022-23 | base_plus_team_advanced | 84.0000 | 0.6217 | 0.2473 | 0.6898 | 0.6071 | 0.6207 | 0.1205 | 0.0162 |
| leave_one_season_out | 2023-24 | base_only | 82.0000 | 0.7321 | 0.2151 | 0.6220 | 0.7073 | 0.7692 | 0.1477 | 0.0339 |
| leave_one_season_out | 2023-24 | base_plus_team_advanced | 82.0000 | 0.7346 | 0.2078 | 0.6058 | 0.6829 | 0.7451 | 0.0911 | 0.0339 |
| leave_one_season_out | 2024-25 | base_only | 84.0000 | 0.6658 | 0.2340 | 0.6600 | 0.6071 | 0.6024 | 0.1230 | 0.0157 |
| leave_one_season_out | 2024-25 | base_plus_team_advanced | 84.0000 | 0.6547 | 0.2361 | 0.6645 | 0.6190 | 0.6098 | 0.0914 | 0.0157 |

## Stability Summary

| scope | subgroup | validation_design | folds | brier_improved_folds | log_loss_improved_folds | roc_auc_improved_folds | average_roc_auc_delta | average_brier_delta | average_log_loss_delta | average_accuracy_delta | average_f1_delta | average_ece_delta | average_probability_movement | best_fold | best_fold_test_season | best_fold_brier_delta | worst_fold | worst_fold_test_season | worst_fold_brier_delta | largest_share_of_positive_brier_gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_games | all_games | expanding_window | 7.0000 | 4.0000 | 4.0000 | 4.0000 | -0.0008 | -0.0003 | -0.0005 | 0.0050 | 0.0063 | -0.0066 | 0.0089 | through_2022-23_test_2023-24 | 2023-24 | -0.0038 | through_2023-24_test_2024-25 | 2024-25 | 0.0021 | 0.8124 |
| all_games | all_games | leave_one_season_out | 10.0000 | 5.0000 | 5.0000 | 5.0000 | -0.0015 | -0.0005 | -0.0011 | -0.0012 | -0.0017 | -0.0058 | 0.0180 | leave_out_2023-24 | 2023-24 | -0.0073 | leave_out_2022-23 | 2022-23 | 0.0026 | 0.6553 |

## Subgroup Summary

| scope | subgroup | validation_design | folds | brier_improved_folds | log_loss_improved_folds | roc_auc_improved_folds | average_roc_auc_delta | average_brier_delta | average_log_loss_delta | average_accuracy_delta | average_f1_delta | average_ece_delta | average_probability_movement | best_fold | best_fold_test_season | best_fold_brier_delta | worst_fold | worst_fold_test_season | worst_fold_brier_delta | largest_share_of_positive_brier_gain |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subgroup | advanced_movement_over_2_pct | expanding_window | 5.0000 | 2.0000 | 2.0000 | 1.0000 | 0.0005 | -0.0007 | -0.0010 | 0.0000 | 0.0000 | -0.0270 | 0.0258 | through_2021-22_test_2022-23 | 2022-23 | -0.0151 | through_2019-20_test_2020-21 | 2020-21 | 0.0099 | 0.7101 |
| subgroup | advanced_movement_over_2_pct | leave_one_season_out | 10.0000 | 4.0000 | 4.0000 | 7.0000 | 0.0091 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | -0.0130 | 0.0276 | leave_out_2023-24 | 2023-24 | -0.0084 | leave_out_2024-25 | 2024-25 | 0.0069 | 0.4535 |
| subgroup | close_prediction_45_55 | expanding_window | 7.0000 | 6.0000 | 6.0000 | 5.0000 | 0.0310 | -0.0006 | -0.0012 | 0.0314 | 0.0201 | -0.0133 | 0.0054 | through_2019-20_test_2020-21 | 2020-21 | -0.0016 | through_2021-22_test_2022-23 | 2022-23 | 0.0014 | 0.2989 |
| subgroup | close_prediction_45_55 | leave_one_season_out | 10.0000 | 7.0000 | 7.0000 | 2.0000 | -0.0819 | 0.0001 | 0.0002 | -0.0056 | -0.0173 | -0.0115 | 0.0066 | leave_out_2023-24 | 2023-24 | -0.0019 | leave_out_2016-17 | 2016-17 | 0.0037 | 0.2666 |
| subgroup | game_number=1 | expanding_window | 7.0000 | 5.0000 | 5.0000 | 3.0000 | 0.0067 | -0.0010 | -0.0021 | 0.0000 | 0.0000 | 0.0007 | 0.0086 | through_2022-23_test_2023-24 | 2023-24 | -0.0085 | through_2023-24_test_2024-25 | 2024-25 | 0.0068 | 0.5453 |
| subgroup | game_number=1 | leave_one_season_out | 10.0000 | 4.0000 | 4.0000 | 2.0000 | -0.0051 | 0.0004 | 0.0013 | 0.0000 | 0.0000 | -0.0319 | 0.0194 | leave_out_2023-24 | 2023-24 | -0.0161 | leave_out_2022-23 | 2022-23 | 0.0098 | 0.7330 |
| subgroup | game_number=2 | expanding_window | 7.0000 | 2.0000 | 2.0000 | 2.0000 | -0.0045 | 0.0019 | 0.0043 | 0.0000 | 0.0000 | -0.0078 | 0.0114 | through_2020-21_test_2021-22 | 2021-22 | -0.0009 | through_2023-24_test_2024-25 | 2024-25 | 0.0063 | 0.5203 |
| subgroup | game_number=2 | leave_one_season_out | 10.0000 | 6.0000 | 6.0000 | 2.0000 | -0.0003 | -0.0020 | -0.0050 | -0.0067 | -0.0061 | -0.0362 | 0.0228 | leave_out_2016-17 | 2016-17 | -0.0095 | leave_out_2024-25 | 2024-25 | 0.0063 | 0.2726 |
| subgroup | game_number=3 | expanding_window | 7.0000 | 1.0000 | 1.0000 | 3.0000 | 0.0038 | 0.0002 | 0.0005 | 0.0095 | 0.0050 | -0.0167 | 0.0092 | through_2022-23_test_2023-24 | 2023-24 | -0.0068 | through_2023-24_test_2024-25 | 2024-25 | 0.0049 | 1.0000 |
| subgroup | game_number=3 | leave_one_season_out | 10.0000 | 4.0000 | 4.0000 | 1.0000 | -0.0009 | 0.0005 | 0.0018 | 0.0067 | 0.0035 | -0.0038 | 0.0154 | leave_out_2023-24 | 2023-24 | -0.0107 | leave_out_2016-17 | 2016-17 | 0.0077 | 0.6033 |
| subgroup | game_number=4 | expanding_window | 7.0000 | 6.0000 | 6.0000 | 5.0000 | 0.0128 | -0.0018 | -0.0037 | 0.0095 | 0.0121 | -0.0116 | 0.0084 | through_2022-23_test_2023-24 | 2023-24 | -0.0049 | through_2020-21_test_2021-22 | 2021-22 | 0.0026 | 0.3152 |
| subgroup | game_number=4 | leave_one_season_out | 10.0000 | 7.0000 | 7.0000 | 4.0000 | 0.0009 | -0.0013 | -0.0028 | 0.0000 | 0.0000 | -0.0056 | 0.0151 | leave_out_2023-24 | 2023-24 | -0.0108 | leave_out_2018-19 | 2018-19 | 0.0046 | 0.4601 |
| subgroup | game_number=5 | expanding_window | 7.0000 | 4.0000 | 4.0000 | 0.0000 | -0.0080 | -0.0005 | -0.0013 | 0.0000 | 0.0000 | -0.0251 | 0.0084 | through_2023-24_test_2024-25 | 2024-25 | -0.0038 | through_2018-19_test_2019-20 | 2019-20 | 0.0042 | 0.4359 |
| subgroup | game_number=5 | leave_one_season_out | 10.0000 | 6.0000 | 6.0000 | 3.0000 | 0.0056 | -0.0024 | -0.0060 | -0.0083 | -0.0057 | -0.0181 | 0.0218 | leave_out_2017-18 | 2017-18 | -0.0112 | leave_out_2020-21 | 2020-21 | 0.0045 | 0.3121 |
| subgroup | game_number=6 | expanding_window | 7.0000 | 3.0000 | 4.0000 | 0.0000 | -0.0071 | 0.0004 | 0.0012 | 0.0143 | 0.0208 | 0.0137 | 0.0070 | through_2018-19_test_2019-20 | 2019-20 | -0.0028 | through_2020-21_test_2021-22 | 2021-22 | 0.0035 | 0.5656 |
| subgroup | game_number=6 | leave_one_season_out | 10.0000 | 6.0000 | 6.0000 | 1.0000 | -0.0092 | -0.0002 | -0.0004 | 0.0000 | 0.0000 | 0.0205 | 0.0118 | leave_out_2023-24 | 2023-24 | -0.0064 | leave_out_2015-16 | 2015-16 | 0.0074 | 0.3468 |
| subgroup | game_number=7 | expanding_window | 7.0000 | 6.0000 | 6.0000 | 0.0000 | 0.0000 | -0.0020 | -0.0043 | 0.0000 | 0.0000 | -0.0009 | 0.0071 | through_2019-20_test_2020-21 | 2020-21 | -0.0072 | through_2020-21_test_2021-22 | 2021-22 | 0.0079 | 0.3278 |
| subgroup | game_number=7 | leave_one_season_out | 10.0000 | 3.0000 | 3.0000 | 0.0000 | -0.0208 | 0.0078 | 0.0186 | 0.0000 | 0.0000 | 0.0329 | 0.0170 | leave_out_2024-25 | 2024-25 | -0.0035 | leave_out_2020-21 | 2020-21 | 0.0219 | 0.5053 |
| subgroup | high_confidence_over_65 | expanding_window | 7.0000 | 2.0000 | 3.0000 | 2.0000 | -0.0111 | 0.0003 | 0.0007 | 0.0000 | 0.0000 | -0.0172 | 0.0113 | through_2022-23_test_2023-24 | 2023-24 | -0.0064 | through_2023-24_test_2024-25 | 2024-25 | 0.0030 | 0.9443 |
| subgroup | high_confidence_over_65 | leave_one_season_out | 10.0000 | 4.0000 | 4.0000 | 5.0000 | -0.0182 | -0.0011 | -0.0024 | 0.0000 | 0.0000 | -0.0002 | 0.0251 | leave_out_2023-24 | 2023-24 | -0.0190 | leave_out_2022-23 | 2022-23 | 0.0038 | 0.7319 |
| subgroup | playoff_round=conference_finals | expanding_window | 7.0000 | 3.0000 | 3.0000 | 4.0000 | 0.0223 | 0.0004 | 0.0013 | 0.0000 | 0.0000 | 0.0465 | 0.0082 | through_2022-23_test_2023-24 | 2023-24 | -0.0047 | through_2019-20_test_2020-21 | 2020-21 | 0.0046 | 0.7095 |
| subgroup | playoff_round=conference_finals | leave_one_season_out | 10.0000 | 5.0000 | 5.0000 | 6.0000 | 0.0411 | -0.0004 | -0.0009 | 0.0000 | 0.0000 | 0.0074 | 0.0156 | leave_out_2015-16 | 2015-16 | -0.0072 | leave_out_2016-17 | 2016-17 | 0.0052 | 0.4611 |
| subgroup | playoff_round=conference_semifinals | expanding_window | 7.0000 | 4.0000 | 4.0000 | 4.0000 | 0.0133 | 0.0013 | 0.0031 | 0.0232 | 0.0303 | -0.0038 | 0.0092 | through_2019-20_test_2020-21 | 2020-21 | -0.0023 | through_2023-24_test_2024-25 | 2024-25 | 0.0106 | 0.4175 |
| subgroup | playoff_round=conference_semifinals | leave_one_season_out | 10.0000 | 2.0000 | 2.0000 | 3.0000 | -0.0068 | 0.0028 | 0.0060 | 0.0043 | 0.0016 | 0.0285 | 0.0174 | leave_out_2016-17 | 2016-17 | -0.0049 | leave_out_2024-25 | 2024-25 | 0.0106 | 0.7045 |
| subgroup | playoff_round=first_round | expanding_window | 7.0000 | 4.0000 | 5.0000 | 2.0000 | -0.0024 | -0.0014 | -0.0032 | -0.0034 | -0.0032 | -0.0004 | 0.0088 | through_2022-23_test_2023-24 | 2023-24 | -0.0070 | through_2019-20_test_2020-21 | 2020-21 | 0.0004 | 0.6912 |
| subgroup | playoff_round=first_round | leave_one_season_out | 10.0000 | 8.0000 | 8.0000 | 4.0000 | -0.0030 | -0.0025 | -0.0056 | -0.0047 | -0.0042 | -0.0098 | 0.0189 | leave_out_2023-24 | 2023-24 | -0.0150 | leave_out_2022-23 | 2022-23 | 0.0038 | 0.4994 |
| subgroup | playoff_round=nba_finals | expanding_window | 7.0000 | 3.0000 | 3.0000 | 3.0000 | -0.0298 | 0.0001 | 0.0007 | 0.0000 | 0.0000 | 0.0865 | 0.0109 | through_2022-23_test_2023-24 | 2023-24 | -0.0077 | through_2023-24_test_2024-25 | 2024-25 | 0.0062 | 0.4185 |
| subgroup | playoff_round=nba_finals | leave_one_season_out | 10.0000 | 4.0000 | 4.0000 | 3.0000 | -0.0185 | 0.0008 | 0.0017 | 0.0000 | 0.0000 | 0.0530 | 0.0192 | leave_out_2023-24 | 2023-24 | -0.0162 | leave_out_2018-19 | 2018-19 | 0.0161 | 0.4536 |
| subgroup | team_a_favorite | expanding_window | 7.0000 | 5.0000 | 5.0000 | 4.0000 | 0.0017 | -0.0005 | -0.0010 | 0.0001 | -0.0010 | -0.0075 | 0.0092 | through_2022-23_test_2023-24 | 2023-24 | -0.0060 | through_2023-24_test_2024-25 | 2024-25 | 0.0039 | 0.7520 |
| subgroup | team_a_favorite | leave_one_season_out | 10.0000 | 8.0000 | 8.0000 | 5.0000 | 0.0009 | -0.0020 | -0.0046 | -0.0016 | -0.0014 | -0.0280 | 0.0188 | leave_out_2023-24 | 2023-24 | -0.0111 | leave_out_2024-25 | 2024-25 | 0.0039 | 0.4300 |
| subgroup | team_a_underdog | expanding_window | 7.0000 | 4.0000 | 4.0000 | 3.0000 | -0.0034 | 0.0000 | 0.0002 | 0.0094 | 0.0391 | -0.0010 | 0.0084 | through_2018-19_test_2019-20 | 2019-20 | -0.0007 | through_2020-21_test_2021-22 | 2021-22 | 0.0011 | 0.3678 |
| subgroup | team_a_underdog | leave_one_season_out | 10.0000 | 2.0000 | 1.0000 | 5.0000 | -0.0008 | 0.0013 | 0.0030 | 0.0000 | 0.0000 | 0.0177 | 0.0169 | leave_out_2023-24 | 2023-24 | -0.0008 | leave_out_2022-23 | 2022-23 | 0.0033 | 0.9400 |

## Decision

RESEARCH-ONLY: the advanced-feature gain is unstable, too small, concentrated in too few folds, or accompanied by recurring ROC-AUC damage.
