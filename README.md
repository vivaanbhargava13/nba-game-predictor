# NBA Playoff Game Predictor

A small Python project that uses `nba_api`, `pandas`, `scikit-learn`, and `seaborn` to predict NBA playoff game winners from team-level rating stats.

## Features

The model uses the difference between two teams for:

Tier 1 team strength:

- Offensive Rating (`OFF_RATING`)
- Defensive Rating (`DEF_RATING`)
- Net Rating (`NET_RATING`)
- Win percentage (`W_PCT`)
- Point differential (`PLUS_MINUS`)
- Pace (`PACE`)

Tier 2 recent form:

- Last 5 and last 10 win percentage
- Last 5 and last 10 estimated net rating
- Last 5 and last 10 average point differential

Tier 3 player availability / star power:

- Top 3 and top 5 combined player points per game
- Top 3 and top 5 combined player minutes per game

Tier 4 home court:

- Whether Team A is home
- Selected home team's own home-court edge, clipped to avoid overpowered home swings
- Legacy home/away split features kept for research ablation, but not used by the default production model

Tier 5 rest / fatigue:

- Team A rest days
- Team B rest days
- Rest-days difference

Next-step features:

- Pre-game Elo difference (`elo_diff`)
- Efficiency differences before the prediction date: effective field goal percentage, true shooting percentage, turnover percentage, rebound share, assist/turnover ratio, and free throw rate
- Stronger same-season head-to-head features: win percentage, margin, net rating, eFG%, TS%, turnover edge, and rebounding
- Style matchup features for three-point profile, two-point scoring proxy, rebounding, turnover creation, and free throw pressure
- Weighted recent-form features for wins, net rating, TS%, and defensive rating
- Star-player features for top 1, top 3, and top 5 production/usage
- Playoff context: seed difference, whether Team A has the higher seed, series game number, elimination-game flag, and series score difference

By default, it trains on regular-season team metrics and playoff game outcomes. That keeps the training setup cleaner than using full playoff metrics to predict playoff games that already happened.

Recent-form, efficiency, head-to-head, Elo, rest, style matchup, and playoff context features are computed only from games before the playoff game or prediction date. Star-power features use regular-season player averages before the game date, so future playoff production is not used.

Prediction context modes:

- `Current Hypothetical`: answers “if these teams played now.” Series fields are forced neutral: `game_number = 1`, `series_score_diff = 0`, and `elimination_game = 0`.
- `Playoff Series Context`: optional user-supplied playoff-game context. You provide game number and each team’s series wins; the app computes series score difference and elimination-game status from those inputs.
- Live scheduled-games mode is future work.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python -m src.predictor train --start-season 2015-16 --end-season 2024-25
```

This saves:

- `models/playoff_predictor.joblib`
- cached NBA API data in `data/raw/`
- processed feature rows in `data/processed/training_matchups_regular_season.csv` or `data/processed/training_matchups_playoffs.csv`
- feature selection results in `data/processed/feature_selection.csv`
- model comparison results in `data/processed/model_comparison.csv`
- model calibration results in `data/processed/model_calibration.csv`
- feature ablation results in `data/processed/feature_ablation.csv`
- home-court feature ablation results in `data/processed/home_feature_ablation.csv`
- feature importances in `data/processed/feature_importances.csv`
- charts in `reports/figures/`

By default, training compares Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, SVC, AdaBoost, and KNN with a season-based validation split:

- Train: `2015-16` through `2022-23`
- Test: `2023-24` through `2024-25`

It compares accuracy, ROC AUC, Brier score, log loss, precision, recall, and F1. Results are sorted by ROC AUC in `data/processed/model_comparison.csv`.

For production model selection, the trainer also compares raw, sigmoid-calibrated, and isotonic-calibrated probabilities for Logistic Regression, Random Forest, and Extra Trees. Calibration curves and expected calibration error are saved to `data/processed/model_calibration.csv`. The selected model metadata records model type, calibration method, train/test seasons, feature set, and validation metrics.

Before model comparison, the trainer ranks features with Extra Trees and evaluates top `5`, `10`, `15`, `20`, `25`, and `all` feature subsets using the same season split. It saves the subset results to `data/processed/feature_selection.csv` and trains the compared models using the best subset.

After model comparison, the trainer runs feature-group ablation for Random Forest and Extra Trees. It evaluates baseline features, corrected-sign features, H2H, style matchups, weighted recent form, star features, all features, and top `10`/`15`/`20` importance subsets. Results are saved to `data/processed/feature_ablation.csv`.

It also runs a home-court ablation that compares old split features, only the binary home-team flag, a clipped home-advantage feature, and clipped split features. Results are saved to `data/processed/home_feature_ablation.csv`. Training records one `selected_home_feature_design` and uses that same design in the printed output, model metadata, and production feature set.

The production artifact saves two mode-specific pipelines in `models/playoff_predictor.joblib`:

- `Current Hypothetical`: normal “if these teams played now” prediction with no series context.
- `Playoff Series Context`: optional user-supplied series context using `game_number`, `series_score_diff`, and `elimination_game`.

The current hypothetical production model uses the corrected baseline feature set, `baseline_plus_corrected_signs`, with the selected home-court design from `home_feature_ablation.csv`:

- `OFF_RATING_DIFF`
- `DEF_RATING_DIFF`
- `NET_RATING_DIFF`
- `W_PCT_DIFF`
- `PLUS_MINUS_DIFF`
- `PACE_DIFF`
- selected home-court feature columns, such as `home_team_A` plus `home_advantage_diff` or clipped split features
- `seed_difference`
- `higher_seed_A`

The playoff-context production model uses the same features plus:

- `game_number`
- `series_score_diff`
- `elimination_game`

Top-N research feature sets do not override these two production modes.

The expanded model pulls more NBA API data than the Tier 1 version because it needs team game logs and player averages. Successful API responses are cached, so the first run is the slowest.

Training prints season and game progress as it builds rows. NBA API requests use retry/backoff with a short delay between calls. Player stats are cached once per season/date and reused across games; if that endpoint fails, training continues with neutral star-power defaults.

Engineered training rows are cached in `data/processed/`. Cache filenames include the stats source, such as `training_matchups_regular_season.csv` and per-season files like `training_matchups_2023-24_regular_season.csv`, so Regular Season and Playoffs feature builds do not collide.

Processed caches include a feature-schema marker. If `FEATURE_COLUMNS` changes, old processed files are ignored and rebuilt automatically.

Force a fresh feature rebuild only when you need it:

```bash
python -m src.predictor train --start-season 2015-16 --end-season 2024-25 --force-refresh
```

## Predict a matchup

```bash
python -m src.predictor predict --team1 BOS --team2 NYK --season 2024-25 --prediction-date 2025-04-19 --home-team team1
```

You can also use full team names:

```bash
python -m src.predictor predict --team1 "Boston Celtics" --team2 "New York Knicks" --season 2024-25 --prediction-date 2025-04-19 --home-team team1
```

The prediction command prints the final feature row before calling the model, which is useful for checking `home_team_A`, `home_advantage_diff`, and any playoff context fields when enabled.

To include explicit playoff-series context from the CLI, provide a valid game number and series score. The total series wins must equal `game_number - 1`.

```bash
python -m src.predictor predict --team1 NYK --team2 BOS --season 2024-25 --prediction-date 2025-05-10 --home-team team1 --prediction-context-mode "Playoff Series Context" --game-number 6 --team-a-series-wins 3 --team-b-series-wins 2
```

To sanity-check home-court sensitivity after retraining the expanded model:

```bash
python -m src.predictor sanity-home-away --team1 CLE --team2 NYK --season 2024-25 --prediction-date 2025-04-19
```

## Run the Streamlit app

Train the model first if `models/playoff_predictor.joblib` does not exist:

```bash
python -m src.predictor train --start-season 2015-16 --end-season 2024-25
```

Then start the GUI:

```bash
streamlit run app.py
```

The app lets you choose Team A, Team B, the home team, season, stats source, prediction date, and prediction mode.

In `Current Hypothetical`, series context is hidden and forced neutral. In `Playoff Series Context`, the app shows compact series-score controls. The score must be valid for the chosen game number, for example Game 7 requires the series wins to add up to 6.

The app also shows explainability outputs:

- Top 10 feature importances
- Feature values used for the current prediction
- Top factors pushing toward Team A or Team B
- Saved prediction explanations in `data/processed/prediction_explanations.csv`
- A matchup explanation chat panel for follow-up questions about the current prediction

The chat panel is grounded only in the current matchup context: teams, season, prediction date, home team, probabilities, model factors, feature values, feature importances, and saved model metrics. It does not include injuries, lineup news, trades, or other external facts.

The app uses the OpenAI API only when `OPENAI_API_KEY` is available. Without a key, it falls back to deterministic local responses and keeps running.

Optional OpenAI setup:

```bash
export OPENAI_API_KEY="your_api_key_here"
export OPENAI_MODEL="gpt-4o-mini"
streamlit run app.py
```

Chat transcripts are saved to `data/processed/prediction_chats.csv`.

## Explore the data

```bash
python -m src.predictor visualize --season 2023-24
```

This creates seaborn plots for rating relationships and team metric correlations.

## Notes

The NBA Stats API can occasionally rate-limit or time out. This project caches successful responses under `data/raw/`, so repeated runs are faster and more reliable.
