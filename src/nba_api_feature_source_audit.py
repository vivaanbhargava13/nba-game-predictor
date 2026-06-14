from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_api.stats import endpoints

from .nba_data import season_range

AUDIT_SEASONS = season_range("2015-16", "2024-25")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_DOC_PATH = Path("docs/nba_api_feature_research_plan.md")

FAMILY_CONFIG = {
    "team_game_logs": {
        "endpoints": ["TeamGameLog", "LeagueGameLog", "TeamGameLogs"],
        "fields": [
            "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA",
            "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB", "AST",
            "TOV", "STL", "BLK", "PF", "PTS", "PLUS_MINUS",
        ],
        "api_cost": "low-medium: one league call or 30 team calls per season/type",
        "leakage_risk": "low when rows are filtered strictly before prediction_date",
        "feature_ideas": (
            "rolling wins/margins; estimated possessions and ratings; rest; "
            "home-away splits; opponent-adjusted form"
        ),
        "priority": 1,
        "deployability_risk": "low",
    },
    "team_advanced_box_scores": {
        "endpoints": ["BoxScoreAdvancedV2", "LeagueDashTeamStats Advanced"],
        "fields": [
            "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "EFG_PCT",
            "TS_PCT", "TM_TOV_PCT", "OREB_PCT", "DREB_PCT", "REB_PCT",
            "AST_TOV", "PIE", "POSS",
        ],
        "api_cost": "high for game boxes; low for season aggregates",
        "leakage_risk": (
            "medium: game box must enter rolling state only after that game; "
            "season aggregates need date-bounded queries"
        ),
        "feature_ideas": (
            "pregame rolling efficiency; shooting/turnover/rebounding form; "
            "style mismatch; PIE trend"
        ),
        "priority": 2,
        "deployability_risk": "medium: game-by-game backfill and endpoint stability",
    },
    "four_factors_misc_scoring_usage": {
        "endpoints": [
            "BoxScoreFourFactorsV2", "BoxScoreMiscV2", "BoxScoreScoringV2",
            "BoxScoreUsageV2", "LeagueDashTeamStats measure types",
        ],
        "fields": [
            "EFG_PCT", "FTA_RATE", "TM_TOV_PCT", "OREB_PCT", "OPP_EFG_PCT",
            "PTS_OFF_TOV", "PTS_2ND_CHANCE", "PTS_FB", "PTS_PAINT",
            "PCT_FGA_3PT", "PCT_PTS_PAINT", "PCT_AST_FGM", "USG_PCT",
        ],
        "api_cost": "very high: several calls per historical game",
        "leakage_risk": "medium: all rolling summaries must stop before target game",
        "feature_ideas": (
            "four-factor matchup edges; paint/transition/second-chance style; "
            "assisted-shot and usage concentration"
        ),
        "priority": 4,
        "deployability_risk": "high: multi-endpoint completeness and request volume",
    },
    "player_availability_rotation": {
        "endpoints": [
            "PlayerGameLogs", "BoxScoreTraditionalV2", "BoxScoreAdvancedV2",
        ],
        "fields": [
            "PLAYER_ID", "MIN", "PTS", "PLUS_MINUS", "START_POSITION",
            "COMMENT", "USG_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING",
            "TS_PCT", "PIE",
        ],
        "api_cost": "medium-high: league player logs plus optional game boxes",
        "leakage_risk": (
            "high: availability and rotation baselines must use only games before "
            "target; DNP comments for target game are forbidden"
        ),
        "feature_ideas": (
            "top-player minutes/usage present; top-7 minutes share; rotation "
            "stability; missing star production; bench compression"
        ),
        "priority": 3,
        "deployability_risk": "medium-high: roster identity and injury timing",
    },
    "play_by_play_momentum": {
        "endpoints": ["PlayByPlayV2", "PlayByPlayV3"],
        "fields": [
            "PERIOD", "PCTIMESTRING", "SCORE", "SCOREMARGIN", "actionType",
            "subType", "scoreHome", "scoreAway", "description",
        ],
        "api_cost": "very high: one large call per game",
        "leakage_risk": (
            "high: target-game events cannot be used; only prior-game summaries "
            "may enter the next prediction"
        ),
        "feature_ideas": (
            "scoring runs; lead changes; quarter control; clutch margin; "
            "fourth-quarter control; timeout response where event sequence permits"
        ),
        "priority": 6,
        "deployability_risk": "high: parsing/version drift and large backfill",
    },
    "shot_profile": {
        "endpoints": ["ShotChartDetail"],
        "fields": [
            "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
            "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_ATTEMPTED_FLAG",
            "SHOT_MADE_FLAG", "PERIOD",
        ],
        "api_cost": "high: player/team/date parameterized requests",
        "leakage_risk": "medium-high: date windows must end before target game",
        "feature_ideas": (
            "rim/midrange/three frequency; corner-three rate; shot-quality proxy; "
            "profile stability; opponent profile allowed"
        ),
        "priority": 5,
        "deployability_risk": "high: expensive extraction and opponent attribution",
    },
    "hustle_effort": {
        "endpoints": ["HustleStatsBoxScore"],
        "fields": [
            "CONTESTED_SHOTS", "CONTESTED_SHOTS_2PT", "CONTESTED_SHOTS_3PT",
            "DEFLECTIONS", "CHARGES_DRAWN", "SCREEN_ASSISTS",
            "LOOSE_BALLS_RECOVERED", "OFF_BOXOUTS", "DEF_BOXOUTS", "BOX_OUTS",
        ],
        "api_cost": "very high: one specialized call per game",
        "leakage_risk": "medium: use only prior-game rolling effort summaries",
        "feature_ideas": (
            "effort differential; contest rate; deflections; loose-ball recovery; "
            "screen-assist and box-out stability"
        ),
        "priority": 8,
        "deployability_risk": "very high: uncertain older-season availability",
    },
    "schedule_rest_travel": {
        "endpoints": ["TeamGameLog/TeamGameLogs dates and MATCHUP"],
        "fields": ["GAME_DATE", "MATCHUP", "TEAM_ID", "GAME_ID"],
        "api_cost": "low: already available in cached team logs",
        "leakage_risk": "low for prior schedule; future schedule must be known pregame",
        "feature_ideas": (
            "rest days; back-to-back; home/away; travel-distance proxy; "
            "time-zone changes; Denver altitude indicator"
        ),
        "priority": 1,
        "deployability_risk": (
            "low for rest/home; medium for travel due to arena/geocode maintenance"
        ),
    },
}


def _endpoint_schema_fields() -> dict[str, list[str]]:
    classes = [
        endpoints.leaguegamelog.LeagueGameLog,
        endpoints.teamgamelog.TeamGameLog,
        endpoints.teamgamelogs.TeamGameLogs,
        endpoints.boxscoreadvancedv2.BoxScoreAdvancedV2,
        endpoints.boxscorefourfactorsv2.BoxScoreFourFactorsV2,
        endpoints.boxscoremiscv2.BoxScoreMiscV2,
        endpoints.boxscorescoringv2.BoxScoreScoringV2,
        endpoints.boxscoreusagev2.BoxScoreUsageV2,
        endpoints.playergamelogs.PlayerGameLogs,
        endpoints.boxscoretraditionalv2.BoxScoreTraditionalV2,
        endpoints.playbyplayv2.PlayByPlayV2,
        endpoints.playbyplayv3.PlayByPlayV3,
        endpoints.shotchartdetail.ShotChartDetail,
        endpoints.hustlestatsboxscore.HustleStatsBoxScore,
        endpoints.leaguedashteamstats.LeagueDashTeamStats,
    ]
    schemas = {}
    for endpoint_class in classes:
        fields = []
        for result_fields in endpoint_class.expected_data.values():
            fields.extend(result_fields)
        schemas[endpoint_class.__name__] = sorted(set(fields))
    return schemas


def _csv_stats(paths: list[Path], key_fields: list[str]) -> dict:
    if not paths:
        return {"files": 0, "rows": 0, "games": 0, "missingness": 1.0}
    rows = 0
    games = set()
    missing_cells = 0
    possible_cells = 0
    field_aliases = {
        "GAME_ID": ("GAME_ID", "Game_ID"),
        "TEAM_ID": ("TEAM_ID", "Team_ID"),
        "AST_TOV": ("AST_TOV", "AST_TO"),
    }
    for path in paths:
        frame = pd.read_csv(path)
        rows += len(frame)
        game_column = "GAME_ID" if "GAME_ID" in frame else "Game_ID"
        if game_column in frame:
            games.update(frame[game_column].astype(str).str.replace(r"\.0$", "", regex=True))
        available = []
        for field in key_fields:
            candidates = field_aliases.get(field, (field,))
            actual_field = next(
                (candidate for candidate in candidates if candidate in frame.columns),
                None,
            )
            if actual_field:
                available.append(actual_field)
        possible_cells += len(frame) * len(key_fields)
        missing_cells += len(frame) * (len(key_fields) - len(available))
        if available:
            missing_cells += int(frame[available].isna().sum().sum())
    missingness = missing_cells / possible_cells if possible_cells else 1.0
    return {
        "files": len(paths),
        "rows": rows,
        "games": len(games),
        "missingness": missingness,
    }


def _reference_games(raw_dir: Path, season: str) -> dict[str, int]:
    return {
        season_type: _csv_stats(
            list(raw_dir.glob(f"team_game_log_{season}_{season_type}_*.csv")),
            ["GAME_ID", "GAME_DATE", "MATCHUP", "PTS"],
        )["games"]
        for season_type in ("regular_season", "playoffs")
    }


def _family_cache_stats(
    family: str,
    season: str,
    raw_dir: Path,
    reference: dict[str, int],
) -> dict:
    fields = FAMILY_CONFIG[family]["fields"]
    if family in {"team_game_logs", "schedule_rest_travel"}:
        paths = [
            *raw_dir.glob(f"team_game_log_{season}_regular_season_*.csv"),
            *raw_dir.glob(f"team_game_log_{season}_playoffs_*.csv"),
        ]
        stats = _csv_stats(list(paths), fields)
        expected = reference["regular_season"] + reference["playoffs"]
        return {
            **stats,
            "game_coverage": stats["games"] / expected if expected else 0.0,
            "aggregate_rows": 0,
            "cache_source": "team_game_log_* regular-season and playoff CSVs",
            "granularity": "team-game",
        }
    if family == "team_advanced_box_scores":
        aggregate_paths = list(raw_dir.glob(f"team_advanced_{season}_*.csv"))
        stats = _csv_stats(aggregate_paths, fields)
        return {
            **stats,
            "games": 0,
            "game_coverage": 0.0,
            "aggregate_rows": stats["rows"],
            "cache_source": "team_advanced_* LeagueDashTeamStats aggregates",
            "granularity": "season aggregate only; no BoxScoreAdvancedV2 cache",
        }
    if family == "player_availability_rotation":
        paths = list(raw_dir.glob(f"player_stats_{season}_*.csv"))
        stats = _csv_stats(paths, ["PLAYER_ID", "TEAM_ID", "MIN", "PTS"])
        return {
            **stats,
            "game_coverage": 0.0,
            "aggregate_rows": stats["rows"],
            "cache_source": "player_stats_* pregame/season-average snapshots",
            "granularity": "player snapshot; no PlayerGameLogs cache",
        }
    return {
        "files": 0,
        "rows": 0,
        "games": 0,
        "missingness": 1.0,
        "game_coverage": 0.0,
        "aggregate_rows": 0,
        "cache_source": "none",
        "granularity": "endpoint schema only; historical cache absent",
    }


def build_feature_source_audit(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    seasons = seasons or AUDIT_SEASONS
    schemas = _endpoint_schema_fields()
    rows = []
    for season in seasons:
        reference = _reference_games(raw_dir, season)
        for family, config in FAMILY_CONFIG.items():
            stats = _family_cache_stats(family, season, raw_dir, reference)
            rows.append(
                {
                    "feature_family": family,
                    "season": season,
                    "endpoint_or_cache_source": "; ".join(config["endpoints"]),
                    "local_cache_source": stats["cache_source"],
                    "source_granularity": stats["granularity"],
                    "fields_available": ", ".join(config["fields"]),
                    "installed_schema_field_count": sum(
                        len(schemas.get(endpoint.split()[0], []))
                        for endpoint in config["endpoints"]
                    ),
                    "cache_file_count": stats["files"],
                    "cached_row_count": stats["rows"],
                    "cached_unique_game_count": stats["games"],
                    "reference_regular_season_games": reference["regular_season"],
                    "reference_playoff_games": reference["playoffs"],
                    "historical_game_coverage_pct": round(
                        stats["game_coverage"] * 100, 2
                    ),
                    "aggregate_or_snapshot_rows": stats["aggregate_rows"],
                    "key_field_missingness_pct": round(
                        stats["missingness"] * 100, 2
                    ),
                    "api_cost_rate_limit_risk": config["api_cost"],
                    "leakage_risk": config["leakage_risk"],
                    "feature_ideas": config["feature_ideas"],
                    "recommended_experiment_priority": config["priority"],
                    "deployability_risk": config["deployability_risk"],
                }
            )
    return pd.DataFrame(rows)


def build_family_priorities(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in audit.groupby("feature_family", sort=False):
        config = FAMILY_CONFIG[family]
        cached_seasons = group.loc[
            group["cache_file_count"].gt(0), "season"
        ].tolist()
        rows.append(
            {
                "priority_rank": config["priority"],
                "feature_family": family,
                "cached_seasons_count": len(cached_seasons),
                "cached_seasons": ", ".join(cached_seasons),
                "median_historical_game_coverage_pct": float(
                    group["historical_game_coverage_pct"].median()
                ),
                "total_cached_rows": int(group["cached_row_count"].sum()),
                "api_cost_rate_limit_risk": config["api_cost"],
                "leakage_risk": config["leakage_risk"],
                "deployability_risk": config["deployability_risk"],
                "next_experiment": config["feature_ideas"],
                "priority_rationale": _priority_rationale(family, group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["priority_rank", "feature_family"]
    ).reset_index(drop=True)


def _priority_rationale(family: str, group: pd.DataFrame) -> str:
    coverage = group["historical_game_coverage_pct"].median()
    if family == "team_game_logs":
        return "Complete local game-level history; cheapest leakage-safe experiment."
    if family == "schedule_rest_travel":
        return "Rest and venue features derive from complete local logs without new calls."
    if family == "team_advanced_box_scores":
        return (
            "High-value efficiency fields exist, but current caches are season aggregates; "
            "pilot a small game-box backfill before committing."
        )
    if family == "player_availability_rotation":
        return (
            "Player snapshots exist across seasons, but true game-level rotation and "
            "availability require a separate cached backfill."
        )
    if coverage == 0:
        return "No local historical game cache; first run a small endpoint coverage pilot."
    return "Partial local evidence; validate completeness before modeling."


def write_research_plan(
    priorities: pd.DataFrame,
    output_path: str | Path = DEFAULT_DOC_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NBA API Feature Research Plan",
        "",
        "This is a source and coverage audit only. It does not train or deploy a model.",
        "",
        "## Evidence Rules",
        "",
        "- Local CSV coverage is measured before considering new API calls.",
        "- Season aggregates do not count as game-level historical coverage.",
        "- Every future experiment must build features from games strictly before the target game.",
        "- Target-game DNP status, box scores, play-by-play, shots, and hustle data are leakage.",
        "",
        "## Recommended Order",
        "",
    ]
    for _, row in priorities.iterrows():
        lines.extend(
            [
                f"### {int(row['priority_rank'])}. {row['feature_family'].replace('_', ' ').title()}",
                "",
                f"- Cached seasons: {row['cached_seasons_count']}/10",
                f"- Median game coverage: {row['median_historical_game_coverage_pct']:.1f}%",
                f"- API cost: {row['api_cost_rate_limit_risk']}",
                f"- Leakage risk: {row['leakage_risk']}",
                f"- Deployability risk: {row['deployability_risk']}",
                f"- Next experiment: {row['next_experiment']}",
                f"- Rationale: {row['priority_rationale']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Suggested Pilots",
            "",
            "1. Reuse cached team logs for rest, schedule, and opponent-adjusted rolling form.",
            "2. Backfill a small stratified sample of BoxScoreAdvancedV2 games and verify field completeness by season.",
            "3. Build rotation features from cached player snapshots, then separately test PlayerGameLogs coverage.",
            "4. Pilot one playoff round for four factors, shot charts, play-by-play, and hustle before any full backfill.",
            "",
            "No source should move to modeling until its cache is reproducible, date-bounded, and coverage-tested.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_source_audit(
    *,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    doc_path: str | Path = DEFAULT_DOC_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = build_feature_source_audit(raw_dir)
    priorities = build_family_priorities(audit)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(
        processed_dir / "nba_api_feature_source_audit.csv", index=False
    )
    priorities.to_csv(
        processed_dir / "nba_api_feature_family_priorities.csv", index=False
    )
    write_research_plan(priorities, doc_path)
    return audit, priorities


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached NBA API feature sources")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--doc-path", default=DEFAULT_DOC_PATH)
    args = parser.parse_args()
    audit, priorities = run_feature_source_audit(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        doc_path=args.doc_path,
    )
    print(
        audit.groupby("feature_family")[
            ["cached_row_count", "historical_game_coverage_pct"]
        ].agg(["sum", "median"]).to_string()
    )
    print("\nPriorities:")
    print(priorities[["priority_rank", "feature_family", "priority_rationale"]].to_string(index=False))


if __name__ == "__main__":
    main()
