from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("reports/.matplotlib").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .nba_data import TEAM_STRENGTH_COLUMNS


def save_team_stat_charts(team_stats: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    chart_paths: list[Path] = []

    rating_plot = output_dir / "offense_defense_net_rating.png"
    plt.figure(figsize=(11, 7))
    scatter = sns.scatterplot(
        data=team_stats,
        x="OFF_RATING",
        y="DEF_RATING",
        size="NET_RATING",
        hue="NET_RATING",
        palette="vlag",
        sizes=(60, 280),
    )
    for _, row in team_stats.iterrows():
        scatter.text(row["OFF_RATING"] + 0.03, row["DEF_RATING"] + 0.03, row["TEAM_ABBREVIATION"], fontsize=8)
    plt.title("Team Offensive vs Defensive Rating")
    plt.tight_layout()
    plt.savefig(rating_plot, dpi=160)
    plt.close()
    chart_paths.append(rating_plot)

    corr_plot = output_dir / "team_metric_correlation.png"
    plt.figure(figsize=(9, 7))
    sns.heatmap(team_stats[TEAM_STRENGTH_COLUMNS].corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Team Metric Correlation")
    plt.tight_layout()
    plt.savefig(corr_plot, dpi=160)
    plt.close()
    chart_paths.append(corr_plot)

    net_rating_plot = output_dir / "net_rating_rank.png"
    ranked = team_stats.sort_values("NET_RATING", ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=ranked,
        x="NET_RATING",
        y="TEAM_ABBREVIATION",
        hue="TEAM_ABBREVIATION",
        palette="viridis",
        legend=False,
    )
    plt.title("Net Rating by Team")
    plt.tight_layout()
    plt.savefig(net_rating_plot, dpi=160)
    plt.close()
    chart_paths.append(net_rating_plot)

    return chart_paths
