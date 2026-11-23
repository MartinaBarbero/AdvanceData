"""Figures and tables for the report. Each plot function returns a matplotlib figure;
save_figure() writes it to figures/ and save_table() writes a table to tables/.

Colours: blue = the model, orange = the pole-sitter baseline (validated pair); a single series is blue.
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"            # titles and values
INK_SOFT = "#52514e"       # axis labels and annotations
GRID = "#e6e5e1"           # hairlines, one shade off the surface
MUTED = "#b9b8b0"          # reference lines
BLUE = "#2a78d6"           # model / single series
ORANGE = "#eb6834"         # pole-sitter baseline

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
    "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False,
})


def save_figure(fig, filename):
    """Saves the figure as figures/<filename>."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=200, bbox_inches="tight")


def save_table(table, filename):
    """Saves a table as tables/<filename> (csv)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / filename)


def plot_metrics_by_season(results):
    """AUC and log-loss of the model and of the pole-sitter baseline, season by season."""
    fig, (ax_auc, ax_ll) = plt.subplots(2, 1, figsize=(8, 6.6), sharex=True)
    years = results["year"]

    panels = [(ax_auc, "AUC", "AUC_pole", "AUC (0.5 = chance, 1 = perfect)"),
              (ax_ll, "log_loss", "log_loss_pole", "Log-loss (lower is better)")]
    for ax, model_col, pole_col, ylabel in panels:
        ax.plot(years, results[model_col], color=BLUE, linewidth=1.8, marker="o", markersize=6,
                label="Logistic regression")
        ax.plot(years, results[pole_col], color=ORANGE, linewidth=1.8, marker="o", markersize=6,
                label="Pole-sitter baseline")
        ax.set_ylabel(ylabel)
        # cost cap starts in 2021, new technical rules in 2022
        ax.axvline(2020.5, color=MUTED, linewidth=0.9)
        ax.axvline(2021.5, color=MUTED, linewidth=0.9)

    # headroom in the top panel for the two labels, so they never cover the data
    low, high = ax_auc.get_ylim()
    ax_auc.set_ylim(low, high + 0.14 * (high - low))
    ax_auc.text(2020.4, 0.985, "cost cap from 2021 ", ha="right", va="top", fontsize=8, color=INK_SOFT,
                transform=ax_auc.get_xaxis_transform())
    ax_auc.text(2021.6, 0.985, " new technical rules 2022", ha="left", va="top", fontsize=8, color=INK_SOFT,
                transform=ax_auc.get_xaxis_transform())
    ax_ll.legend(loc="lower left", ncol=2)          # bottom-left of the lower panel is empty
    ax_auc.set_title("Out-of-sample performance by season (walk-forward)", loc="left")
    ax_ll.set_xlabel("Test season (model trained on all earlier seasons)")
    ax_ll.set_xticks(list(years))
    fig.tight_layout()
    return fig


def _forest(ax, labels, estimate, low, high, xlabel):
    """Point estimates with 95% intervals, one row per label (first label on top)."""
    ypos = list(range(len(labels)))[::-1]
    ax.axvline(0, color=MUTED, linewidth=1.0)
    ax.hlines(ypos, low, high, color=BLUE, linewidth=1.8)
    ax.plot(estimate, ypos, "o", color=BLUE, markersize=6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def plot_period_differences(diffs):
    """Post-cap minus pre-cap AUC and log-loss for each model, with race-level bootstrap intervals.
    `diffs` maps a model name to the table returned by modeling.bootstrap_diff."""
    names = list(diffs)
    fig, (ax_auc, ax_ll) = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    for ax, metric, label in [(ax_auc, "AUC", "Difference in AUC"), (ax_ll, "log_loss", "Difference in log-loss")]:
        est = [diffs[n].loc[metric, "diff"] for n in names]
        low = [diffs[n].loc[metric, "ci_low"] for n in names]
        high = [diffs[n].loc[metric, "ci_high"] for n in names]
        _forest(ax, names, est, low, high, label)
    ax_auc.set_title("Change in performance after the cost cap (2021-2024 minus 2017-2020)", loc="left")
    fig.text(0.5, -0.02, "Points: estimated difference. Bars: 95% bootstrap interval (races resampled). "
             "An interval containing 0 means no detectable change.", ha="center", fontsize=8, color=INK_SOFT)
    fig.tight_layout()
    return fig


def _nice_label(name):
    return (name.replace("_x_post", " (change after 2021)").replace("post_cap", "after 2021 (level shift)")
            .replace("_", " "))


def plot_coefficients(table):
    """Weights of the model and their change after 2021 (output of modeling.interaction_table)."""
    labels = [_nice_label(n) for n in table.index]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    _forest(ax, labels, table["coef"], table["ci_low"], table["ci_high"],
            "Weight in the logistic regression (log-odds per standard deviation)")
    ax.set_title("What the model uses, and how its weights change after the cost cap", loc="left")
    fig.tight_layout()
    return fig


def plot_baseline_gap(gap):
    """Log-loss of the model minus log-loss of the pole-sitter baseline, by period (modeling.baseline_gap_by_period)."""
    order = ["pre-cap", "post-cap", "extension 2025", "2026 new rules"]
    names = {"pre-cap": "2017-2020 (before the cap)", "post-cap": "2021-2024 (after the cap)",
             "extension 2025": "2025", "2026 new rules": "2026 (new technical rules)"}
    table = gap.set_index("period").loc[order]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    _forest(ax, [names[p] for p in order], table["gap"], table["ci_low"], table["ci_high"],
            "Model minus pole-sitter baseline, in log-loss (negative = model better)")
    ax.set_title("Does the model beat the pole-sitter baseline?", loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Exploratory figures (notebook 03). They only use columns of model_table.csv:
# year, race_id, team, grid, win, team_pace3.
# ---------------------------------------------------------------------------

def plot_winners_by_team(data, top=10):
    """Number of race wins per team (lineages merged), the teams with most wins."""
    wins = (data[data["win"] == 1].groupby("team").size().sort_values().tail(top))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(wins.index, wins.values, color=BLUE, height=0.65)
    ax.set_xlabel("Race wins")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Race wins by team, {int(data['year'].min())}-{int(data['year'].max())}", loc="left")
    return fig


def plot_grid_vs_win_rate(data, max_grid=10):
    """Share of starts ending in a win, by grid slot, before and after the cost cap."""
    d = data[data["grid"] <= max_grid]
    pre, post = d[d["year"] <= 2020], d[d["year"] >= 2021]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for part, color, label in [(pre, BLUE, "2014-2020 (before the cap)"),
                               (post, ORANGE, f"2021-{int(data['year'].max())} (after the cap)")]:
        rate = part.groupby("grid")["win"].mean() * 100
        ax.plot(rate.index, rate.values, color=color, linewidth=1.8, marker="o", markersize=6, label=label)
    ax.set_xlabel("Starting grid position")
    ax.set_ylabel("Starts that ended in a win (%)")
    ax.set_xticks(range(1, max_grid + 1))
    ax.legend()
    ax.set_title("Chance of winning by starting position", loc="left")
    return fig


def plot_team_pace_over_time(data, n_teams=4):
    """Recent pace gap to the fastest team (team_pace3), one panel per team, race by race."""
    top_teams = (data[data["win"] == 1].groupby("team").size().sort_values(ascending=False)
                 .head(n_teams).index)
    d = data.drop_duplicates(["race_id", "team"]).copy()
    races_in_year = d.groupby("year")["race_id"].transform("nunique")
    d["season_time"] = d["year"] + (d["race_id"] % 100 - 1) / races_in_year
    top = d["team_pace3"].quantile(0.98)          # a few extreme early values would squash the lines

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    for ax, team in zip(axes.ravel(), top_teams):
        part = d[d["team"] == team].sort_values("season_time")
        ax.plot(part["season_time"], part["team_pace3"], color=BLUE, linewidth=1.4)
        ax.axvline(2021, color=MUTED, linewidth=0.9)
        ax.set_title(team, loc="left", fontsize=10)
        ax.set_ylim(0, top)
    for ax in axes[1]:
        ax.set_xlabel("Season")
    for ax in axes[:, 0]:
        ax.set_ylabel("Pace gap to fastest team (%)")
    fig.suptitle("Recent race pace of the most successful teams (lower = faster)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_probability_ranking(prediction, title, top=10):
    """Predicted probability of winning for the most likely drivers of one race (output of prediction.predict_race)."""
    best = prediction.head(top).iloc[::-1]                     # most likely driver at the top
    labels = [f"{d} ({t})" for d, t in zip(best["driver"], best["team"])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(labels, best["win_probability"] * 100, color=BLUE, height=0.65)
    ax.set_xlabel("Predicted probability of winning (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    return fig


def _style_season_axis(ax, years, cap_year):
    """Common look of the season charts: light grid, cost cap marker, one tick per season."""
    ax.axvline(cap_year - 0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(list(years))
    ax.set_xlabel("Season")
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def plot_competitiveness_gap(table, cap_year=2021):
    """Spread of the final team points per race, one point per season."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table.index, table["std_per_race"], marker="o", color=BLUE, linewidth=2)
    ax.set_ylim(0, table["std_per_race"].max() * 1.2)
    _style_season_axis(ax, table.index, cap_year)
    ax.text(cap_year - 0.4, ax.get_ylim()[1] * 0.97, "Cost cap from 2021",
            color="grey", fontsize=9, va="top")
    ax.set_ylabel("Std. dev. of final team points, per race")
    ax.set_title("How spread out the teams are in the standings")
    return fig


def plot_pace_gap_by_group(table, top_label, other_label, cap_year=2021):
    """Recent pace gap of the strongest teams against all the others, by season."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(table.index, table[top_label], marker="o", color=BLUE, linewidth=2, label=top_label)
    ax.plot(table.index, table[other_label], marker="o", color="#eb6834", linewidth=2, label=other_label)
    ax.set_ylim(bottom=0)
    _style_season_axis(ax, table.index, cap_year)
    ax.set_ylabel("Recent pace gap to the fastest team (%)")
    ax.set_title("Pace gap of the strongest teams and of the rest")
    ax.legend(frameon=False)
    return fig