"""Season-level competitiveness: how spread out the teams are in the standings
and how large the pace gap is between the strongest teams and the rest.

Reads the raw Kaggle files for the standings and the model table for the pace.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import data_prep

START_YEAR = 2014
END_YEAR = 2024
CAP_YEAR = 2021

TOP_LABEL = "Three strongest teams before 2021"
OTHER_LABEL = "All other teams"


def _read_kaggle(name):
    """Read one raw Kaggle csv file."""
    return pd.read_csv(Path(data_prep.KAGGLE_DIR) / f"{name}.csv")


def load_final_standings(start_year=START_YEAR, end_year=END_YEAR):
    """Final constructor points and position of every team in every season."""
    standings = _read_kaggle("constructor_standings")
    races = _read_kaggle("races")[["raceId", "year", "round"]]
    constructors = _read_kaggle("constructors")[["constructorId", "name"]]

    data = standings.merge(races, on="raceId")
    data = data[data["year"].between(start_year, end_year)]
    # The standings are published after every race: keep the last one of each season.
    final = data.sort_values("round").groupby(["year", "constructorId"]).tail(1)
    final = final.merge(constructors, on="constructorId").rename(columns={"name": "team"})
    return final[["year", "team", "points", "position"]].sort_values(["year", "position"]).reset_index(drop=True)


def competitiveness_table(final_standings):
    """Spread of the final points in each season.

    The spread is divided by the number of races, so that the short 2020 season
    can be compared with the others.
    """
    races = _read_kaggle("races")
    years = final_standings["year"]
    races = races[races["year"].between(years.min(), years.max())]
    n_races = races.groupby("year")["round"].max().rename("num_races")

    table = final_standings.groupby("year")["points"].agg(std="std", min="min", max="max", mean="mean")
    table["range"] = table["max"] - table["min"]
    table = table.join(n_races)
    table["std_per_race"] = table["std"] / table["num_races"]
    table["range_per_race"] = table["range"] / table["num_races"]
    return table.round(2)


def strongest_teams(final_standings, cap_year=CAP_YEAR, how_many=3):
    """The teams with the highest average final points before the cost cap."""
    before_cap = final_standings[final_standings["year"] < cap_year]
    average_points = before_cap.groupby("team")["points"].mean().sort_values(ascending=False)
    return average_points.head(how_many).index.tolist()


def pace_gap_by_group(model_table, top_teams):
    """Average recent pace gap (%) of the strongest teams and of all the others, by season.

    Uses team_pace3 from the model table, so it only contains information
    available before each race. One row per team and race is used (not per driver).
    """
    missing = set(top_teams) - set(model_table["team"])
    if missing:
        raise ValueError(f"These team names are not in the model table: {sorted(missing)}")

    team_races = model_table.drop_duplicates(["race_id", "team"]).dropna(subset=["team_pace3"]).copy()
    team_races["group"] = np.where(team_races["team"].isin(top_teams), TOP_LABEL, OTHER_LABEL)

    table = team_races.groupby(["year", "group"])["team_pace3"].mean().unstack()
    table["difference"] = table[OTHER_LABEL] - table[TOP_LABEL]
    return table.round(2)