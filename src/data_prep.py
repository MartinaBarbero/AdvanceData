"""Builds the model table: one row = one driver in one race (2014-2026).

Sources
  1. Kaggle Formula 1 dataset (data/*.csv): results and lap times up to 2024
  2. Jolpica-F1 API (data/external/*.csv, downloaded by src/fetch_jolpica.py): 2025 and 2026

Every feature is computed with information available BEFORE the race it describes.
"""
from pathlib import Path

import pandas as pd

# paths are built from the position of this file, so they work from any working directory
ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "data"
EXTERNAL_DIR = ROOT / "data" / "external"
PROCESSED_DIR = ROOT / "data" / "processed"
MIN_TEAM_LAPS = 10               # fewer laps in a race: the team's median lap is not representative
NEW_YEARS = (2025, 2026)        # seasons that come from Jolpica instead of Kaggle

RESULT_COLUMNS = ["year", "round", "driver_id", "constructor_id", "constructor_name",
                  "grid", "position_order", "points"]

# teams that changed name or owner are treated as one team
LINEAGE = {
    "Force India": "Aston Martin lineage", "Racing Point": "Aston Martin lineage", "Aston Martin": "Aston Martin lineage",
    "Toro Rosso": "AlphaTauri lineage", "AlphaTauri": "AlphaTauri lineage", "RB F1 Team": "AlphaTauri lineage",
    "Lotus F1": "Renault lineage", "Renault": "Renault lineage", "Alpine F1 Team": "Renault lineage",
    "Sauber": "Sauber lineage", "Alfa Romeo": "Sauber lineage", "Audi": "Sauber lineage",
    "Marussia": "Marussia lineage", "Manor Marussia": "Marussia lineage",
}


def _read_kaggle_lookups():
    """Small Kaggle tables used to translate numeric ids into names."""
    races = pd.read_csv(KAGGLE_DIR / "races.csv")[["raceId", "year", "round"]]
    drivers = pd.read_csv(KAGGLE_DIR / "drivers.csv")[["driverId", "driverRef"]]
    constructors = pd.read_csv(KAGGLE_DIR / "constructors.csv")[["constructorId", "constructorRef", "name"]]
    return races, drivers, constructors


def _read_external(prefix):
    """Reads results_<year>.csv or laps_<year>.csv for the Jolpica seasons."""
    frames = []
    for year in NEW_YEARS:
        path = EXTERNAL_DIR / f"{prefix}_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run src/fetch_jolpica.py first (see README).")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def load_results():
    """Race results from both sources in one table, with the key race_id = year * 100 + round."""
    races, drivers, constructors = _read_kaggle_lookups()
    results = pd.read_csv(KAGGLE_DIR / "results.csv")

    kaggle = (results[["raceId", "driverId", "constructorId", "grid", "positionOrder", "points"]]
              .merge(races, on="raceId").merge(drivers, on="driverId").merge(constructors, on="constructorId")
              .rename(columns={"driverRef": "driver_id", "constructorRef": "constructor_id",
                               "name": "constructor_name", "positionOrder": "position_order"}))
    kaggle["grid"] = kaggle["grid"].replace(0, 20)      # pit-lane start counts as the last slot

    jolpica = _read_external("results")
    all_results = (pd.concat([kaggle[RESULT_COLUMNS], jolpica[RESULT_COLUMNS]], ignore_index=True)
                   .sort_values(["year", "round"]).reset_index(drop=True))
    all_results["race_id"] = all_results["year"] * 100 + all_results["round"]
    all_results["team"] = all_results["constructor_name"].map(LINEAGE).fillna(all_results["constructor_name"])
    return all_results


def load_laps():
    """Lap times from both sources: year, round, lap, driver_id, seconds, race_id."""
    races, drivers, _ = _read_kaggle_lookups()
    lap_times = pd.read_csv(KAGGLE_DIR / "lap_times.csv")
    kaggle = (lap_times.merge(races, on="raceId").merge(drivers, on="driverId")
              .rename(columns={"driverRef": "driver_id"}))
    kaggle["seconds"] = kaggle["milliseconds"] / 1000
    kaggle = kaggle[["year", "round", "lap", "driver_id", "seconds"]]

    all_laps = pd.concat([kaggle, _read_external("laps")], ignore_index=True)
    all_laps["race_id"] = all_laps["year"] * 100 + all_laps["round"]
    return all_laps


def add_team_pace(all_results, all_laps):
    """team_pace3 = average gap (%) to the fastest team over the team's previous 3 races."""
    # attach the team to every lap and drop lap 1 (standing start)
    laps = (all_laps[all_laps["lap"] > 1]
            .merge(all_results[["race_id", "driver_id", "team"]], on=["race_id", "driver_id"]))

    # median lap per team per race; a team with very few laps in a race (early retirements,
    # red flags) has no representative pace, so that team-race is left out
    tp = (laps.groupby(["race_id", "team"])
              .agg(median_lap=("seconds", "median"), n_laps=("seconds", "size"))
              .reset_index())
    tp = tp[tp["n_laps"] >= MIN_TEAM_LAPS].copy()

    # distance (%) from the fastest team of that race
    tp["pace_gap_pct"] = (tp["median_lap"] / tp.groupby("race_id")["median_lap"].transform("min") - 1) * 100

    # one row for every race the team took part in, also those without a usable pace
    # (for example a race run behind the safety car): the window then uses the races that have one
    team_races = (all_results[["race_id", "team"]].drop_duplicates()
                  .merge(tp[["race_id", "team", "pace_gap_pct"]], on=["race_id", "team"], how="left")
                  .sort_values("race_id"))

    # shift(1): the current race is excluded, so the value is known before the race
    team_races["team_pace3"] = (team_races.groupby("team")["pace_gap_pct"]
                                          .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))

    return all_results.merge(team_races[["race_id", "team", "team_pace3"]], on=["race_id", "team"], how="left")


def add_driver_features(all_results):
    """Target and driver features. Only races BEFORE the current one are used."""
    table = all_results.sort_values(["race_id", "driver_id"]).reset_index(drop=True)
    table["win"] = (table["position_order"] == 1).astype(int)          # target

    table["driver_wins_before"] = table.groupby("driver_id")["win"].cumsum() - table["win"]
    table["driver_team_wins_before"] = table.groupby(["driver_id", "team"])["win"].cumsum() - table["win"]
    table["driver_races_before"] = table.groupby("driver_id").cumcount()
    table["driver_form3"] = (table.groupby("driver_id")["points"]
                                  .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))
    return table


def build_model_table():
    """Runs the whole preparation and returns the table (nothing is written to disk)."""
    all_results = load_results()
    all_laps = load_laps()
    table = add_team_pace(all_results, all_laps)
    return add_driver_features(table)


def save_model_table(table):
    """Writes data/processed/model_table.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(PROCESSED_DIR / "model_table.csv", index=False)