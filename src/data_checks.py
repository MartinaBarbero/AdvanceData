"""Basic checks on the raw Kaggle files: size, missing values, coverage by season."""
from pathlib import Path

import pandas as pd

import data_prep

RAW_FILES = ["constructor_standings", "races", "results", "lap_times", "constructors"]
MISSING_MARKER = r"\N"  # the Ergast/Kaggle files write missing values as \N


def _read_kaggle(name):
    return pd.read_csv(Path(data_prep.KAGGLE_DIR) / f"{name}.csv")


def check_raw_files(files=RAW_FILES):
    """Return two tables: rows and columns of each file, and the columns with missing values."""
    shapes, missing = [], []
    for name in files:
        data = _read_kaggle(name)
        shapes.append({"file": name, "rows": len(data), "columns": data.shape[1]})
        counts = (data == MISSING_MARKER).sum()
        for column, n in counts[counts > 0].items():
            missing.append({"file": name, "column": column, "missing_values": int(n),
                            "share_percent": round(100 * n / len(data), 1)})
    missing_columns = ["file", "column", "missing_values", "share_percent"]
    return (pd.DataFrame(shapes).set_index("file"),
            pd.DataFrame(missing, columns=missing_columns).set_index("file"))


def season_overview(start_year, end_year):
    """Number of races and of recorded laps in each season."""
    races = _read_kaggle("races")
    laps = _read_kaggle("lap_times")
    races = races[races["year"].between(start_year, end_year)]

    n_races = races.groupby("year")["round"].max().rename("races")
    n_laps = (laps.merge(races[["raceId", "year"]], on="raceId")
                  .groupby("year").size().rename("laps"))
    return pd.concat([n_races, n_laps], axis=1)