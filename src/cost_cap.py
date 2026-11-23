"""Second data source: official Formula 1 cost cap levels, one row per season.

The levels come from the FIA Financial Regulations and from Formula 1's own
explainer (see the source columns in data/external/cost_cap.csv). They are the
base amounts, before the adjustments for inflation and for the number of races.
"""
from pathlib import Path

import pandas as pd

import data_prep

COST_CAP_FILE = Path(data_prep.EXTERNAL_DIR) / "cost_cap.csv"
CAP_YEAR = 2021
REQUIRED_COLUMNS = ["year", "cost_cap_base_musd", "source", "source_url"]


def load_cost_cap(path=COST_CAP_FILE):
    """Read the cost cap table and check that it has the expected shape."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Cost cap file not found: {path}")
    cap = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in cap.columns]
    if missing:
        raise ValueError(f"The cost cap file is missing these columns: {missing}")
    if cap["year"].duplicated().any():
        raise ValueError("The cost cap file has more than one row for the same year.")
    if cap["year"].min() != CAP_YEAR:
        raise ValueError(f"The first cost cap season should be {CAP_YEAR}, found {cap['year'].min()}.")
    return cap.sort_values("year").reset_index(drop=True)


def add_cost_cap(data, cap=None):
    """Add cost_cap_base_musd to a table with a 'year' column (0 before the cap existed).

    Note: the 2026 level (215) is not comparable with the earlier years, because
    from 2026 the cap covers more types of cost.
    """
    cap = load_cost_cap() if cap is None else cap
    out = data.merge(cap[["year", "cost_cap_base_musd"]], on="year", how="left")
    out["cost_cap_base_musd"] = out["cost_cap_base_musd"].fillna(0)
    return out