"""Predicted probability of winning a race, using only information known before it starts.

For a chosen race the model is trained on the seasons BEFORE that race's season (the same rule as the
walk-forward evaluation), and applied to the drivers of that race. The variables of each driver were
built with previous races only (see data_prep.py), so nothing from the race itself is used.
"""
import modeling

FIRST_TEST_YEAR = 2017      # needs at least three earlier seasons (2014-2016) to train on


def driver_name(driver_id):
    """Readable name from the driver id, for example max_verstappen -> Max Verstappen."""
    return str(driver_id).replace("_", " ").title()


def list_races(data):
    """One row per race that can be predicted: year, round, race_id."""
    races = (data[data["year"] >= FIRST_TEST_YEAR][["year", "round", "race_id"]]
             .drop_duplicates().sort_values("race_id").reset_index(drop=True))
    return races


def predict_race(data, race_id, feats=None):
    """Probabilities of winning for every driver of one race, sorted from most to least likely.

    Columns: driver (readable name), driver_id, team, the model variables, win_probability (from the model) and
    race_share (the same probabilities rescaled to sum to 1, since a race has exactly one winner).
    """
    feats = list(feats or modeling.MAIN_FEATURES)
    race = data[data["race_id"] == race_id]
    if race.empty:
        raise ValueError(f"Race {race_id} is not in the data.")
    year = int(race["year"].iloc[0])
    if year < FIRST_TEST_YEAR:
        raise ValueError(f"Predictions start in {FIRST_TEST_YEAR}: earlier seasons are needed to train the model.")

    train = data[data["year"] < year]                       # only earlier seasons
    model = modeling.fit_model(train, feats)

    table = race[["driver_id", "team"] + feats].copy()
    table.insert(0, "driver", table["driver_id"].map(driver_name))
    table["win_probability"] = model.predict_proba(race[feats])[:, 1]
    table["race_share"] = table["win_probability"] / table["win_probability"].sum()
    table = table.sort_values("win_probability", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def check_entry(prediction, driver_id, team):
    """Returns the row of this driver and team in the race, or raises a clear error if they did not race together."""
    row = prediction[(prediction["driver_id"] == driver_id) & (prediction["team"] == team)]
    if row.empty:
        raise ValueError("This driver and team combination did not take part in the selected race.")
    return row.iloc[0]