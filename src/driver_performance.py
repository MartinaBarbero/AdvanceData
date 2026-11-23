"""Driver performance over time: race wins per season of the most successful drivers."""
import prediction


def wins_by_driver_season(data, top=6):
    """Wins per season (columns) of the `top` drivers with most wins (rows).

    `data` is the model table: one row per driver and race, win = 1 for the winner.
    Seasons without a win are filled with 0. Rows are ordered by total wins.
    """
    wins = data[data["win"] == 1]
    top_ids = wins.groupby("driver_id").size().sort_values(ascending=False).head(top).index

    table = (wins[wins["driver_id"].isin(top_ids)]
             .groupby(["driver_id", "year"]).size()
             .unstack("year", fill_value=0)
             .reindex(top_ids))
    seasons = range(int(data["year"].min()), int(data["year"].max()) + 1)
    table = table.reindex(columns=seasons, fill_value=0)
    table.index = [prediction.driver_name(driver_id) for driver_id in table.index]
    return table