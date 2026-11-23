"""Check of the research question taken literally: a model trained ONLY on seasons before the cost cap
(2014-2020) and never updated, applied to 2021-2024.

The walk-forward evaluation retrains every season, so from 2021 on it also learns from post-cap races.
Here the model is frozen at 2020 and compared with the walk-forward model on the same test races.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from modeling import CAP_YEAR, fit_model


def frozen_and_walk_forward_predictions(data, feats, last_train_year=CAP_YEAR - 1, last_year=2024):
    """Probabilities for every race of the test seasons from two models:
    p_frozen: trained once on seasons up to `last_train_year`, never updated;
    p_walk:   trained again each season on all earlier seasons (the walk-forward model)."""
    frozen = fit_model(data[data["year"] <= last_train_year], feats)
    parts = []
    for year in range(last_train_year + 1, last_year + 1):
        test = data[data["year"] == year]
        walk = fit_model(data[data["year"] < year], feats)          # only earlier seasons
        parts.append(test[["race_id", "year", "win"]].assign(
            p_frozen=frozen.predict_proba(test[feats])[:, 1],
            p_walk=walk.predict_proba(test[feats])[:, 1]))
    return pd.concat(parts, ignore_index=True)


def _metrics(d):
    """AUC and log-loss of both models on the rows of d."""
    return {"AUC_frozen": roc_auc_score(d["win"], d["p_frozen"]),
            "AUC_walk": roc_auc_score(d["win"], d["p_walk"]),
            "log_loss_frozen": log_loss(d["win"], d["p_frozen"]),
            "log_loss_walk": log_loss(d["win"], d["p_walk"])}


def frozen_vs_walk_forward(data, feats, n_boot=1000, seed=0):
    """Two tables: metrics by test season (plus all seasons together), and the difference
    frozen minus walk-forward with a 95% bootstrap interval that resamples whole races."""
    pred = frozen_and_walk_forward_predictions(data, feats)

    by_year = pd.DataFrame({year: _metrics(d) for year, d in pred.groupby("year")}).T
    by_year.loc["all"] = _metrics(pred)

    rng = np.random.default_rng(seed)
    groups = pred.groupby("race_id").indices
    race_ids = np.array(list(groups.keys()))

    def diff(d):
        m = _metrics(d)
        return np.array([m["AUC_frozen"] - m["AUC_walk"], m["log_loss_frozen"] - m["log_loss_walk"]])

    point = diff(pred)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(race_ids, size=len(race_ids), replace=True)        # resample whole races
        boot.append(diff(pred.iloc[np.concatenate([groups[r] for r in pick])]))
    boot = np.array(boot)
    difference = pd.DataFrame({"diff": point,
                               "ci_low": np.quantile(boot, 0.025, axis=0),
                               "ci_high": np.quantile(boot, 0.975, axis=0)},
                              index=["AUC", "log_loss"])
    return by_year.round(3), difference.round(3)