"""Logistic regression, walk-forward evaluation and race-level bootstrap analyses.

All functions take the model table (see data_prep.py) as `data`.
Every season is predicted with a model trained on EARLIER seasons only (no random split).
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import PROCESSED_DIR

CAP_YEAR = 2021          # first season with the cost cap

FEATURES_6 = ["grid", "team_pace3", "driver_form3", "driver_wins_before",
              "driver_team_wins_before", "driver_races_before"]
MAIN_FEATURES = ["grid", "driver_form3", "driver_wins_before", "team_pace3"]      # project specification
FEATURE_SETS = {
    "6 features (current)": FEATURES_6,
    "4 features (project spec)": MAIN_FEATURES,
    "grid + driver form only": ["grid", "driver_form3"],
    "grid only": ["grid"],
}


def load_model_table(first_year=2014, last_year=2026):
    """Reads data/processed/model_table.csv for the chosen seasons."""
    path = PROCESSED_DIR / "model_table.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the data preparation first (see README).")
    data = pd.read_csv(path)
    data = data[data["year"].between(first_year, last_year)].copy()
    data["driver_form3"] = data["driver_form3"].fillna(0)      # debut race: no points history yet
    return data


def period_of(year):
    """Label of the season in the analysis."""
    if year < CAP_YEAR:
        return "pre-cap"
    if year <= 2024:
        return "post-cap"
    return "extension 2025" if year == 2025 else "2026 new rules"


def fit_model(train, feats):
    """Median imputation -> standardisation -> logistic regression, learned on the training seasons only."""
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=1000))
    return model.fit(train[feats], train["win"])


def walk_forward(data, feats, first_year=2017, last_year=2026):
    """One row per test season: model metrics and pole-sitter baseline metrics."""
    rows = []
    for year in range(first_year, last_year + 1):
        train, test = data[data["year"] < year], data[data["year"] == year]     # only earlier seasons
        p = fit_model(train, feats).predict_proba(test[feats])[:, 1]
        top_pick = test.assign(p=p).sort_values("p", ascending=False).groupby("race_id").head(1)

        # pole-sitter baseline: rates learned on the training seasons
        pole_rate = train.loc[train["grid"] == 1, "win"].mean()
        other_rate = train.loc[train["grid"] != 1, "win"].mean()
        p_base = np.where(test["grid"] == 1, pole_rate, other_rate)

        rows.append({
            "year": year,
            "period": period_of(year),
            "AUC": roc_auc_score(test["win"], p),
            "log_loss": log_loss(test["win"], p),
            "top_pick_hit": top_pick["win"].mean(),
            "AUC_pole": roc_auc_score(test["win"], -test["grid"]),      # better grid = higher score
            "log_loss_pole": log_loss(test["win"], p_base),
            "pole_hit": test.loc[test["grid"] == 1, "win"].mean(),
        })
    return pd.DataFrame(rows)


def ablation_table(data, feature_sets=FEATURE_SETS):
    """Mean AUC, log-loss and top-pick hit rate by period for each set of features."""
    frames = []
    for name, feats in feature_sets.items():
        res = walk_forward(data, feats)
        res["model"] = name
        frames.append(res)
    out = pd.concat(frames)
    return out.groupby(["model", "period"])[["AUC", "log_loss", "top_pick_hit"]].mean().round(3)


def top_pick_table(data, feats, year):
    """Weights of the model trained before `year`, and its top pick against winner and pole in each race."""
    model = fit_model(data[data["year"] < year], feats)
    weights = pd.Series(model[-1].coef_[0], index=feats).round(2)

    test = data[data["year"] == year]
    test = test.assign(p=model.predict_proba(test[feats])[:, 1])
    top = (test.sort_values("p", ascending=False).groupby("race_id").head(1)
           [["race_id", "driver_id", "p"]].rename(columns={"driver_id": "top_pick"}))
    winner = test[test["win"] == 1][["race_id", "driver_id"]].rename(columns={"driver_id": "winner"})
    pole = test[test["grid"] == 1][["race_id", "driver_id"]].rename(columns={"driver_id": "pole"})
    return weights, top.merge(winner, on="race_id").merge(pole, on="race_id").round(2)


def walk_forward_predictions(data, feats, first_year=2017, last_year=2024):
    """Out-of-sample probabilities for every race, each season predicted from earlier ones only."""
    parts = []
    for year in range(first_year, last_year + 1):
        train, test = data[data["year"] < year], data[data["year"] == year]
        p = fit_model(train, feats).predict_proba(test[feats])[:, 1]
        parts.append(test[["race_id", "year", "win"]].assign(p=p))
    return pd.concat(parts, ignore_index=True)


def bootstrap_diff(pred, n_boot=500, seed=0):
    """Post-cap minus pre-cap AUC and log-loss, resampling whole races inside each period."""
    rng = np.random.default_rng(seed)
    pre = pred[pred["year"] < CAP_YEAR].reset_index(drop=True)
    post = pred[pred["year"] >= CAP_YEAR].reset_index(drop=True)
    groups = {"pre": pre.groupby("race_id").indices, "post": post.groupby("race_id").indices}

    def metrics(d):
        return np.array([roc_auc_score(d["win"], d["p"]), log_loss(d["win"], d["p"])])

    def draw(d, g):
        ids = np.array(list(g.keys()))
        pick = rng.choice(ids, size=len(ids), replace=True)
        return d.iloc[np.concatenate([g[r] for r in pick])]

    point = metrics(post) - metrics(pre)
    boot = np.array([metrics(draw(post, groups["post"])) - metrics(draw(pre, groups["pre"]))
                     for _ in range(n_boot)])
    return pd.DataFrame({"diff": point,
                         "ci_low": np.quantile(boot, 0.025, axis=0),
                         "ci_high": np.quantile(boot, 0.975, axis=0)},
                        index=["AUC", "log_loss"]).round(3)


def _fit_coefs(Z, y):
    model = LogisticRegression(C=1e6, max_iter=2000).fit(Z, y)      # C very large = almost no penalty
    return pd.Series(model.coef_[0], index=Z.columns)


def interaction_table(d, feats, n_boot=300, seed=0):
    """Weights of each feature and how they change after the cap, with race-level bootstrap intervals."""
    d = d.reset_index(drop=True)
    post = (d["year"] >= CAP_YEAR).astype(int)
    X = d[feats].fillna(d[feats].median())
    Z = (X - X.mean()) / X.std()                        # standardised: weights are comparable
    for f in feats:
        Z[f + "_x_post"] = Z[f] * post                  # does the weight of f change after the cap?
    Z["post_cap"] = post
    y = d["win"]
    base = _fit_coefs(Z, y)
    rng = np.random.default_rng(seed)
    groups = d.groupby("race_id").indices
    race_ids = np.array(list(groups.keys()))
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(race_ids, size=len(race_ids), replace=True)     # resample whole races
        idx = np.concatenate([groups[r] for r in pick])
        boot.append(_fit_coefs(Z.iloc[idx], y.iloc[idx]))
    boot = pd.DataFrame(boot)
    return pd.DataFrame({"coef": base, "ci_low": boot.quantile(0.025),
                         "ci_high": boot.quantile(0.975)}).round(2)


def _row_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def baseline_gap_by_period(data, feats, first_year=2017, last_year=2026, n_boot=2000, seed=0):
    """Log-loss of the model minus log-loss of the pole-sitter baseline (negative = model better),
    with race-level bootstrap intervals for each period."""
    parts = []
    for year in range(first_year, last_year + 1):
        train, test = data[data["year"] < year], data[data["year"] == year]
        p = fit_model(train, feats).predict_proba(test[feats])[:, 1]
        pole_rate = train.loc[train["grid"] == 1, "win"].mean()          # baseline learned on training seasons
        other_rate = train.loc[train["grid"] != 1, "win"].mean()
        p_base = np.where(test["grid"] == 1, pole_rate, other_rate)
        parts.append(pd.DataFrame({"year": year, "race_id": test["race_id"].values,
                                   "gap": _row_loss(test["win"].values, p) - _row_loss(test["win"].values, p_base)}))
    loss = pd.concat(parts, ignore_index=True)
    loss["period"] = loss["year"].map(period_of)

    rng = np.random.default_rng(seed)
    out = []
    for period, d in loss.groupby("period"):
        by_race = d.groupby("race_id")["gap"].agg(["sum", "count"])
        s, c = by_race["sum"].to_numpy(), by_race["count"].to_numpy()
        boot = []
        for _ in range(n_boot):
            i = rng.integers(0, len(s), len(s))                          # resample whole races
            boot.append(s[i].sum() / c[i].sum())
        out.append({"period": period, "gap": s.sum() / c.sum(),
                    "ci_low": np.quantile(boot, 0.025), "ci_high": np.quantile(boot, 0.975)})
    return pd.DataFrame(out).round(4)