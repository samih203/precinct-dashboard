"""
models/predictor.py
Trains gradient-boosted regression models to predict:
  1. Voter turnout (% of registered voters who vote)
  2. Democratic vote margin (dem_share - rep_share)

In production:
  - Augment with census tract data, early vote data, canvassing contact rates
  - Retrain on each new election cycle
  - Validate against out-of-sample precincts
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score


FEATURE_COLS = [
    "pct_dem_registered",
    "pct_rep_registered",
    "pct_npa_registered",
    "pct_hispanic",
    "pct_black",
    "pct_senior",
    "median_income",
    "registered_voters",
    "avg_turnout",
    "avg_dem_share",
    "turnout_trend",
    "margin_trend",
    "competitiveness",
]

PRECINCT_TYPE_MAP = {
    "Urban Core": 0, "Urban Fringe": 1, "Inner Suburb": 2, "Outer Suburb": 3,
    "Exurban": 4, "Rural": 5, "College Town": 6, "Retirement": 7,
    "Hispanic Majority": 8, "Mixed Suburban": 9,
}

COUNTY_MAP = {
    "Orange": 0, "Seminole": 1, "Osceola": 2, "Brevard": 3, "Volusia": 4,
}


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].copy()
    X["precinct_type_enc"] = df["precinct_type"].map(PRECINCT_TYPE_MAP).fillna(5)
    X["county_enc"] = df["county"].map(COUNTY_MAP).fillna(0)
    return X.values


def train_models(df: pd.DataFrame):
    X = prepare_features(df)
    y_turnout = df["2024_turnout"].values
    y_margin = df["2024_margin"].values

    turnout_model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ])

    margin_model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ))
    ])

    turnout_model.fit(X, y_turnout)
    margin_model.fit(X, y_margin)

    # Cross-validation scores
    to_cv = cross_val_score(turnout_model, X, y_turnout, cv=5, scoring="r2")
    mg_cv = cross_val_score(margin_model, X, y_margin, cv=5, scoring="r2")

    to_preds = turnout_model.predict(X)
    mg_preds = margin_model.predict(X)

    metrics = {
        "turnout_r2": float(np.mean(to_cv)),
        "turnout_mae": float(mean_absolute_error(y_turnout, to_preds)),
        "margin_r2": float(np.mean(mg_cv)),
        "margin_mae": float(mean_absolute_error(y_margin, mg_preds)),
    }

    # Feature importances
    feat_names = FEATURE_COLS + ["precinct_type_enc", "county_enc"]
    to_imp = dict(zip(feat_names, turnout_model.named_steps["gbr"].feature_importances_))
    mg_imp = dict(zip(feat_names, margin_model.named_steps["gbr"].feature_importances_))

    return turnout_model, margin_model, metrics, to_imp, mg_imp


def predict(df: pd.DataFrame, turnout_model, margin_model,
            national_env: float = 0.0, ground_game_boost: float = 0.0):
    """
    Generate predictions with optional scenario modifiers.

    national_env: float, shift in dem share relative to baseline (-1 to +1, e.g. 0.03 = +3pp national wave)
    ground_game_boost: float, turnout boost from canvassing/GOTV effort (0 to 0.10)
    """
    X = prepare_features(df)
    pred_turnout = np.clip(turnout_model.predict(X) + ground_game_boost, 0.20, 0.95)
    pred_margin = np.clip(margin_model.predict(X) + national_env * 2, -1.0, 1.0)
    pred_dem_share = np.clip((pred_margin + 1) / 2, 0.05, 0.95)
    pred_votes_cast = (df["registered_voters"].values * pred_turnout).astype(int)
    pred_dem_votes = (pred_votes_cast * pred_dem_share).astype(int)
    pred_rep_votes = pred_votes_cast - pred_dem_votes

    return pd.DataFrame({
        "precinct_id": df["precinct_id"].values,
        "pred_turnout": np.round(pred_turnout, 4),
        "pred_dem_share": np.round(pred_dem_share, 4),
        "pred_margin": np.round(pred_margin, 4),
        "pred_votes_cast": pred_votes_cast,
        "pred_dem_votes": pred_dem_votes,
        "pred_rep_votes": pred_rep_votes,
    })


def save_models(turnout_model, margin_model, path: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "turnout_model.pkl"), "wb") as f:
        pickle.dump(turnout_model, f)
    with open(os.path.join(path, "margin_model.pkl"), "wb") as f:
        pickle.dump(margin_model, f)


def load_models(path: str):
    with open(os.path.join(path, "turnout_model.pkl"), "rb") as f:
        turnout_model = pickle.load(f)
    with open(os.path.join(path, "margin_model.pkl"), "rb") as f:
        margin_model = pickle.load(f)
    return turnout_model, margin_model


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "precincts.csv")
    df = pd.read_csv(data_path)

    print("Training models...")
    tm, mm, metrics, to_imp, mg_imp = train_models(df)
    print(f"\nTurnout model  — R²: {metrics['turnout_r2']:.3f} | MAE: {metrics['turnout_mae']:.4f}")
    print(f"Margin  model  — R²: {metrics['margin_r2']:.3f} | MAE: {metrics['margin_mae']:.4f}")

    print("\nTop turnout predictors:")
    for k, v in sorted(to_imp.items(), key=lambda x: -x[1])[:5]:
        print(f"  {k:30s} {v:.3f}")

    model_path = os.path.join(os.path.dirname(__file__))
    save_models(tm, mm, model_path)
    print(f"\nModels saved → {model_path}")
