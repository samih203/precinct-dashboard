"""
generate_data.py
Creates realistic synthetic precinct data for Central Florida congressional districts.
In production, replace with real data from:
  - Florida Division of Elections: dos.myflorida.com/elections
  - Redistricting: flsenate.gov/Redistricting
  - Census: census.gov/geographies/mapping-files
"""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

COUNTIES = {
    "Orange": {"center": (28.538, -81.379), "precincts": 30, "lean": "D"},
    "Seminole": {"center": (28.717, -81.209), "precincts": 20, "lean": "R"},
    "Osceola": {"center": (28.167, -81.120), "precincts": 15, "lean": "D"},
    "Brevard": {"center": (28.210, -80.742), "precincts": 12, "lean": "R"},
    "Volusia": {"center": (29.028, -81.048), "precincts": 10, "lean": "R"},
}

ELECTIONS = [
    {"year": 2016, "type": "Presidential", "national_dem_share": 48.2},
    {"year": 2018, "type": "Midterm", "national_dem_share": 53.4},
    {"year": 2020, "type": "Presidential", "national_dem_share": 51.3},
    {"year": 2022, "type": "Midterm", "national_dem_share": 47.8},
    {"year": 2024, "type": "Presidential", "national_dem_share": 48.5},
]

PRECINCT_TYPES = {
    "Urban Core":         {"turnout_base": 0.68, "dem_base": 0.72, "variance": 0.04},
    "Urban Fringe":       {"turnout_base": 0.63, "dem_base": 0.62, "variance": 0.05},
    "Inner Suburb":       {"turnout_base": 0.67, "dem_base": 0.52, "variance": 0.06},
    "Outer Suburb":       {"turnout_base": 0.70, "dem_base": 0.45, "variance": 0.05},
    "Exurban":            {"turnout_base": 0.65, "dem_base": 0.38, "variance": 0.06},
    "Rural":              {"turnout_base": 0.60, "dem_base": 0.32, "variance": 0.07},
    "College Town":       {"turnout_base": 0.55, "dem_base": 0.66, "variance": 0.08},
    "Retirement":         {"turnout_base": 0.76, "dem_base": 0.46, "variance": 0.05},
    "Hispanic Majority":  {"turnout_base": 0.58, "dem_base": 0.61, "variance": 0.07},
    "Mixed Suburban":     {"turnout_base": 0.65, "dem_base": 0.50, "variance": 0.06},
}

def generate_precincts():
    rows = []
    precinct_id = 1

    for county, meta in COUNTIES.items():
        lat_c, lon_c = meta["center"]
        n = meta["precincts"]
        lean = meta["lean"]

        type_pool = list(PRECINCT_TYPES.keys())
        if lean == "D":
            weights = [4, 3, 2, 2, 1, 1, 2, 1, 3, 2]
        else:
            weights = [1, 2, 3, 4, 3, 2, 1, 3, 1, 2]
        weights = [w / sum(weights) for w in weights]

        for i in range(n):
            ptype = np.random.choice(type_pool, p=weights)
            params = PRECINCT_TYPES[ptype]

            # Geography: scatter precincts around county center
            lat = lat_c + np.random.uniform(-0.35, 0.35)
            lon = lon_c + np.random.uniform(-0.35, 0.35)

            # Demographics
            reg_voters = int(np.random.lognormal(7.8, 0.5))
            reg_voters = max(500, min(reg_voters, 8000))
            pct_dem_reg = np.clip(params["dem_base"] + np.random.normal(0, 0.08), 0.15, 0.85)
            pct_rep_reg = np.clip(1 - pct_dem_reg - np.random.uniform(0.05, 0.15), 0.10, 0.75)
            pct_npa_reg = max(0.05, 1 - pct_dem_reg - pct_rep_reg)

            pct_hispanic = np.clip(
                0.25 + 0.50 * (ptype == "Hispanic Majority") + np.random.normal(0, 0.08), 0.02, 0.85
            )
            pct_black = np.clip(np.random.beta(1.5, 8), 0.01, 0.55)
            pct_senior = np.clip(
                0.35 + 0.25 * (ptype == "Retirement") + np.random.normal(0, 0.08), 0.08, 0.65
            )
            median_income = int(np.clip(
                45000 + 25000 * (ptype in ("Inner Suburb", "Outer Suburb", "Retirement")) +
                np.random.normal(0, 12000), 22000, 140000
            ))

            base_row = {
                "precinct_id": f"P{precinct_id:04d}",
                "county": county,
                "precinct_name": f"{county}-{i+1:03d}",
                "precinct_type": ptype,
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "registered_voters": reg_voters,
                "pct_dem_registered": round(pct_dem_reg, 4),
                "pct_rep_registered": round(pct_rep_reg, 4),
                "pct_npa_registered": round(pct_npa_reg, 4),
                "pct_hispanic": round(pct_hispanic, 4),
                "pct_black": round(pct_black, 4),
                "pct_senior": round(pct_senior, 4),
                "median_income": median_income,
            }

            # Generate election results for each cycle
            for election in ELECTIONS:
                year = election["year"]
                etype = election["type"]
                nat_dem = election["national_dem_share"] / 100

                # Turnout varies by election type and precinct characteristics
                midterm_penalty = -0.08 if etype == "Midterm" else 0.0
                senior_boost = 0.06 * pct_senior
                registration_density = reg_voters / 4000
                urban_boost = 0.03 if ptype in ("Urban Core", "Urban Fringe") else 0.0

                turnout = np.clip(
                    params["turnout_base"] + midterm_penalty + senior_boost + urban_boost +
                    np.random.normal(0, params["variance"]),
                    0.25, 0.92
                )

                # Dem vote share: anchored to registration + national environment
                nat_effect = (nat_dem - 0.50) * 0.4
                dem_share = np.clip(
                    params["dem_base"] + nat_effect + np.random.normal(0, params["variance"]),
                    0.05, 0.95
                )

                votes_cast = int(reg_voters * turnout)
                dem_votes = int(votes_cast * dem_share)
                rep_votes = votes_cast - dem_votes

                base_row[f"{year}_turnout"] = round(turnout, 4)
                base_row[f"{year}_votes_cast"] = votes_cast
                base_row[f"{year}_dem_votes"] = dem_votes
                base_row[f"{year}_rep_votes"] = rep_votes
                base_row[f"{year}_dem_share"] = round(dem_share, 4)
                base_row[f"{year}_margin"] = round(dem_share - (1 - dem_share), 4)

            rows.append(base_row)
            precinct_id += 1

    return pd.DataFrame(rows)


def compute_features(df):
    """Derive features used in the ML model."""
    years = [e["year"] for e in ELECTIONS]

    df["avg_turnout"] = df[[f"{y}_turnout" for y in years]].mean(axis=1)
    df["avg_dem_share"] = df[[f"{y}_dem_share" for y in years]].mean(axis=1)
    df["avg_margin"] = df[[f"{y}_margin" for y in years]].mean(axis=1)
    df["turnout_trend"] = df[f"{years[-1]}_turnout"] - df[f"{years[0]}_turnout"]
    df["margin_trend"] = df[f"{years[-1]}_margin"] - df[f"{years[0]}_margin"]

    # Competitiveness score (0=safe D, 1=safe R, 0.5=tossup)
    df["competitiveness"] = df["avg_margin"].apply(lambda x: 1 - abs(x))

    return df


if __name__ == "__main__":
    print("Generating precinct data...")
    df = generate_precincts()
    df = compute_features(df)
    out_path = os.path.join(os.path.dirname(__file__), "precincts.csv")
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} precincts → {out_path}")

    summary = df.groupby("county").agg(
        precincts=("precinct_id", "count"),
        total_voters=("registered_voters", "sum"),
        avg_turnout_2024=("2024_turnout", "mean"),
        avg_dem_share_2024=("2024_dem_share", "mean"),
    )
    print("\nCounty summary (2024):")
    print(summary.to_string())
