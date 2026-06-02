"""
data/ingest_real_data.py
════════════════════════════════════════════════════════════════════════════════
BallotBase — Real Florida Election Data Ingestion Pipeline
════════════════════════════════════════════════════════════════════════════════

Downloads real precinct-level election results from two sources:

  PRIMARY:   Florida Division of Elections (FL DOS)
             https://dos.fl.gov/elections/data-statistics/elections-data/precinct-level-election-results/
             Tab-delimited .txt files inside zip archives — the official source.

  FALLBACK:  OpenElections Florida (pre-cleaned CSVs)
             https://github.com/openelections/openelections-data-fl
             Standardized CSVs, easier to parse, slightly slower to update.

Then fetches:
  REGISTRATION: FL DOS voter registration by precinct
  DEMOGRAPHICS: US Census ACS 5-year estimates (block group level)

Outputs a single precincts.csv that BallotBase (app.py) reads directly.

USAGE:
  python data/ingest_real_data.py

  # Or target specific counties only:
  python data/ingest_real_data.py --counties Orange Seminole Osceola

  # Or use OpenElections fallback:
  python data/ingest_real_data.py --source openelections
════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import io
import zipfile
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COUNTIES = ["Orange", "Seminole", "Osceola", "Brevard", "Volusia"]

COUNTY_FIPS = {
    "Orange":   "095",
    "Seminole": "117",
    "Osceola":  "097",
    "Brevard":  "009",
    "Volusia":  "127",
}

# FL DOS zip file URLs — one per general election year
# Format: https://dos.fl.gov/media/{id}/filename.zip
# These are the real URLs from the FL DOS precinct results page.
# If a URL breaks, go to:
#   dos.fl.gov/elections/data-statistics/elections-data/precinct-level-election-results/
# and find the updated zip link for that year's general election.
FL_DOS_URLS = {
    2024: "https://dos.fl.gov/media/ibqpxcb2/20g_precinct_electionresults.zip",
    2022: "https://dos.fl.gov/media/zc3fmfpb/22g_precinct_electionresults.zip",
    2020: "https://dos.fl.gov/media/cqxjrcub/20g_precinct_electionresults.zip",
    2018: "https://dos.fl.gov/media/pjxjl2ek/18g_precinct_electionresults.zip",
    2016: "https://dos.fl.gov/media/ncdbkhpc/16g_precinct_electionresults.zip",
}

# OpenElections fallback — raw CSV files on GitHub
# These are pre-cleaned and cover the same elections.
OPENELECTIONS_BASE = (
    "https://raw.githubusercontent.com/openelections/openelections-data-fl/master"
)
OPENELECTIONS_FILES = {
    2024: "2024/20241105__fl__general__precinct.csv",
    2022: "2022/20221108__fl__general__precinct.csv",
    2020: "2020/20201103__fl__general__precinct.csv",
    2018: "2018/20181106__fl__general__precinct.csv",
    2016: "2016/20161108__fl__general__precinct.csv",
}

# FL DOS voter registration file (updated monthly)
VOTER_REG_URL = (
    "https://dos.fl.gov/media/uflnxbfz/statewide_voterregistration_byparty_byprecinct.zip"
)

# US Census ACS 5-year API (free, no key required for basic variables)
CENSUS_API = "https://api.census.gov/data/2022/acs/acs5"

ELECTIONS = list(FL_DOS_URLS.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
def download(url: str, label: str) -> bytes:
    """Download a URL with progress indication."""
    print(f"  Downloading {label}...")
    try:
        r = requests.get(url, timeout=60, headers={"User-Agent": "BallotBase/1.0"})
        r.raise_for_status()
        size_kb = len(r.content) / 1024
        print(f"    ✓ {size_kb:.0f} KB received")
        return r.content
    except requests.RequestException as e:
        print(f"    ✗ Failed: {e}")
        return None


def read_zip_txt(content: bytes) -> pd.DataFrame:
    """
    Unzip FL DOS precinct results and parse the tab-delimited .txt file inside.
    FL DOS zip files contain one statewide .txt file.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        txt_files = [f for f in z.namelist() if f.endswith(".txt")]
        if not txt_files:
            raise ValueError("No .txt file found in zip")
        with z.open(txt_files[0]) as f:
            # FL DOS uses tab-delimited, latin-1 encoding
            df = pd.read_csv(f, sep="\t", encoding="latin-1", dtype=str)
    return df


# ── FL DOS Ingestion ──────────────────────────────────────────────────────────
def parse_fldos_year(content: bytes, year: int,
                     target_counties: list) -> pd.DataFrame:
    """
    Parse one FL DOS precinct results zip file.

    FL DOS column names vary slightly by year. Common schema:
      County, Precinct, Contest Name, Choice, Party Code, Vote Total
      (or: CountyName, PrecinctName, RaceName, CandidateName, PartyCode, CanVotes)

    We normalize to a standard set of columns.
    """
    df = read_zip_txt(content)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Normalize column names across years
    rename_map = {
        "countyname":       "county",
        "county_name":      "county",
        "precinctname":     "precinct",
        "precinct_name":    "precinct",
        "racename":         "race",
        "contest_name":     "race",
        "contest":          "race",
        "candidatename":    "candidate",
        "choice":           "candidate",
        "partycode":        "party",
        "party_code":       "party",
        "canvotes":         "votes",
        "vote_total":       "votes",
        "totalvotes":       "votes",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Filter to target counties
    df["county"] = df["county"].str.strip().str.title()
    df = df[df["county"].isin(target_counties)].copy()

    # Filter to US House races only (congressional)
    df = df[df["race"].str.contains(
        "United States Representative|U\.S\. Representative|Congress",
        case=False, na=False
    )].copy()

    # Clean votes
    df["votes"] = pd.to_numeric(df["votes"].str.replace(",", ""), errors="coerce").fillna(0).astype(int)
    df["party"] = df["party"].str.strip().str.upper()
    df["precinct"] = df["precinct"].str.strip()
    df["year"] = year

    return df[["county", "precinct", "race", "candidate", "party", "votes", "year"]]


def aggregate_fldos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot party-level vote rows into one row per precinct with
    dem_votes, rep_votes, votes_cast, turnout columns.
    """
    # Sum by county + precinct + party
    agg = df.groupby(["county", "precinct", "year", "party"])["votes"].sum().reset_index()

    # Pivot to wide: one column per party
    pivot = agg.pivot_table(
        index=["county", "precinct", "year"],
        columns="party",
        values="votes",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    pivot["dem_votes"] = pivot.get("DEM", pd.Series(0, index=pivot.index))
    pivot["rep_votes"] = pivot.get("REP", pd.Series(0, index=pivot.index))
    pivot["votes_cast"] = pivot["dem_votes"] + pivot["rep_votes"]
    pivot["dem_share"]  = (pivot["dem_votes"] / pivot["votes_cast"].clip(lower=1)).round(4)
    pivot["margin"]     = (pivot["dem_share"] - (1 - pivot["dem_share"])).round(4)

    return pivot[["county", "precinct", "year", "votes_cast",
                  "dem_votes", "rep_votes", "dem_share", "margin"]]


# ── OpenElections Fallback ─────────────────────────────────────────────────────
def fetch_openelections_year(year: int, target_counties: list) -> pd.DataFrame:
    """
    Download and parse OpenElections Florida precinct CSV for a given year.

    OpenElections schema: county, precinct, office, district, party, candidate, votes
    """
    url = f"{OPENELECTIONS_BASE}/{OPENELECTIONS_FILES[year]}"
    content = download(url, f"OpenElections FL {year}")
    if content is None:
        return pd.DataFrame()

    df = pd.read_csv(io.BytesIO(content), dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    df["county"] = df["county"].str.strip().str.title()
    df = df[df["county"].isin(target_counties)].copy()

    # Filter to US House
    df = df[df["office"].str.contains(
        "U.S. House|US House|Representative|Congress",
        case=False, na=False
    )].copy()

    df["votes"] = pd.to_numeric(df.get("votes", 0), errors="coerce").fillna(0).astype(int)
    df["party"] = df["party"].str.strip().str.upper()
    df["precinct"] = df["precinct"].str.strip()
    df["year"] = year

    return df[["county", "precinct", "year", "party", "votes"]]


def aggregate_openelections(df: pd.DataFrame) -> pd.DataFrame:
    """Same aggregation logic as FL DOS but from OpenElections schema."""
    agg = df.groupby(["county", "precinct", "year", "party"])["votes"].sum().reset_index()
    pivot = agg.pivot_table(
        index=["county", "precinct", "year"],
        columns="party", values="votes",
        aggfunc="sum", fill_value=0,
    ).reset_index()
    pivot.columns.name = None

    pivot["dem_votes"] = pivot.get("DEM", pd.Series(0, index=pivot.index))
    pivot["rep_votes"] = pivot.get("REP", pd.Series(0, index=pivot.index))
    pivot["votes_cast"] = pivot["dem_votes"] + pivot["rep_votes"]
    pivot["dem_share"]  = (pivot["dem_votes"] / pivot["votes_cast"].clip(lower=1)).round(4)
    pivot["margin"]     = (pivot["dem_share"] - (1 - pivot["dem_share"])).round(4)

    return pivot[["county", "precinct", "year", "votes_cast",
                  "dem_votes", "rep_votes", "dem_share", "margin"]]


# ── Voter Registration ────────────────────────────────────────────────────────
def fetch_voter_registration(target_counties: list) -> pd.DataFrame:
    """
    Download FL DOS statewide voter registration by party by precinct.
    Returns one row per precinct with registration party breakdowns.
    """
    content = download(VOTER_REG_URL, "Voter Registration")
    if content is None:
        print("  ⚠ Voter registration unavailable — will estimate from results data")
        return pd.DataFrame()

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith(".csv") or f.endswith(".txt")]
            if not csv_files:
                return pd.DataFrame()
            with z.open(csv_files[0]) as f:
                reg = pd.read_csv(f, dtype=str, encoding="latin-1")
    except Exception as e:
        print(f"  ⚠ Could not parse registration file: {e}")
        return pd.DataFrame()

    reg.columns = [c.strip().lower().replace(" ", "_") for c in reg.columns]

    # Normalize county/precinct columns
    for alias in ["countyname", "county_name"]:
        if alias in reg.columns:
            reg = reg.rename(columns={alias: "county"})
    for alias in ["precinctname", "precinct_name", "precinct_id"]:
        if alias in reg.columns:
            reg = reg.rename(columns={alias: "precinct"})

    reg["county"] = reg["county"].str.strip().str.title()
    reg = reg[reg["county"].isin(target_counties)].copy()
    reg["precinct"] = reg["precinct"].astype(str).str.strip()

    # Party columns: DEM, REP, NPA (No Party Affiliation), OTH
    for col in ["dem", "rep", "npa", "oth", "total"]:
        if col in reg.columns:
            reg[col] = pd.to_numeric(reg[col].str.replace(",", ""), errors="coerce").fillna(0)

    if "total" not in reg.columns:
        reg["total"] = reg.get("dem", 0) + reg.get("rep", 0) + reg.get("npa", 0)

    reg = reg[reg["total"] > 0].copy()
    reg["pct_dem_registered"] = (reg["dem"] / reg["total"].clip(lower=1)).round(4)
    reg["pct_rep_registered"] = (reg["rep"] / reg["total"].clip(lower=1)).round(4)
    reg["pct_npa_registered"] = (reg.get("npa", 0) / reg["total"].clip(lower=1)).round(4)

    return reg[["county", "precinct", "total", "dem", "rep",
                "pct_dem_registered", "pct_rep_registered", "pct_npa_registered"]].rename(
        columns={"total": "registered_voters"}
    )


# ── Census Demographics ───────────────────────────────────────────────────────
def fetch_census_demographics(target_counties: list) -> pd.DataFrame:
    """
    Pull ACS 5-year demographic estimates at block group level.
    Variables:
      B01001_001E — Total population
      B19013_001E — Median household income
      B03001_003E — Hispanic or Latino
      B02001_003E — Black or African American alone
      B01001_020E + B01001_021E + ... — Population 65+ (male)
      B01001_044E + B01001_045E + ... — Population 65+ (female)

    Note: Census data is at block group level, not precinct level.
    This function returns county-level aggregates as a fallback.
    For true precinct-level demographics, you'd need a spatial join with shapefiles.
    """
    STATE_FIPS = "12"  # Florida
    VARS = "B01001_001E,B19013_001E,B03001_003E,B02001_003E,B01001_020E,B01001_021E,B01001_022E,B01001_023E,B01001_024E,B01001_025E,B01001_044E,B01001_045E,B01001_046E,B01001_047E,B01001_048E,B01001_049E"

    rows = []
    for county, fips in COUNTY_FIPS.items():
        if county not in target_counties:
            continue
        url = (
            f"{CENSUS_API}?get={VARS}"
            f"&for=block%20group:*&in=state:{STATE_FIPS}%20county:{fips}"
        )
        try:
            r = requests.get(url, timeout=30)
            data = r.json()
            headers = data[0]
            for row in data[1:]:
                d = dict(zip(headers, row))
                pop = int(d.get("B01001_001E") or 0)
                if pop == 0:
                    continue

                hispanic = int(d.get("B03001_003E") or 0)
                black    = int(d.get("B02001_003E") or 0)
                income   = int(d.get("B19013_001E") or 0)

                # Sum 65+ age groups
                senior_vars = ["B01001_020E","B01001_021E","B01001_022E","B01001_023E",
                               "B01001_024E","B01001_025E","B01001_044E","B01001_045E",
                               "B01001_046E","B01001_047E","B01001_048E","B01001_049E"]
                senior = sum(int(d.get(v) or 0) for v in senior_vars)

                rows.append({
                    "county":       county,
                    "total_pop":    pop,
                    "hispanic_pop": hispanic,
                    "black_pop":    black,
                    "senior_pop":   senior,
                    "median_income": income if income > 0 else np.nan,
                })
        except Exception as e:
            print(f"  ⚠ Census API failed for {county}: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    county_agg = df.groupby("county").agg(
        total_pop=("total_pop", "sum"),
        hispanic_pop=("hispanic_pop", "sum"),
        black_pop=("black_pop", "sum"),
        senior_pop=("senior_pop", "sum"),
        median_income=("median_income", "median"),
    ).reset_index()

    county_agg["pct_hispanic"] = (county_agg["hispanic_pop"] / county_agg["total_pop"].clip(lower=1)).round(4)
    county_agg["pct_black"]    = (county_agg["black_pop"]    / county_agg["total_pop"].clip(lower=1)).round(4)
    county_agg["pct_senior"]   = (county_agg["senior_pop"]   / county_agg["total_pop"].clip(lower=1)).round(4)
    county_agg["median_income"] = county_agg["median_income"].fillna(55000).astype(int)

    print(f"  ✓ Census demographics fetched for {len(county_agg)} counties")
    return county_agg[["county", "pct_hispanic", "pct_black", "pct_senior", "median_income"]]


# ── Precinct Type Classifier ──────────────────────────────────────────────────
def classify_precinct_type(row: pd.Series) -> str:
    """
    Heuristically assign a precinct type based on registration and size.
    In production, replace with a spatial join to census urban/rural codes.
    """
    reg = row.get("registered_voters", 1000)
    dem_pct = row.get("pct_dem_registered", 0.5)
    npa_pct = row.get("pct_npa_registered", 0.15)
    hispanic = row.get("pct_hispanic", 0.2)
    senior = row.get("pct_senior", 0.2)

    if senior > 0.40:               return "Retirement"
    if hispanic > 0.55:             return "Hispanic Majority"
    if npa_pct > 0.25:              return "College Town"
    if dem_pct > 0.65 and reg > 2500: return "Urban Core"
    if dem_pct > 0.55:              return "Urban Fringe"
    if reg > 3000:                  return "Inner Suburb"
    if dem_pct < 0.35:              return "Rural"
    if dem_pct < 0.42:              return "Exurban"
    if dem_pct < 0.50:              return "Outer Suburb"
    return "Mixed Suburban"


# ── Compute Derived Features ──────────────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    years = ELECTIONS
    df["avg_turnout"]   = df[[f"{y}_turnout"    for y in years if f"{y}_turnout" in df.columns]].mean(axis=1)
    df["avg_dem_share"] = df[[f"{y}_dem_share"  for y in years if f"{y}_dem_share" in df.columns]].mean(axis=1)
    df["avg_margin"]    = df[[f"{y}_margin"     for y in years if f"{y}_margin" in df.columns]].mean(axis=1)

    first_y = min(y for y in years if f"{y}_turnout" in df.columns)
    last_y  = max(y for y in years if f"{y}_turnout" in df.columns)
    df["turnout_trend"] = df[f"{last_y}_turnout"] - df[f"{first_y}_turnout"]
    df["margin_trend"]  = df[f"{last_y}_margin"]  - df[f"{first_y}_margin"]
    df["competitiveness"] = df["avg_margin"].abs().apply(lambda x: round(1 - min(x, 1), 4))

    return df


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(source: str = "fldos", counties: list = None) -> pd.DataFrame:
    target = counties or TARGET_COUNTIES
    print(f"\n{'═'*60}")
    print(f"  BallotBase Real Data Ingestion")
    print(f"  Source:   {source.upper()}")
    print(f"  Counties: {', '.join(target)}")
    print(f"{'═'*60}\n")

    # ── Step 1: Election results ──────────────────────────────────────────────
    all_results = []
    for year in ELECTIONS:
        print(f"[{year}] Fetching precinct results...")
        if source == "openelections":
            raw = fetch_openelections_year(year, target)
            if raw.empty:
                print(f"  ⚠ Skipping {year} — no data returned")
                continue
            agg = aggregate_openelections(raw)
        else:
            url = FL_DOS_URLS.get(year)
            content = download(url, f"FL DOS {year} General Election")
            if not content:
                print(f"  ⚠ Skipping {year} — download failed")
                continue
            try:
                raw = parse_fldos_year(content, year, target)
                agg = aggregate_fldos(raw)
            except Exception as e:
                print(f"  ⚠ Parse error for {year}: {e} — trying OpenElections fallback")
                raw = fetch_openelections_year(year, target)
                if raw.empty:
                    continue
                agg = aggregate_openelections(raw)

        agg["year"] = year
        print(f"  ✓ {len(agg)} precincts parsed")
        all_results.append(agg)

    if not all_results:
        raise RuntimeError("No election data could be fetched. Check your internet connection.")

    results_df = pd.concat(all_results, ignore_index=True)

    # ── Step 2: Pivot to wide format (one row per precinct) ───────────────────
    print("\nPivoting to wide format...")
    years_present = sorted(results_df["year"].unique())
    base = results_df[["county", "precinct"]].drop_duplicates()

    for year in years_present:
        yr_df = results_df[results_df["year"] == year][
            ["county", "precinct", "votes_cast", "dem_votes", "rep_votes", "dem_share", "margin"]
        ].copy()
        yr_df = yr_df.rename(columns={
            "votes_cast": f"{year}_votes_cast",
            "dem_votes":  f"{year}_dem_votes",
            "rep_votes":  f"{year}_rep_votes",
            "dem_share":  f"{year}_dem_share",
            "margin":     f"{year}_margin",
        })
        base = base.merge(yr_df, on=["county", "precinct"], how="left")

    print(f"  ✓ {len(base)} unique precincts across {len(target)} counties")

    # ── Step 3: Voter registration ────────────────────────────────────────────
    print("\nFetching voter registration...")
    reg_df = fetch_voter_registration(target)

    if not reg_df.empty:
        base = base.merge(reg_df, on=["county", "precinct"], how="left")
        # Fill missing registration with estimate from votes cast
        for col in ["registered_voters", "pct_dem_registered", "pct_rep_registered", "pct_npa_registered"]:
            if col not in base.columns:
                base[col] = np.nan
    else:
        # Estimate registered voters from max votes cast across all years
        vote_cols = [f"{y}_votes_cast" for y in years_present if f"{y}_votes_cast" in base.columns]
        base["registered_voters"] = base[vote_cols].max(axis=1).fillna(1000).astype(int)
        # Estimate registration from 2024 dem share
        if "2024_dem_share" in base.columns:
            base["pct_dem_registered"] = (base["2024_dem_share"] * 0.85 + 0.075).clip(0.1, 0.85).round(4)
            base["pct_rep_registered"] = (1 - base["2024_dem_share"]) * 0.85 + 0.075
            base["pct_rep_registered"] = base["pct_rep_registered"].clip(0.1, 0.85).round(4)
            base["pct_npa_registered"] = (1 - base["pct_dem_registered"] - base["pct_rep_registered"]).clip(0.05, 0.30).round(4)
        print("  ⚠ Registration estimated from vote totals")

    base["registered_voters"] = base["registered_voters"].fillna(
        base[[f"{y}_votes_cast" for y in years_present if f"{y}_votes_cast" in base.columns]].max(axis=1)
    ).fillna(1000).astype(int)

    # ── Step 4: Turnout ───────────────────────────────────────────────────────
    for year in years_present:
        vc_col = f"{year}_votes_cast"
        to_col = f"{year}_turnout"
        if vc_col in base.columns:
            base[to_col] = (base[vc_col] / base["registered_voters"].clip(lower=1)).clip(0.1, 1.0).round(4)

    # ── Step 5: Census demographics ───────────────────────────────────────────
    print("\nFetching Census demographics...")
    census_df = fetch_census_demographics(target)

    if not census_df.empty:
        base = base.merge(census_df, on="county", how="left")
    else:
        # Fallback: county-level defaults
        defaults = {
            "Orange":   {"pct_hispanic": 0.29, "pct_black": 0.22, "pct_senior": 0.14, "median_income": 58000},
            "Seminole": {"pct_hispanic": 0.19, "pct_black": 0.12, "pct_senior": 0.18, "median_income": 68000},
            "Osceola":  {"pct_hispanic": 0.52, "pct_black": 0.11, "pct_senior": 0.13, "median_income": 51000},
            "Brevard":  {"pct_hispanic": 0.11, "pct_black": 0.10, "pct_senior": 0.22, "median_income": 57000},
            "Volusia":  {"pct_hispanic": 0.13, "pct_black": 0.11, "pct_senior": 0.24, "median_income": 51000},
        }
        for col in ["pct_hispanic","pct_black","pct_senior","median_income"]:
            base[col] = base["county"].map({c: v[col] for c, v in defaults.items()})
        print("  ⚠ Using hardcoded county-level demographic defaults")

    # ── Step 6: Precinct IDs, types, coordinates ──────────────────────────────
    print("\nFinalizing precinct metadata...")
    base["precinct_id"]   = ["P" + str(i+1).zfill(4) for i in range(len(base))]
    base["precinct_name"] = base["county"] + "-" + base["precinct"].astype(str)
    base["precinct_type"] = base.apply(classify_precinct_type, axis=1)

    # Approximate coordinates (county centroids + random offset)
    # In production: replace with shapefile centroid computation
    county_centers = {
        "Orange":   (28.538, -81.379),
        "Seminole": (28.717, -81.209),
        "Osceola":  (28.167, -81.120),
        "Brevard":  (28.210, -80.742),
        "Volusia":  (29.028, -81.048),
    }
    np.random.seed(99)
    base["latitude"]  = base["county"].map(lambda c: county_centers.get(c, (28.5, -81.3))[0]) + np.random.uniform(-0.30, 0.30, len(base))
    base["longitude"] = base["county"].map(lambda c: county_centers.get(c, (28.5, -81.3))[1]) + np.random.uniform(-0.30, 0.30, len(base))
    base["latitude"]  = base["latitude"].round(5)
    base["longitude"] = base["longitude"].round(5)

    # ── Step 7: Fill gaps and compute derived features ────────────────────────
    for col in ["pct_dem_registered","pct_rep_registered","pct_npa_registered",
                "pct_hispanic","pct_black","pct_senior"]:
        if col in base.columns:
            base[col] = base[col].fillna(base[col].median()).round(4)

    base = compute_features(base)

    # Reorder columns to match app.py expectations
    id_cols = ["precinct_id","county","precinct_name","precinct_type","latitude","longitude",
               "registered_voters","pct_dem_registered","pct_rep_registered","pct_npa_registered",
               "pct_hispanic","pct_black","pct_senior","median_income"]
    elec_cols = []
    for y in sorted(years_present):
        for suffix in ["turnout","votes_cast","dem_votes","rep_votes","dem_share","margin"]:
            col = f"{y}_{suffix}"
            if col in base.columns:
                elec_cols.append(col)
    feature_cols = ["avg_turnout","avg_dem_share","avg_margin","turnout_trend","margin_trend","competitiveness"]

    final_cols = [c for c in id_cols + elec_cols + feature_cols if c in base.columns]
    base = base[final_cols].copy()

    # Drop precincts with no vote data at all
    vote_cols = [c for c in base.columns if "_votes_cast" in c]
    base = base[base[vote_cols].sum(axis=1) > 0].reset_index(drop=True)

    print(f"\n{'═'*60}")
    print(f"  ✓ Final dataset: {len(base)} precincts")
    print(f"  ✓ Elections:     {', '.join(str(y) for y in years_present)}")
    print(f"  ✓ Counties:      {', '.join(base['county'].unique())}")
    print(f"{'═'*60}\n")

    return base


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BallotBase Real Data Ingestion")
    parser.add_argument("--source", choices=["fldos", "openelections"], default="fldos",
                        help="Data source (default: fldos)")
    parser.add_argument("--counties", nargs="+", default=None,
                        help="Counties to include (default: all 5 Central FL counties)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: data/precincts.csv next to this script)")
    args = parser.parse_args()

    counties = args.counties or TARGET_COUNTIES

    df = run_pipeline(source=args.source, counties=counties)

    out_path = args.out or os.path.join(os.path.dirname(__file__), "precincts.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
    print(f"\nSample (first 3 rows):")
    print(df[["county","precinct_name","precinct_type","registered_voters",
              "2024_turnout","2024_margin"]].head(3).to_string(index=False))
