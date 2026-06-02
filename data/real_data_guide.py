"""
data/real_data_guide.py
────────────────────────────────────────────────────────────────────────────────
HOW TO REPLACE SYNTHETIC DATA WITH REAL FLORIDA ELECTION DATA
────────────────────────────────────────────────────────────────────────────────

STEP 1 — Florida Division of Elections (official precinct results)
──────────────────────────────────────────────────────────────────
URL: https://dos.myflorida.com/elections/data-statistics/elections-data/

Files to download:
  • "General Election Results by Precinct" for 2016, 2018, 2020, 2022, 2024
  • Format: SOV files (.txt) or downloadable Excel
  • Filter for: Orange, Seminole, Osceola, Brevard, Volusia counties
  • Columns you need: CountyCode, Precinct, RaceDescription, VoteFor,
                      CanName, PartyCode, Votes

Example loading code:
"""

import pandas as pd
import os

def load_fdle_sov(filepath: str, year: int) -> pd.DataFrame:
    """
    Load a Florida Division of Elections SOV file.
    Adjust column names to match what DOS actually exports.
    """
    df = pd.read_csv(filepath, sep="\t", encoding="latin-1")

    # Keep only US House races (congressional)
    cong = df[df["RaceDescription"].str.contains("United States Representative", na=False)].copy()

    # Aggregate to precinct level
    result = cong.groupby(["CountyCode", "Precinct", "PartyCode"])["Votes"].sum().unstack(fill_value=0)
    result = result.reset_index()

    result["year"] = year
    result["votes_cast"] = result.get("DEM", 0) + result.get("REP", 0) + result.get("NPA", 0)
    result["dem_votes"] = result.get("DEM", 0)
    result["rep_votes"] = result.get("REP", 0)
    result["dem_share"] = result["dem_votes"] / result["votes_cast"].clip(lower=1)
    result["margin"] = result["dem_share"] - (1 - result["dem_share"])

    return result[["CountyCode", "Precinct", "year", "votes_cast",
                   "dem_votes", "rep_votes", "dem_share", "margin"]]


"""
STEP 2 — Registered Voter Counts
─────────────────────────────────
URL: https://dos.myflorida.com/elections/data-statistics/voter-registration-statistics/

Download: "Voter Registration by Precinct" (updated monthly)
Columns: CountyCode, Precinct, Party, RegisteredVoters

This gives you party registration breakdowns per precinct.
"""

def load_voter_reg(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    pivot = df.pivot_table(index=["CountyCode", "Precinct"],
                           columns="Party", values="RegisteredVoters",
                           aggfunc="sum", fill_value=0).reset_index()
    pivot["total_registered"] = pivot.sum(axis=1, numeric_only=True)
    pivot["pct_dem"] = pivot.get("DEM", 0) / pivot["total_registered"].clip(lower=1)
    pivot["pct_rep"] = pivot.get("REP", 0) / pivot["total_registered"].clip(lower=1)
    pivot["pct_npa"] = pivot.get("NPA", 0) / pivot["total_registered"].clip(lower=1)
    return pivot


"""
STEP 3 — Precinct Shapefiles (for choropleth maps instead of scatter)
───────────────────────────────────────────────────────────────────────
URL: https://redistricting.myflorida.com / county GIS portals

Orange County GIS: https://www.orangecountyfl.net/911EmergencyManagement/GISOpenData.aspx
Seminole County:   https://www.seminolecountyfl.gov/departments-services/information-technologies-gis/
Osceola County:    https://www.osceola.org/departments-o-r/planning/gis-mapping/
Brevard County:    https://gis.brevardfl.gov/
Volusia County:    https://maps.vcgov.org/

Download: Precinct boundary shapefiles (.shp)
Then use geopandas to join with your results data.
"""

def load_shapefile_and_merge(shp_path: str, results_df: pd.DataFrame):
    import geopandas as gpd
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84 lat/lon

    # Join on precinct ID (column names vary by county)
    merged = gdf.merge(results_df, left_on="PRECINCT", right_on="Precinct", how="left")
    return merged


"""
STEP 4 — Census Demographics (for feature engineering)
────────────────────────────────────────────────────────
Use the Census API or download directly.
API: https://api.census.gov/data/2020/acs/acs5

Variables useful for turnout/margin prediction:
  B01001_001E  — Total population
  B01001_020E:B01001_025E — Men 65+
  B01001_044E:B01001_049E — Women 65+
  B03001_003E  — Hispanic or Latino
  B02001_003E  — Black or African American alone
  B19013_001E  — Median household income
  B15003_022E  — Bachelor's degree holders

Note: Census data is at block group / tract level, not precinct level.
You'll need to do a spatial join (geopandas) to assign census data to precincts.
"""

def fetch_census_data(state_fips="12", county_fips="095"):
    """Example for Orange County (FIPS 095). Requires census API key."""
    import requests
    API_KEY = "YOUR_KEY_FROM_api.census.gov/data/key_signup.html"
    url = (
        f"https://api.census.gov/data/2020/acs/acs5"
        f"?get=B01001_001E,B19013_001E,B03001_003E,B02001_003E"
        f"&for=block%20group:*&in=state:{state_fips}%20county:{county_fips}"
        f"&key={API_KEY}"
    )
    r = requests.get(url)
    data = r.json()
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    df = df.rename(columns={
        "B01001_001E": "total_pop",
        "B19013_001E": "median_income",
        "B03001_003E": "hispanic_pop",
        "B02001_003E": "black_pop",
    })
    return df


"""
STEP 5 — Early Vote / VBM Data (for real-time cycle tracking)
──────────────────────────────────────────────────────────────
URL: https://dos.myflorida.com/elections/data-statistics/absentee-ballot-request/

During an active election, FL DOS publishes daily early vote files.
This lets you track returned ballots by party and update your turnout model
in real time.
"""

"""
STEP 6 — Putting it all together
──────────────────────────────────
Once you have the real data, replace the synthetic CSV with your merged dataset.
The app.py will work with any DataFrame that has these columns:

REQUIRED:
  precinct_id, county, precinct_name, precinct_type (optional, can assign via lookup)
  latitude, longitude  (can compute centroid from shapefile)
  registered_voters
  pct_dem_registered, pct_rep_registered, pct_npa_registered
  {year}_turnout, {year}_votes_cast, {year}_dem_votes, {year}_rep_votes,
  {year}_dem_share, {year}_margin   — for years 2016, 2018, 2020, 2022, 2024

OPTIONAL (improves model):
  pct_hispanic, pct_black, pct_senior, median_income

USEFUL RESOURCES:
  • Florida election law & data:  dos.myflorida.com/elections
  • OpenElections (cleaned data):  github.com/openelections/openelections-data-fl
  • DAVE's Redistricting:          davesredistricting.org
  • Ballotpedia precinct results:  ballotpedia.org
  • MIT Election Lab:              electionlab.mit.edu/data
  • MGGG (redistricting tools):    mggg.org
"""

if __name__ == "__main__":
    print("See comments in this file for data integration steps.")
    print("Key sources:")
    print("  1. FL Division of Elections: dos.myflorida.com/elections")
    print("  2. County GIS portals (shapefiles)")
    print("  3. US Census API (census.gov)")
    print("  4. OpenElections FL: github.com/openelections/openelections-data-fl")
