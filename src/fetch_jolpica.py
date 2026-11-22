"""Download race results from the Jolpica F1 API (Ergast-compatible).
Data source: https://api.jolpi.ca/ergast/f1  (see the API's terms of use)
"""
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
OUT_DIR = Path("AdvanceData/data/external")   # kernel starts from the workspace root

def get_json(url, params=None, retries=8):
    """GET with a simple retry when the server says 'too many requests' (429)."""
    for attempt in range(retries):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            time.sleep(15 * (attempt + 1))   
            continue
        response.raise_for_status()
    raise RuntimeError(f"Failed after {retries} tries: {url}")


def fetch_season_results(year, page_size=100):
    """One row per driver per race for a season (the API returns pages)."""
    rows, offset = [], 0
    while True:
        data = get_json(f"{BASE_URL}/{year}/results.json",
                        {"limit": page_size, "offset": offset})["MRData"]
        for race in data["RaceTable"]["Races"]:
            for res in race["Results"]:
                rows.append({
                    "year": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "date": race["date"],
                    "driver_id": res["Driver"]["driverId"],
                    "driver_name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                    "constructor_id": res["Constructor"]["constructorId"],
                    "constructor_name": res["Constructor"]["name"],
                    "grid": int(res["grid"]),
                    "position_order": int(res["position"]),
                    "points": float(res["points"]),
                    "status": res["status"],
                })
        offset += page_size
        if offset >= int(data["total"]):
            break
        time.sleep(0.5)          # be gentle with a free community service
    return pd.DataFrame(rows)
def lap_time_to_seconds(text):
    """Convert '1:27.452' or '58.123' into seconds."""
    if ":" in text:
        minutes, seconds = text.split(":")
        return int(minutes) * 60 + float(seconds)
    return float(text)


def fetch_race_laps(year, round_number, page_size=100):
    """One row per driver per lap for a single race (the API returns pages)."""
    rows, offset = [], 0
    while True:
        data = get_json(f"{BASE_URL}/{year}/{round_number}/laps.json",
                        {"limit": page_size, "offset": offset})["MRData"]
        races = data["RaceTable"]["Races"]
        if not races or not races[0]["Laps"]:        # empty page = no more data
            break
        for lap in races[0]["Laps"]:
            for timing in lap["Timings"]:
                rows.append({"year": year, "round": round_number, "lap": int(lap["number"]),
                             "driver_id": timing["driverId"],
                             "seconds": lap_time_to_seconds(timing["time"])})
        offset += page_size
        time.sleep(1.0)
    return rows


LAPS_DIR = OUT_DIR / "laps_by_race"


def fetch_season_laps(year):
    """Lap times for every race of a season; each race is saved as soon as it is done."""
    LAPS_DIR.mkdir(parents=True, exist_ok=True)
    rounds = pd.read_csv(OUT_DIR / f"results_{year}.csv")["round"].unique()
    frames = []
    for round_number in rounds:
        cache_file = LAPS_DIR / f"laps_{year}_{int(round_number):02d}.csv"
        if cache_file.exists():                        # already downloaded: skip it
            frames.append(pd.read_csv(cache_file))
            continue
        race_laps = pd.DataFrame(fetch_race_laps(year, int(round_number)))
        race_laps.to_csv(cache_file, index=False)      # save right away
        frames.append(race_laps)
        print(year, "round", round_number, "done")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in (2025, 2026):
        if (OUT_DIR / f"laps_{year}.csv").exists():
            print(year, "laps already downloaded, skipping")
            continue
        results = fetch_season_results(year)
        results.to_csv(OUT_DIR / f"results_{year}.csv", index=False)
        print(year, "results rows:", len(results), "races:", results["round"].nunique())
        laps = fetch_season_laps(year)
        laps.to_csv(OUT_DIR / f"laps_{year}.csv", index=False)
        print(year, "lap rows:", len(laps))


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in (2025, 2026):
        results = fetch_season_results(year)
        results.to_csv(OUT_DIR / f"results_{year}.csv", index=False)
        print(year, "results rows:", len(results), "races:", results["round"].nunique())
        laps = fetch_season_laps(year)
        laps.to_csv(OUT_DIR / f"laps_{year}.csv", index=False)
        print(year, "lap rows:", len(laps))