"""Collect NBA regular-season game logs 2018-19 through 2023-24 via nba_api.

Saves one CSV per season under data/. Uses LeagueGameFinder which returns one
row per (team, game) — i.e., two rows per game (home + away). The feature
engineering step pivots that into one row per game.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import LeagueID, SeasonTypeAllStar

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]


def fetch_season(season: str, retries: int = 4) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                league_id_nullable=LeagueID.nba,
                season_nullable=season,
                season_type_nullable=SeasonTypeAllStar.regular,
                timeout=60,
            )
            df = finder.get_data_frames()[0]
            df["SEASON"] = season
            return df
        except Exception as exc:
            last_err = exc
            wait = 2 ** attempt
            print(f"  attempt {attempt + 1} failed ({exc}); sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {season}: {last_err}")


def main() -> None:
    all_frames: list[pd.DataFrame] = []
    for season in SEASONS:
        out_path = DATA_DIR / f"games_{season.replace('-', '_')}.csv"
        if out_path.exists():
            print(f"[skip] {season} already cached at {out_path.name}")
            all_frames.append(pd.read_csv(out_path))
            continue
        print(f"[fetch] {season} ...")
        df = fetch_season(season)
        df.to_csv(out_path, index=False)
        print(f"  saved {len(df)} rows -> {out_path.name}")
        all_frames.append(df)
        time.sleep(1.5)  # be polite

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = DATA_DIR / "games_all_seasons.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {len(combined)} rows, {combined['SEASON'].nunique()} seasons -> {combined_path.name}")


if __name__ == "__main__":
    main()
