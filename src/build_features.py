"""Build train/test feature tables and game sequences from raw NBA game logs.

Input:  data/games_all_seasons.csv (from collect_data.py)
Output: data/tabular_train.csv, data/tabular_test.csv
        data/sequences.npz  (X_train, y_train, X_test, y_test for LSTM/Transformer)
        data/feature_names.json
        data/team_markets.csv (used by bias analysis)

Test season = 2023-24. Train seasons = 2018-19 .. 2022-23.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

TRAIN_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]
TEST_SEASON = "2023-24"

# Per-team box-score numeric columns we'll use to build rolling averages.
BOX_COLS = [
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "STL", "BLK", "TOV",
    "PLUS_MINUS",
]

ROLLING_WINDOWS = [5, 10, 20]

# Approximate metro-area market sizes (TV households). Used for bias analysis.
# Source: well-known Nielsen DMA rankings (rough buckets).
TEAM_MARKETS = {
    # large markets (top 10 metros)
    "NYK": "large", "BKN": "large", "LAL": "large", "LAC": "large",
    "CHI": "large", "PHI": "large", "DAL": "large", "HOU": "large",
    "WAS": "large", "ATL": "large", "BOS": "large",
    # medium
    "GSW": "medium", "MIA": "medium", "DET": "medium", "PHX": "medium",
    "MIN": "medium", "DEN": "medium", "ORL": "medium", "CLE": "medium",
    "POR": "medium", "SAC": "medium",
    # small
    "MEM": "small", "OKC": "small", "MIL": "small", "IND": "small",
    "NOP": "small", "SAS": "small", "CHA": "small", "UTA": "small",
    "TOR": "medium",  # large Canadian metro
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA / "games_all_seasons.csv")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    # WL = "W" / "L"; drop rows with missing outcome (rare cancellations)
    df = df.dropna(subset=["WL"]).copy()
    df["WIN"] = (df["WL"] == "W").astype(int)
    df["HOME"] = (~df["MATCHUP"].str.contains("@")).astype(int)
    return df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"]).reset_index(drop=True)


def add_team_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team rolling stats over previous N games (shifted: no leakage)."""
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    g = df.groupby("TEAM_ID", group_keys=False)

    # rest days
    df["REST_DAYS"] = g["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=20)
    df["B2B"] = (df["REST_DAYS"] <= 1).astype(int)

    # rolling averages of box stats over previous N games (shift(1) to exclude current)
    for col in BOX_COLS:
        for w in ROLLING_WINDOWS:
            df[f"{col}_R{w}"] = g[col].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=1).mean()
            )
    # season-to-date win pct prior to this game
    df["WIN_STD"] = g["WIN"].transform(lambda s: s.shift(1).expanding().mean())
    # home & away win pct STD
    df["HOME_FLAG_PREV"] = g["HOME"].shift(1)
    df["WIN_HOME_STD"] = g.apply(
        lambda d: d["WIN"].where(d["HOME"] == 1).shift(1).expanding().mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    df["WIN_AWAY_STD"] = g.apply(
        lambda d: d["WIN"].where(d["HOME"] == 0).shift(1).expanding().mean(),
        include_groups=False,
    ).reset_index(level=0, drop=True)
    return df


def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Add head-to-head win-rate for each team against the specific opponent prior to this game."""
    # extract opponent abbreviation from MATCHUP
    opp = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*([A-Z0-9]+)")[0]
    df["OPP_ABBR"] = opp
    pair = df.groupby(["TEAM_ABBREVIATION", "OPP_ABBR"], group_keys=False)
    df["H2H_WINPCT"] = pair["WIN"].transform(lambda s: s.shift(1).expanding().mean())
    return df


def join_home_away(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot two team-rows per game into one game-row with home_/away_ prefixed feats."""
    feature_cols = ["REST_DAYS", "B2B", "WIN_STD", "WIN_HOME_STD", "WIN_AWAY_STD",
                    "H2H_WINPCT"] + [
        f"{c}_R{w}" for c in BOX_COLS for w in ROLLING_WINDOWS
    ]

    home_rows = df[df["HOME"] == 1].copy()
    away_rows = df[df["HOME"] == 0].copy()

    keep_home = ["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ABBREVIATION", "WIN"] + feature_cols
    keep_away = ["GAME_ID", "TEAM_ABBREVIATION"] + feature_cols

    home = home_rows[keep_home].rename(
        columns={"TEAM_ABBREVIATION": "HOME_TEAM", "WIN": "HOME_WIN",
                 **{c: f"HOME_{c}" for c in feature_cols}}
    )
    away = away_rows[keep_away].rename(
        columns={"TEAM_ABBREVIATION": "AWAY_TEAM",
                 **{c: f"AWAY_{c}" for c in feature_cols}}
    )

    games = home.merge(away, on="GAME_ID", how="inner")

    # diff features (home minus away) — often informative
    for c in feature_cols:
        games[f"DIFF_{c}"] = games[f"HOME_{c}"] - games[f"AWAY_{c}"]

    return games


def build_sequences(df: pd.DataFrame, seq_len: int = 10):
    """For each game build a sequence of last `seq_len` games of each team.

    Returns X: (n_games, 2*seq_len, n_box_cols), y: (n_games,).
    The first seq_len rows of each team-season are padded with zeros.
    Label = 1 if home team won.
    """
    df = df.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    box_cols_seq = BOX_COLS + ["HOME"]
    df_box = df[["TEAM_ID", "GAME_DATE", "GAME_ID"] + box_cols_seq].copy()
    # per-team history list of arrays
    team_hist: dict[int, list[np.ndarray]] = {}
    for tid, sub in df_box.groupby("TEAM_ID", sort=False):
        team_hist[tid] = sub[box_cols_seq].fillna(0).to_numpy(dtype=np.float32)

    # mapping (team_id, game_id) -> position in that team's history
    pos_idx = {}
    for tid, sub in df_box.groupby("TEAM_ID", sort=False):
        for i, gid in enumerate(sub["GAME_ID"].to_numpy()):
            pos_idx[(tid, gid)] = i

    # identify games with both home and away rows
    game_rows = df[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "HOME", "WIN"]].copy()
    g_home = game_rows[game_rows["HOME"] == 1].rename(
        columns={"TEAM_ID": "HOME_TID", "WIN": "HOME_WIN"}
    )[["GAME_ID", "GAME_DATE", "SEASON", "HOME_TID", "HOME_WIN"]]
    g_away = game_rows[game_rows["HOME"] == 0].rename(columns={"TEAM_ID": "AWAY_TID"})[
        ["GAME_ID", "AWAY_TID"]
    ]
    pairs = g_home.merge(g_away, on="GAME_ID", how="inner")

    n_feat = len(box_cols_seq)
    X = np.zeros((len(pairs), 2 * seq_len, n_feat), dtype=np.float32)
    y = pairs["HOME_WIN"].to_numpy(dtype=np.float32)
    meta_season = pairs["SEASON"].to_numpy()

    for i, row in enumerate(pairs.itertuples(index=False)):
        for offset, tid in enumerate([row.HOME_TID, row.AWAY_TID]):
            pos = pos_idx.get((tid, row.GAME_ID))
            if pos is None:
                continue
            start = max(0, pos - seq_len)
            hist = team_hist[tid][start:pos]
            if hist.shape[0] == 0:
                continue
            slot_start = offset * seq_len + (seq_len - hist.shape[0])
            slot_end = (offset + 1) * seq_len
            X[i, slot_start:slot_end] = hist

    return X, y, meta_season


def main() -> None:
    print("Loading raw...")
    raw = load_raw()
    print(f"  {len(raw)} team-game rows across seasons: {sorted(raw['SEASON'].unique())}")

    print("Computing team rolling stats + rest...")
    raw = add_team_rolling(raw)
    print("Computing head-to-head win pct...")
    raw = add_h2h(raw)

    print("Pivoting to one-row-per-game tabular features...")
    games = join_home_away(raw)
    games = games.dropna(subset=["HOME_WIN"]).copy()
    # drop earliest games of each season that lack rolling history (NaNs in rolling cols)
    games = games.fillna(0)

    train = games[games["SEASON"].isin(TRAIN_SEASONS)].copy()
    test = games[games["SEASON"] == TEST_SEASON].copy()
    print(f"  train games: {len(train)}, test games: {len(test)}")

    feature_cols = [c for c in games.columns if c.startswith(("HOME_", "AWAY_", "DIFF_"))
                    and c not in ("HOME_TEAM", "AWAY_TEAM", "HOME_WIN")]

    train.to_csv(DATA / "tabular_train.csv", index=False)
    test.to_csv(DATA / "tabular_test.csv", index=False)
    with open(DATA / "feature_names.json", "w") as f:
        json.dump(feature_cols, f)

    print("Building sequence tensors for LSTM/Transformer...")
    X, y, seasons = build_sequences(raw, seq_len=10)
    train_mask = np.isin(seasons, TRAIN_SEASONS)
    test_mask = seasons == TEST_SEASON
    np.savez(
        DATA / "sequences.npz",
        X_train=X[train_mask], y_train=y[train_mask],
        X_test=X[test_mask], y_test=y[test_mask],
    )
    print(f"  sequences train: {train_mask.sum()}, test: {test_mask.sum()}, shape per game: {X.shape[1:]}")

    # market size table
    mk = pd.DataFrame(
        [(k, v) for k, v in TEAM_MARKETS.items()],
        columns=["TEAM_ABBR", "MARKET"],
    )
    mk.to_csv(DATA / "team_markets.csv", index=False)

    print("\nDone. Features:", len(feature_cols))


if __name__ == "__main__":
    main()
