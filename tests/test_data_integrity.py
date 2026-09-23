from pathlib import Path
import pandas as pd
import pytest
import json

DATA = Path(__file__).resolve().parent.parent / "data"

@pytest.fixture(scope="module")
def data():
    train = pd.read_csv(DATA / "tabular_train.csv", parse_dates=["GAME_DATE"])
    test = pd.read_csv(DATA / "tabular_test.csv", parse_dates=["GAME_DATE"])
    return train, test



def test_train_ends_before_test_begins(data):
    train, test = data
    assert train["GAME_DATE"].max() < test["GAME_DATE"].min()

def test_test_set_is_one_season(data):
    train, test = data

    test_seasons = set(test["SEASON"])

    assert len(test_seasons) == 1
    assert test_seasons.isdisjoint(set(train["SEASON"]))

def test_no_game_appears_in_both_splits(data):
    train, test = data

    test_id = set(test["GAME_ID"])

    assert test_id.isdisjoint(set(train["GAME_ID"]))

def test_features_exclude_target_and_count_108():
    feature_cols = json.loads((DATA / "feature_names.json").read_text())

    assert "HOME_WIN" not in feature_cols
    assert len(feature_cols) == 108