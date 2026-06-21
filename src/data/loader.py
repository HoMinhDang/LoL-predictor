import pandas as pd
from src.config import DATA_PATH


def load_csv(path: str = None) -> pd.DataFrame:
    path = path or DATA_PATH
    return pd.read_csv(path, low_memory=False)


def filter_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['position'] == 'team'].copy()


def filter_by_league(df: pd.DataFrame, league: str = None) -> pd.DataFrame:
    if league is None or league.lower() == 'all':
        return df
    return df[df['league'] == league].copy()
