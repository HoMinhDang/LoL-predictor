import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import EARLY_FEATURES, TARGET, TEST_SIZE, RANDOM_STATE


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EARLY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[EARLY_FEATURES].copy()


def get_nan_report(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    report = []
    for col in EARLY_FEATURES:
        if col in df.columns:
            n = df[col].isna().sum()
            report.append({
                'feature': col,
                'nan_count': n,
                'nan_pct': round(n / total * 100, 2) if total > 0 else 0,
            })
    return pd.DataFrame(report)


def get_nan_league_report(df: pd.DataFrame) -> pd.DataFrame:
    report = []
    for league in df['league'].unique():
        lg_df = df[df['league'] == league]
        has_nan = lg_df[EARLY_FEATURES].isna().any(axis=1).sum()
        report.append({
            'league': league,
            'total_rows': len(lg_df),
            'rows_with_nan': int(has_nan),
            'nan_pct': round(has_nan / len(lg_df) * 100, 2) if len(lg_df) > 0 else 0,
        })
    return pd.DataFrame(report).sort_values('rows_with_nan', ascending=False).reset_index(drop=True)


def handle_nulls(X: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    if strategy == 'drop':
        return X.dropna()
    elif strategy == 'median':
        return X.fillna(X.median(numeric_only=True))
    else:
        raise ValueError(f"Unknown NaN strategy: {strategy}. Use 'drop' or 'median'.")


def build_dataset(df: pd.DataFrame, nan_strategy: str = 'drop') -> (pd.DataFrame, pd.Series):
    X = prepare_features(df)
    X = handle_nulls(X, strategy=nan_strategy)
    y = df.loc[X.index, TARGET].values
    return X, y


def split_data(X, y, test_size: float = None, random_state: int = None):
    test_size = test_size or TEST_SIZE
    random_state = random_state or RANDOM_STATE
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_scaler():
    return StandardScaler()
