import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_config_features():
    from src.config import EARLY_FEATURES, TARGET
    assert len(EARLY_FEATURES) > 0
    assert TARGET == 'result'
    assert 'turretplates' not in EARLY_FEATURES
    assert 'team kpm' not in EARLY_FEATURES
    assert 'ckpm' not in EARLY_FEATURES


def test_loader_filters():
    from src.data.loader import load_csv, filter_team_rows, filter_by_league
    df = load_csv()
    team_df = filter_team_rows(df)
    assert len(team_df) > 0
    assert (team_df['position'] == 'team').all()
    lpl = filter_by_league(df, 'LPL')
    assert len(lpl) > 0
    assert (lpl['league'] == 'LPL').all()
    all_df = filter_by_league(df, None)
    assert len(all_df) == len(df)


def test_prepare_features():
    from src.data.loader import load_csv, filter_team_rows
    from src.data.preprocessor import prepare_features
    df = filter_team_rows(load_csv())
    X = prepare_features(df)
    assert list(X.columns) == [
        'golddiffat15', 'xpdiffat15', 'csdiffat15',
        'killsat15', 'assistsat15', 'deathsat15',
        'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15',
        'firstblood', 'firstdragon', 'firstherald',
        'firsttower', 'firstmidtower', 'firsttothreetowers',
        'void_grubs',
    ]


def test_handle_nulls_drop():
    from src.data.preprocessor import handle_nulls
    import pandas as pd
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
    result = handle_nulls(df, strategy='drop')
    assert len(result) == 1


def test_handle_nulls_median():
    from src.data.preprocessor import handle_nulls
    import pandas as pd
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': [4, 5, np.nan]})
    result = handle_nulls(df, strategy='median')
    assert len(result) == 3
    assert not result.isna().any().any()


def test_build_dataset():
    from src.data.loader import load_csv, filter_team_rows
    from src.data.preprocessor import build_dataset
    df = filter_team_rows(load_csv())
    X, y = build_dataset(df, nan_strategy='drop')
    assert len(X) > 0
    assert len(X) == len(y)
    assert X.isna().sum().sum() == 0


def test_split_data():
    from src.data.loader import load_csv, filter_team_rows
    from src.data.preprocessor import build_dataset, split_data
    df = filter_team_rows(load_csv())
    X, y = build_dataset(df, nan_strategy='drop')
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) > len(X_test)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_scaler():
    from src.data.preprocessor import get_scaler
    scaler = get_scaler()
    assert scaler is not None
    data = np.array([[1, 2], [3, 4], [5, 6]])
    scaled = scaler.fit_transform(data)
    assert scaled.mean() < 0.01


def test_models_fit():
    from src.data.loader import load_csv, filter_team_rows
    from src.data.preprocessor import build_dataset, split_data, get_scaler
    from src.models.svm_model import SVMModel
    from src.models.rf_model import RFModel

    df = filter_team_rows(load_csv())
    X, y = build_dataset(df, nan_strategy='drop')
    X_train, X_test, y_train, y_test = split_data(X, y)
    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for ModelClass in [SVMModel, RFModel]:
        model = ModelClass()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)
        assert len(preds) == len(y_test)
        assert proba.shape == (len(y_test), 2)
        assert model.best_params_ is not None


def test_metrics():
    from src.evaluation.metrics import compute_metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])
    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1


def test_nan_report():
    from src.data.loader import load_csv, filter_team_rows
    from src.data.preprocessor import get_nan_report, get_nan_league_report
    df = filter_team_rows(load_csv())
    report = get_nan_report(df)
    assert len(report) == 16
    assert 'nan_count' in report.columns
    league_report = get_nan_league_report(df)
    assert len(league_report) > 0
    assert 'league' in league_report.columns
