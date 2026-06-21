import mlflow
from src.config import (
    EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAMES
)
from src.data.loader import load_csv, filter_team_rows, filter_by_league
from src.data.preprocessor import build_dataset, split_data, get_scaler
from src.models.svm_model import SVMModel
from src.models.rf_model import RFModel
from src.models.xgb_model import XGBModel
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curve

MODEL_CLASSES = {
    'svm': SVMModel,
    'rf': RFModel,
    'xgb': XGBModel,
}


def _train_single(df, display_name, use_mlflow):
    X, y = build_dataset(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    for model_name in MODEL_NAMES:
        model_cls = MODEL_CLASSES[model_name]
        model = model_cls()

        gs = model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        if use_mlflow:
            with mlflow.start_run(run_name=f"{model_name}_{display_name}"):
                mlflow.log_param("league", display_name)
                mlflow.log_param("model", model_name)
                mlflow.log_params(dict(gs.best_params_))
                mlflow.log_metric("cv_best_score", gs.best_score_)
                mlflow.log_metrics(metrics)

                cm_path = plot_confusion_matrix(y_test, y_pred, model_name, display_name)
                roc_path = plot_roc_curve(y_test, y_proba, model_name, display_name)
                mlflow.log_artifact(cm_path)
                mlflow.log_artifact(roc_path)
                if model_name == 'xgb':
                    mlflow.sklearn.log_model(
                        model.model, f"model_{model_name}",
                        serialization_format='cloudpickle'
                    )
                else:
                    mlflow.sklearn.log_model(model.model, f"model_{model_name}")

        results[model_name] = {
            'best_params': model.best_params_,
            'cv_best_score': gs.best_score_,
            'metrics': metrics,
            'model': model,
        }

    return results


def train_for_league(league: str = None, run_name: str = None, use_mlflow: bool = True):
    df = load_csv()
    df = filter_team_rows(df)
    df = filter_by_league(df, league)

    if league is None:
        display_name = run_name or 'all_leagues'
    else:
        display_name = run_name or league

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    return _train_single(df, display_name, use_mlflow)
