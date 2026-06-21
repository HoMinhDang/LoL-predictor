import argparse
import json
import os
import joblib
import mlflow
import mlflow.sklearn

from src.config import (
    EARLY_FEATURES, MODEL_NAMES, ROOT_DIR,
    EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
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


def train_and_save(league=None, output_dir=None, use_mlflow=True):
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, 'models_saved')
    os.makedirs(output_dir, exist_ok=True)

    if league is None:
        display_name = 'all_leagues'
    else:
        display_name = league

    df = load_csv()
    df = filter_team_rows(df)
    df = filter_by_league(df, league)

    X, y = build_dataset(df, nan_strategy='drop')
    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    metadata = {
        'features': EARLY_FEATURES.copy(),
        'league': display_name,
        'available_models': list(MODEL_NAMES),
        'n_samples': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

    for model_name in MODEL_NAMES:
        model = MODEL_CLASSES[model_name]()
        gs = model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        metrics = compute_metrics(y_test, y_pred, y_proba)

        joblib.dump(model.model, os.path.join(output_dir, f'{model_name}_model.pkl'))
        metadata[model_name] = {
            'best_params': model.best_params_,
            'cv_best_score': gs.best_score_,
            'metrics': metrics,
        }

        if use_mlflow:
            with mlflow.start_run(run_name=f"prod_{model_name}_{display_name}"):
                mlflow.log_param("league", display_name)
                mlflow.log_param("model", model_name)
                mlflow.log_param("source", "train.py")
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
                        serialization_format='cloudpickle',
                    )
                else:
                    mlflow.sklearn.log_model(model.model, f"model_{model_name}")

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Models saved to {output_dir}/")
    print(f"Total samples: {len(X)} (train: {len(X_train)}, test: {len(X_test)})")
    for model_name in MODEL_NAMES:
        m = metadata[model_name]
        print(f"  {model_name}: accuracy={m['metrics']['accuracy']:.4f}, "
              f"roc_auc={m['metrics']['roc_auc']:.4f}")

    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and save models with MLflow tracking")
    parser.add_argument('--league', type=str, default=None,
                        help='League name. Default: all.')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for saved models.')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging.')
    args = parser.parse_args()

    train_and_save(
        league=args.league,
        output_dir=args.output_dir,
        use_mlflow=not args.no_mlflow,
    )
