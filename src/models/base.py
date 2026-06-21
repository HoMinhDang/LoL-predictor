import abc
import joblib
import mlflow
import mlflow.sklearn


class BaseModel(abc.ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.best_params_ = None

    @abc.abstractmethod
    def build_estimator(self, **params):
        pass

    @abc.abstractmethod
    def get_param_grid(self):
        pass

    def fit(self, X, y):
        from sklearn.model_selection import GridSearchCV
        from src.config import CV_FOLDS

        estimator = self.build_estimator()
        param_grid = self.get_param_grid()
        gs = GridSearchCV(
            estimator, param_grid, cv=CV_FOLDS,
            scoring='roc_auc', n_jobs=-1, verbose=0
        )
        gs.fit(X, y)
        self.model = gs.best_estimator_
        self.best_params_ = gs.best_params_
        return gs

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

    def log_mlflow(self, params: dict = None, metrics: dict = None, model_artifact: str = None):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if model_artifact:
            mlflow.sklearn.log_model(self.model, model_artifact)
