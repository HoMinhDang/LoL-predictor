from xgboost import XGBClassifier
from src.models.base import BaseModel
from src.config import XGB_PARAM_GRID


class XGBModel(BaseModel):
    def __init__(self):
        super().__init__('xgb')

    def build_estimator(self, **params):
        return XGBClassifier(random_state=42, **params)

    def get_param_grid(self):
        return XGB_PARAM_GRID
