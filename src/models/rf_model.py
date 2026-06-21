from sklearn.ensemble import RandomForestClassifier
from src.models.base import BaseModel
from src.config import RF_PARAM_GRID


class RFModel(BaseModel):
    def __init__(self):
        super().__init__('rf')

    def build_estimator(self, **params):
        return RandomForestClassifier(random_state=42, **params)

    def get_param_grid(self):
        return RF_PARAM_GRID
