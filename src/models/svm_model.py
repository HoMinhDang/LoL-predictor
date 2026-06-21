from sklearn.svm import SVC
from src.models.base import BaseModel
from src.config import SVM_PARAM_GRID


class SVMModel(BaseModel):
    def __init__(self):
        super().__init__('svm')

    def build_estimator(self, **params):
        return SVC(probability=True, random_state=42, **params)

    def get_param_grid(self):
        return SVM_PARAM_GRID
