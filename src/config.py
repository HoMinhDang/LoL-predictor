import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "2026_LoL_esports_match_data_from_OraclesElixir.csv")

MLFLOW_TRACKING_URI = "sqlite:///" + ROOT_DIR.replace("\\", "/") + "/mlruns.db"
EXPERIMENT_NAME = "early_game_winner_prediction"

EARLY_FEATURES = [
    'golddiffat15', 'xpdiffat15', 'csdiffat15',
    'killsat15', 'assistsat15', 'deathsat15',
    'opp_killsat15', 'opp_assistsat15', 'opp_deathsat15',
    'firstblood', 'firstdragon', 'firstherald',
    'firsttower', 'firstmidtower', 'firsttothreetowers',
    'void_grubs',
]
# Excluded (full-game stats, not early-game):
# 'turretplates', 'opp_turretplates', 'team kpm', 'ckpm'

TARGET = 'result'

SVM_PARAM_GRID = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

RF_PARAM_GRID = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
}

XGB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 6, 9],
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

MODEL_NAMES = ['svm', 'rf', 'xgb']
