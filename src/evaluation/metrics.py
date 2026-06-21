import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from src.config import ROOT_DIR  # noqa: E402


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics['roc_auc'] = 0.5
    return metrics


def plot_confusion_matrix(y_true, y_pred, model_name: str, league: str, filename: str = None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} - {league}')
    if filename is None:
        filename = os.path.join(ROOT_DIR, 'reports', f'cm_{model_name}_{league}.png')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    return filename


def plot_roc_curve(y_true, y_proba, model_name: str, league: str, filename: str = None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {model_name} - {league}')
    plt.legend(loc='lower right')
    if filename is None:
        filename = os.path.join(ROOT_DIR, 'reports', f'roc_{model_name}_{league}.png')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    return filename
