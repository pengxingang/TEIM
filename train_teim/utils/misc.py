import numpy as np
from easydict import EasyDict
import yaml
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, median_absolute_error, roc_auc_score
from scipy.stats import pearsonr

def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def load_model_from_ckpt(path, model: torch.nn.Module):
    pl_model = torch.load(path)
    if 'state_dict' in pl_model:
        pl_model = pl_model['state_dict'] # from lightning module
    state_dict = {k[k.find('.')+1:]: v for k, v in pl_model.items()}
    model.load_state_dict(state_dict)
    return model

def calc_auc_aupr(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    return auc, aupr

def get_scores_dist(y_true, y_pred, y_mask):
    coef, mae, mape = [], [], []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):
        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        try:
            coef_, _ = pearsonr(y_true_, y_pred_)
        except Exception:
            coef_ = np.nan
        coef.append(coef_)

        mae_ = median_absolute_error(y_true_, y_pred_)
        mae.append(mae_)

        mape_ = np.median(np.abs((np.array(y_true_) - np.array(y_pred_)) / np.array(y_true_)))
        mape.append(mape_)
    avg_coef = np.nanmean(coef)
    avg_mae = np.mean(mae)
    avg_mape = np.mean(mape)
    return [avg_coef, avg_mae, avg_mape], [coef, mae, mape]


def get_scores_contact(y_true, y_pred, y_mask):
    coef = []
    for y_true_, y_pred_, y_mask_ in zip(y_true, y_pred, y_mask):
        # y_true_, y_pred_, y_mask_ = y_true_.reshape([-1]), y_pred_.reshape([-1]), y_mask_.reshape([-1]).astype('bool')
        y_true_ = y_true_[y_mask_.astype('bool')]
        y_pred_ = y_pred_[y_mask_.astype('bool')]
        try:
            coef_ = roc_auc_score(y_true_, y_pred_)
        except Exception:
            coef_ = np.nan
        coef.append(coef_)

    avg_coef = np.mean(coef)
    return [avg_coef], [coef]