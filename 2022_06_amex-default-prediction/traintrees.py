import os
import numpy as np

from utils import amex_metric_mod
from config import PATH_WORKING, SEED


def load_data(folder='model'):    

    X_train = np.load(os.path.join(PATH_WORKING, 'results', folder, 'X_train.npy'))
    y_train = np.load(os.path.join(PATH_WORKING, 'results', folder, 'y_train.npy'))

    X_valid = np.load(os.path.join(PATH_WORKING, 'results', folder, 'X_valid.npy'))
    y_valid = np.load(os.path.join(PATH_WORKING, 'results', folder, 'y_valid.npy'))

    return X_train, y_train, X_valid, y_valid


def load_data2():

    import dataload
    dtrain, dvalid, _ = dataload.load(train=True, valid=True, test=False)

    X_train = dtrain['X'].sum(axis=1) / dtrain['mask'].sum(axis=1, keepdims=True) 
    y_train = dtrain['y']

    X_valid = dvalid['X'].sum(axis=1) / dvalid['mask'].sum(axis=1, keepdims=True)
    y_valid = dvalid['y']

    return X_train, y_train, X_valid, y_valid


def train_lgbm():

    X_train, y_train, X_valid, y_valid = load_data()

    import lightgbm as lgb    

    dtrain = lgb.Dataset(data=X_train, label=y_train.ravel())
    dvalid = lgb.Dataset(data=X_valid, label=y_valid.ravel())

    
    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "reg_lambda": 50,
        "min_child_samples": 2400,
        "num_leaves": 220,
        "colsample_bytree": 0.19,
        "random_state": 1,
        'verbose': -1
    }

    def lgb_amex_metric(y_pred, dtrain):
        """Custom LGBM eval metric: amex metric"""
        y_true = dtrain.get_label()
        return 'amex_metric', amex_metric_mod(y_true, y_pred), True

    model = lgb.train(
        params = params,
        train_set = dtrain,
        valid_sets = [dvalid],
        feval = lgb_amex_metric,
        num_boost_round = 2000, #10500,
        early_stopping_rounds = 100, # if used best model with feval returned, otherwise not!
        verbose_eval = 10
    )

    y_pred = model.predict(X_valid)
    print('LGBM amex metric:', amex_metric_mod(y_valid, y_pred)) # 0.787892496785727


def train_xgb():

    X_train, y_train, X_valid, y_valid = load_data()

    import xgboost as xgb
    print('XGB Version', xgb.__version__)

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)

    # XGB MODEL PARAMETERS
    params = { 
        'max_depth': 4, 
        'learning_rate': 0.03, 
        'subsample': 0.8,
        'colsample_bytree': 0.6, 
        'objective': 'binary:logistic',
        'random_state': 222
    }

    def xgb_amex_metric(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label()
        return 'amex_metric', 1 - amex_metric_mod(y_true, y_pred)

    model = xgb.train(
        params = params,
        dtrain = dtrain,
        evals = [(dvalid, 'valid')], #[(dtrain,'train'), (dvalid,'valid')],
        feval = xgb_amex_metric,
        num_boost_round = 2000,
        early_stopping_rounds = 100,
        verbose_eval = 10
    ) 

    #model.save_model(f'XGB_v{VER}_fold{fold}.xgb')

    y_pred = model.predict(dvalid)
    print('XGB amex metric:', amex_metric_mod(y_valid, y_pred)) # 0.7861501205056763


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-01-amex/traintrees.py
    """
    train_lgbm()
    #train_xgb()