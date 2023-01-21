import os
import gc
import dill

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

from utils import correlation_score
from config import *

# data directory
DATA_DIR = os.path.join(PATH_WORKING, 'data_cite_x512')
pipe_y = dill.load(open(os.path.join(DATA_DIR, 'pipe_y.dill'), 'rb'))

# model directory with results
MODEL_DIR = os.path.join(PATH_WORKING, 'lgb_cite_x512')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class CFG:
    n_folds = 10


def train_lgbm():

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')
    
    scores = [] 

    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        init_params = {
            'num_iterations': 1000,
            'learning_rate': 0.05,            
            'max_depth': 10,
            'num_leaves': 200,
            'reg_alpha': 0.03, 
            'reg_lambda': 0.002,             
            #'subsample': 0.6,            
            #'min_data_in_leaf': 263,
            'colsample_bytree': 0.8,
            'random_seed': 4243,
            'early_stopping_round': 100,
            'verbosity': -1,
        }

        models = []
        y_pred = np.zeros(y_test.shape)

        for j in range(y_train.shape[1]):
            
            fit_params = {
                'X': X_train,
                'y': y_train[:,j],
                'eval_set': [(X_test, y_test[:,j])],
                'eval_metric': 'rmse',
                "verbose": False,
            }            
            
            model = lgb.LGBMRegressor(**init_params).fit(**fit_params)

            models.append(model)     

            y_pred[:,j] = model.predict(X_test)

            print(f'Feature: {j}   MSE: {mean_squared_error(y_test[:,j], y_pred[:,j])}')
        
        dill.dump(models, open(os.path.join(MODEL_DIR, f'models_{k}.dill'), 'wb'))

        y_pred = pipe_y.inverse_transform(y_pred)    

        score = correlation_score(y_true[test_index], y_pred)
        scores.append(score)
        print('Fold:', k, 'Score:', score)
           
        del X_train, X_test, y_train, y_test, y_pred
        gc.collect()
        #break
    
    print('CV score:', np.mean(scores))


class ModelWrapper:

    def __init__(self, models):
        self.models = models
    
    def predict(self, X):
        y = np.zeros((X.shape[0], len(self.models)))
        for j, model in enumerate(self.models):
            y[:,j] = model.predict(X)
        return y


def predict_test():

    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_pred = None     

    for k in range(CFG.n_folds):
        print('Fold:', k)
        
        trees = dill.load(open(os.path.join(MODEL_DIR, f'models_{k}.dill'), 'rb'))

        model =  ModelWrapper(trees)
          
        y_fold = model.predict(X_test)

        if y_pred is None:
            y_pred = y_fold
        else:
            y_pred += y_fold
    
    y_pred /= CFG.n_folds

    y_pred = pipe_y.inverse_transform(y_pred) 

    assert y_pred.shape == SHAPES['cite']

    np.save(os.path.join(MODEL_DIR, 'y_pred.npy'), y_pred)


def predict_folds():

    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))

    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):   
        print('Fold:', k)  

        X_test = X[test_index]
        
        trees = dill.load(open(os.path.join(MODEL_DIR, f'models_{k}.dill'), 'rb'))

        model =  ModelWrapper(trees)
          
        y_fold = model.predict(X_test)
        y_fold = pipe_y.inverse_transform(y_fold)

        assert y_fold.shape[1] == SHAPES['cite'][1]

        np.save(os.path.join(MODEL_DIR, f'y_fold_{k}.npy'), y_fold)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/train_lgb_cite.py
    """
    #train_lgbm()
    #predict_test()
    predict_folds()


#CV score: 0.8961377694285847