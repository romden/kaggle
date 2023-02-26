import sys, os, json
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

# ------ hyperopt------
import imp
imp.load_package('py4j','/opt/anaconda/4.2.0/lib/python3.5/site-packages/py4j')

pkgs = ['/home/BF0772/aapf/libs/hyperopt-master']
for pkg in pkgs:
    if pkg not in sys.path:
        sys.path.append(pkg)

from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.fmin import generate_trials_to_calculate
# ------------------

from config import PATH_WORKING, SEED
from dataprep import load_features
from runs import run_kfolds, run_kfolds_multiclass
from thresholding import find_single_threshold


def init_space():
    
    # define a search space
    space = {
        'n_estimators': hp.randint('n_estimators', 100, 900),

        'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
        'max_depth': hp.randint('max_depth', 2, 16),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),

        'reg_lambda': hp.uniform('reg_lambda', 0, 2),
        'reg_alpha': hp.uniform('reg_alpha', 0, 2),

        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1),
        'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1),
        'colsample_bynode': hp.uniform('colsample_bynode', 0.3, 1),                       
    }

    return space


def main():
    """Searches for best parameter setting for classifier using hyperopt library"""
    # load data
    features, targets = load_features()

    qgroups = [
        (1,2,3), # level_group = '0-4'
        (4,5),   # level_group = '5-12'
        (6,7),
        (8,9),
        (10,11),
        (12,13),
        (14,15), # level_group = '13-22'
        (16,17,18), 
    ]

    # objective function for hyperopt to optimize 
    def obj_fn(params):
        values, _ = run_kfolds(features, targets, params=params) # returns oof predictions
        score, _ = find_single_threshold(values)

        #values, _ = run_kfolds_multiclass(features, targets, params=params, qgroups=qgroups, n_splits=5)      
        #y_true = [values[f'y_true_q{q}'] for q in range(1, 19) if f'y_true_q{q}' in values]
        #y_pred = [values[f'y_pred_q{q}'] for q in range(1, 19) if f'y_pred_q{q}' in values]
        #score = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')  
        
        return -score

    # search space for params
    space = init_space()

    # minimize the objective over the space
    best = fmin(
        fn=obj_fn,
        space=space,
        algo=tpe.suggest,
        max_evals=30
    )    
    
    best_params = space_eval(space, best)
    print('hyperopt:')
    print(best_params)
    if True:
        data = {k: v.item() if isinstance(v, (np.generic, np.bool_)) else v for k, v in best_params.items()}
        filename = f'{PATH_WORKING}/hyperopt/best_params_01.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    """
    Usage:
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu

    python other-research/kaggle-2023-03-predict-student-performance-from-game-play/tune.py
    """
    main()


"""
best_params_00 threshold: 0.6200000000000002 score: 0.6857482328946108 LB = 0.685
"""