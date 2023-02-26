import sys, os, json, time
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)

from sklearn.metrics import f1_score

from config import PATH_WORKING, SEED
from builder import build_XGBClassifier
from dataprep import load_features
from runs import run_kfolds, run_kfolds_multiclass
from thresholding import find_single_threshold


def save_models(folder, models, data=None):
    os.makedirs(folder, exist_ok=True)
    for key, model in models.items():
        model.save_model( f'{folder}/{key}.json' )
    if data is not None:
        with open(f'{folder}/data.json', 'w') as f:
            f.write( json.dumps(data, indent=4) )


def train_single(): 
    """Train function."""    
    
    # load data where:
    # features - pd.df with MultiIndex ('session_id', 'level_group')
    # targets - pd.df with MultiIndex ('session_id', 'question')
    features, targets = load_features()

    # tuned params
    params = json.load(open(f'{PATH_WORKING}/hyperopt/best_params_00.json', 'r'))    
    models = {}    
    tic = time.time() 

    # iterate through questions
    for q in range(1, 19):            

        if q <= 3: 
            level_group = '0-4'
        elif q <= 13: 
            level_group = '5-12'
        else: 
            level_group = '13-22'

        # train data
        train_features = features.xs(level_group, level='level_group')
        train_sessions = train_features.index
        X_train = train_features.values
        y_train = targets.xs(q, level='question').loc[train_sessions, 'correct']
        
        # train model        
        model = build_XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=0)

        models[f'model_q{q}'] = model
        
        runtime = round(time.time()-tic)
        logging.info(f' question {q} runtime {runtime}')

    #save_models(f'{PATH_WORKING}/model/xgb', models, data=None)


def train_kfolds():

    logging.info(' Running train_kfolds ...')
       
    # load data
    features, targets = load_features()
    params = json.load(open(f'{PATH_WORKING}/hyperopt/best_params_01.json', 'r'))

    values, models = run_kfolds(features, targets, params=params, n_splits=10, questions=tuple(range(1,19)))

    score, threshold = find_single_threshold(values)
    score = round(score, 4)
    print('score:', score)

    save_models(f'{PATH_WORKING}/xgb_01', models, data={'threshold': threshold, 'score': score, 'params': params})


def train_kfolds_multiclass():

    logging.info(' Running train_kfolds_multiclass ...')
    tic = time.time()
       
    # load data
    features, targets = load_features()
    params = json.load(open(f'{PATH_WORKING}/hyperopt/best_params_02_multiclass.json', 'r'))
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
    """
    qgroups = [
        # level_group = '13-22'
        (14,15,16,17,18),
        # level_group = '5-12'
        (4,5,6,7,8),
        (9,10,11,12,13),
        # level_group = '0-4' 
        (1,2,3),         
    ]"""

    values, models = run_kfolds_multiclass(features, targets, params=params, n_splits=10, qgroups=qgroups)

    y_true = [values[f'y_true_q{q}'] for q in range(1, 19) if f'y_true_q{q}' in values]
    y_pred = [values[f'y_pred_q{q}'] for q in range(1, 19) if f'y_pred_q{q}' in values]

    score = f1_score(np.concatenate(y_true), np.concatenate(y_pred), average='macro')
    score = round(score, 4)
    print('score:', score)
    logging.info(f'runtime {round(time.time()-tic)}')

    save_models(f'{PATH_WORKING}/xgb_0_multiclass2', models, data={'score': score, 'params': params})


if __name__ == "__main__":
    """
    Usage:
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu

    python other-research/kaggle-2023-03-predict-student-performance-from-game-play/train.py
    """
    #train_single()
    #train_kfolds()
    train_kfolds_multiclass()
        

# runtime
# ---------- XGB Version 0.90
# hist     1643
# gpu_hist  641  
# ---------- XGB Version 1.6.2   
# gpu_hist  645


# qgroup mostly 2: score: 0.6447 runtime 12857
# qgroup mostly 5: score: 0.6257 runtime 27467