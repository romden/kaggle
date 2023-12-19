import numpy as np
import pandas as pd
import sys, os, json, time

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

from config import PATH_WORKING, SEED
from dataprep import load_features
from runs import run_kfolds


def load_models():
    import xgboost as xgb
    folder = f'{PATH_WORKING}/xgb_00'
    models = {}
    for k in range(10):
        for q in range(1,19):
            key = f'model_fold{k}_q{q}'
            model = xgb.XGBClassifier()
            model.load_model(f'{folder}/{key}.json')
            models[key] = model
    return models


def find_best_vec(y_true, y_pred, t_min=0.4, t_max=0.81, t_step=0.01):
    """Find best score and threshold vectorized implementation"""
    thresholds = np.arange(t_min, t_max, t_step)
    m = thresholds.size

    y_true = y_true.ravel()
    y_pred = np.tile(y_pred.reshape(-1, 1), (1, m))

    preds = (y_pred > thresholds.reshape(1,-1)).astype('int8')

    scores = [f1_score(y_true, preds[:,j], average='macro') for j in range(m)]

    idx = np.argmax(scores)
    best_score = scores[idx]
    best_threshold = thresholds[idx]
    return best_score, best_threshold


def find_best_iter(y_true, y_pred, t_min=0.4, t_max=0.81, t_step=0.01):
    """Find best score and threshold iterative implementation"""
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    best_score = -float('inf')
    best_threshold = None

    for threshold in np.arange(t_min, t_max, t_step):
        y_hat = (y_pred > threshold).astype('int8')
        score = f1_score(y_true, y_hat, average='macro')
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_score, best_threshold


def find_separate_thresholds(values, n_qns=18):
    """for each question, find best threshold separately"""
    thresholds = []
    trues = []
    preds = []
    for q in range(1, n_qns+1):
        y_true = values[f'y_true_q{q}']
        y_pred = values[f'y_pred_q{q}']
        _, threshold = find_best_vec(y_true, y_pred)
        thresholds.append(threshold)
        trues.append(y_true)
        preds.append((y_pred > threshold).astype('int8')) 
    score = f1_score(np.concatenate(trues), np.concatenate(preds), average='macro')
    return score, thresholds


def find_single_threshold(values):
    trues = [values[f'y_true_q{q}'] for q in range(1, 19) if f'y_true_q{q}' in values]
    preds = [values[f'y_pred_q{q}'] for q in range(1, 19) if f'y_pred_q{q}' in values]
    score, threshold = find_best_vec(np.concatenate(trues), np.concatenate(preds))
    return score, threshold


def eval_thresholds(values, thresholds):
    trues = []
    preds = []
    for key, threshold in thresholds.items():
        y_true = values[f'y_true_{key}']
        y_pred = values[f'y_pred_{key}']
        y_hat = (y_pred > threshold).astype('int8')
        trues.append(y_true)
        preds.append(y_hat)
    score = f1_score(np.concatenate(trues), np.concatenate(preds), average='macro')
    return score


def line_search(values, thresholds, question):
    # current values    
    score = eval_thresholds(values, thresholds)
    threshold = thresholds[question]
    gain = 0
    # define search range
    delta = 0.1
    t_min = max(0, threshold-delta)
    t_max = min(1, threshold+delta)
    t_step = 0.01
    for t in np.arange(t_min, t_max, t_step):
        tmp_thresh = dict(thresholds)
        tmp_thresh[question] = t
        tmp_score = eval_thresholds(values, tmp_thresh)
        tmp_gain = max(0, tmp_score-score)
        if tmp_gain > gain:
            gain = tmp_gain
            threshold = t
    return (question, gain, threshold)


def search_greedy():
    # compute oof values
    values, _ = run_kfolds(*load_features(), models=load_models())
    # init 
    thresholds = {f'q{q}': 0.62 for q in range(1,19)} # init values
    tol = 1e-4 # stopping criterion
    it = 1 # iteration counter
    prev = None
    # iteratively search for improvement
    while True:
        tic = time.time()
        results = [line_search(values, thresholds, question) for question in thresholds.keys() if question != prev]
        if not any([res[1] > tol for res in results]):
            # stop search
            print('stop due to stopping criterion is met')
            break
        # sort in descending order based on gain
        results.sort(key=lambda x: -x[1])
        question, gain, threshold = results[0] # take max gain
        # update threshold and continue search
        thresholds[question] = threshold
        prev = question
        # verbose
        runtime = round(time.time()-tic)
        print(f'iteration: {it}\t| runtime: {runtime}\t| question: {question}\t| gain: {gain:.4f}')
     
    print('score', eval_thresholds(values, thresholds))
    print('params', thresholds)


def submission_threshold(train=True):     
    # load data
    features, targets = load_features()
    if train:
        # train and comput oof values
        params = json.load(open(f'{PATH_WORKING}/hyperopt/best_params_00.json', 'r'))
        values, models = run_kfolds(features, targets, params=params)
    else:
        # given models only compute oof values
        models = load_models()
        values, _ = run_kfolds(features, targets, models=models)

    # compute    
    #score, thresholds = find_separate_thresholds(values)
    score, thresholds = find_single_threshold(values)

    print('score:', score)
    print('thresholds:', thresholds)

    if train:
        folder = f'{PATH_WORKING}/xgb_00'
        os.makedirs(folder, exist_ok=True)
        for key, model in models.items():
            filename = f'{folder}/{key}.json'
            model.save_model(filename)
        filename = f'{folder}/data.json'        
        data = {'score': round(score, 4)}
        data['thresholds'] = [round(t, 2) for t in thresholds] if isinstance(thresholds, list) else [round(thresholds, 2)]
        with open(filename, 'w') as f:
            f.write(json.dumps(data, indent=4))


if __name__ == "__main__":
    """
    Usage:
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu

    python other-research/kaggle-2023-03-predict-student-performance-from-game-play/thresholding.py
    """  
    #search_greedy()  
    #submission_threshold(train=False)    


"""
best_params_00 threshold: 0.6200000000000002 score: 0.6857482328946108 LB = 0.685

score 0.6863899049275841 greedy improvement
"""