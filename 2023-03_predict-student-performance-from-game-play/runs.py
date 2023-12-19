import sys, os, json
import numpy as np
import pandas as pd
import itertools

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.INFO)

from sklearn.model_selection import KFold, GroupKFold, GroupShuffleSplit, train_test_split

from builder import build_XGBClassifier


def add_values(values, key, y):
    if key not in values:
        values[key] = []
    values[key].append(y)
    return values

def create_multilabels(*args):    
    bin2label = {}
    label2bin = {}
    for i, tpl in enumerate(itertools.product(*args)):
        bin2label[tpl] = i 
        label2bin[i] = tpl
    return bin2label, label2bin

def convert_bin2label(lst_targets):
    """
    Args:
        lst_targets - list of binary targets - multi labels problem
    Returns:
        labels - 1d array for multi class problem
    """
    bin2label, _ = create_multilabels(*[[0,1] for _ in range(len(lst_targets))])
    labels = [bin2label[tpl] for tpl in zip(*lst_targets)]
    return np.array(labels, dtype='int32')

def convert_label2bin(labels, n_outputs):
    """
    Args:
        labels - 1d array of labels
        n_outputs - number of targets in output
    Returns:
        lst_targets - list of binary targets NOTE: Only predictions (not probabilities)
    """    
    _, label2bin = create_multilabels(*[[0,1] for _ in range(n_outputs)])
    labels = labels.ravel()
    arr = np.zeros((n_outputs, labels.size), dtype='int32')
    for j, lbl in enumerate(labels):
        arr[:,j] = label2bin[lbl]
    return list(arr)


def run_kfolds(features, targets, params=None, models=None, n_splits=10, questions=None):
    """
    features - pd.df with MultiIndex ('session_id', 'level_group')
    targets - pd.df with MultiIndex ('session_id', 'question')
    params or models should be provided depending on the goal, train or predict respectively
    Returns:
        oof predictions
    """    
    assert params is None or models is None
    assert params is not None or models is not None  
    
    if questions is None:
        questions = tuple(range(1,19)) 
    
    # init {'0-4': (1, 4), '5-12': (4, 14), '13-22': (14, 19)}
    values = {} # oof predictions
    if models is None:
        models = {}
       
    session_ids = features.xs(slice(None), level='level_group').index.unique()
    kfolds = GroupKFold(n_splits=n_splits).split(X=session_ids, groups=session_ids)  
    
    for k, (train_index, test_index) in enumerate(kfolds):        
        
        # iterate through questions range(1,19)
        for q in questions: 

            logging.info(f' Fold {k} | question {q}')

            if q <= 3: 
                level_group = '0-4'
            elif q <= 13: 
                level_group = '5-12'
            else: 
                level_group = '13-22'

            group_features = features.xs(level_group, level='level_group')

            # test data
            test_features = group_features.iloc[test_index]
            test_sessions = test_features.index
            X_test = test_features.values
            y_test = targets.xs(q, level='question').loc[test_sessions, 'correct']

            name = f'model_fold{k}_q{q}'
            if name not in models:
                # train data
                train_features = group_features.iloc[train_index]
                train_sessions = train_features.index
                X_train = train_features.values
                y_train = targets.xs(q, level='question').loc[train_sessions, 'correct'] 
            
                # train model        
                model = build_XGBClassifier(**params)
                model.fit(X_train, y_train, verbose=0)   #, eval_set=[(X_test, y_test)]
                models[name] = model
                
            model = models[f'model_fold{k}_q{q}']

            # compute predictions
            y_pred = model.predict_proba(X_test)[:,1]
            
            values = add_values(values, f'y_true_q{q}', y_test)
            values = add_values(values, f'y_pred_q{q}', y_pred)
    
    for q in questions:
        values[f'y_true_q{q}'] = np.concatenate(values[f'y_true_q{q}'])
        values[f'y_pred_q{q}'] = np.concatenate(values[f'y_pred_q{q}'])
    
    return values, models


def run_kfolds_multiclass(features, targets, params=None, models=None, n_splits=10, qgroups=None):
    """
    features - pd.df with MultiIndex ('session_id', 'level_group')
    targets - pd.df with MultiIndex ('session_id', 'question')
    params or models should be provided depending on the goal, train or predict respectively
    Returns:
        oof predictions
    """    
    assert params is None or models is None
    assert params is not None or models is not None    

    def prepare_features(q, train_index, test_index):

        if q <= 3: 
            level_group = '0-4'
        elif q <= 13: 
            level_group = '5-12'
        else: 
            level_group = '13-22'

        group_features = features.xs(level_group, level='level_group')

        # train data
        train_features = group_features.iloc[train_index]
        train_sessions = train_features.index
        X_train = train_features.values
        #y_train = targets.xs(q, level='question').loc[train_sessions, 'correct'] 

        # test data
        test_features = group_features.iloc[test_index]
        test_sessions = test_features.index
        X_test = test_features.values
        #y_test = targets.xs(q, level='question').loc[test_sessions, 'correct']

        return X_train, train_sessions, X_test, test_sessions
   
    # init
    values = {} # oof predictions
    if models is None:
        models = {}        

    session_ids = features.xs(slice(None), level='level_group').index.unique()
    kfolds = GroupKFold(n_splits=n_splits).split(X=session_ids, groups=session_ids)
    
    for k, (train_index, test_index) in enumerate(kfolds):        
        
        # iterate through questions - list of groupped q
        for qgroup in qgroups: 

            logging.info(f' Fold {k} | group {qgroup}')

            # there can be only one group, so features are common
            X_train, train_sessions, X_test, test_sessions = prepare_features(qgroup[0], train_index, test_index)

            # labels for the group of questions           
            lst_y_train = [targets.xs(q, level='question').loc[train_sessions, 'correct'] for q in qgroup]
            lst_y_test = [targets.xs(q, level='question').loc[test_sessions, 'correct'] for q in qgroup]

            name = 'model_fold{}_q{}'.format(k, '_'.join([str(q) for q in qgroup])) # model name

            if name not in models:            
                # train model        
                y_train = convert_bin2label(lst_y_train)
                params['objective'] = 'multi:softmax'
                params['num_class'] = 2**len(qgroup)
                model = build_XGBClassifier(**params)
                model.fit(X_train, y_train, verbose=0)
                models[name] = model
                
            model = models[name]

            # compute predictions
            #labels = model.predict_proba(X_test)#[:,1]
            labels = model.predict(X_test)#[:,1]

            lst_y_pred = convert_label2bin(labels, n_outputs=len(qgroup))

            for q, y_test, y_pred in zip(qgroup, lst_y_test, lst_y_pred):
                values = add_values(values, f'y_true_q{q}', y_test)
                values = add_values(values, f'y_pred_q{q}', y_pred)
    
    for q in itertools.chain(*qgroups):
        values[f'y_true_q{q}'] = np.concatenate(values[f'y_true_q{q}'])
        values[f'y_pred_q{q}'] = np.concatenate(values[f'y_pred_q{q}'])
    
    return values, models