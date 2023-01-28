import sys, os, json, dill, gc
import numpy as np
import pandas as pd

#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import datagen
import losses
from bayesian_builder3 import CellModel
from utils import correlation_score
from config import *

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

# model directory with results
PATH_MODEL = os.path.join(PATH_WORKING, 'single_model')
if not os.path.exists(PATH_MODEL):
    os.mkdir(PATH_MODEL)


def load_data(dstype='cite'):    
    if dstype == 'cite':
        # data directory
        PATH_X = os.path.join(PATH_WORKING, 'data_cite_x512')
        PATH_Y = os.path.join(PATH_WORKING, 'data_cite_y140')
        # load data
        y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')    

    elif dstype == 'multi':
        # data directory
        PATH_X = os.path.join(PATH_WORKING, 'data_multi_x512')
        PATH_Y = os.path.join(PATH_WORKING, 'data_multi_y512')
        # load data
        y_true = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values.astype('float32') 

    features = {}
    for key in ['train', 'test', 'unlabeled']:
        X = np.load(os.path.join(PATH_X, f'X_{key}.npy'))
        meta = np.load(os.path.join(PATH_WORKING, f'metadata_{dstype}', f'X_{key}.npy'))
        features[key] = np.concatenate([X, meta], axis=1)

    targets = {
        'train': np.load(os.path.join(PATH_Y, 'y_train.npy')),
        'pipe': dill.load(open(os.path.join(PATH_Y, 'pipe_y.dill'), 'rb')),
        'true': y_true
    }

    return features, targets


def get_train_xy(X_train, X_unlabeled, X_test, y_train):
    xx = np.concatenate([X_train, X_unlabeled, X_test], axis=0)
    y1 = np.concatenate([y_train, np.ones((y_train.shape[0],1))], axis=1)  
    y2 = np.zeros((X_unlabeled.shape[0],y_train.shape[1]+1))
    y3 = np.zeros((X_test.shape[0],y_train.shape[1]+1))
    yy = np.concatenate([y1, y2, y3], axis=0)
    return xx, yy


def cv():    

    folder = os.path.join(PATH_MODEL, 'cv')
    if not os.path.exists(folder):
        os.mkdir(folder)    

    # load data
    data = {key: dict(zip(('features', 'targets'), load_data(key))) for key in ['cite', 'multi']}

    params = {
        'n_folds': 5,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'epochs': 100
    }

    scores = {key: [] for key in ['cite', 'multi']}
    
    kfolds = {}
    for key in ['cite', 'multi']:
        X = data[key]['features']['train']
        kfolds[key] = iter(KFold(n_splits=params['n_folds'], shuffle=True, random_state=SEED*4).split(X))

    for k in range(params['n_folds']):
        
        # =========================== data
        features = {}
        targets = {}
        data_size = {} # for tfp model build
        test_indexes = {}
        for key in ['cite', 'multi']:
            train_index, test_index = next(kfolds[key])
            test_indexes[key] = test_index
            data_size[key] = len(train_index)

            X_train = data[key]['features']['train'][train_index]
            y_train = data[key]['targets']['train'][train_index]
            X_test = data[key]['features']['train'][test_index]
            y_test = data[key]['targets']['train'][test_index]
            X_unlabeled = data[key]['features']['unlabeled']

            xx, yy = get_train_xy(X_train, X_unlabeled, X_test, y_train)
            features[key] = xx
            targets[key] = yy

        ds = datagen.create_ds_single_model(features, targets, batch_size=params['batch_size'], seed=2324, shuffle=True)
        
        # =========================== model training
        n_features = {'cite': 512, 'multi': 512}
        n_targets = {'cite': 140, 'multi': 512}
        latent_dim = 64
        units = [256, 128]

        tf.keras.backend.clear_session()
        model = CellModel(n_features, n_targets, latent_dim, units, data_size)
               
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), 
                      loss={key: losses.loss_single_model for key in ['cite', 'multi']})
        model.fit(ds, epochs=params['epochs'])
        model.save_weights(os.path.join(folder, f'fold_{k}', 'weights'))
        #model.load_weights(os.path.join(folder, f'fold_{k}', 'weights'))
        
        # =========================== prediction        
        for key in ['cite', 'multi']:
            test_index = test_indexes[key]
            X_test = data[key]['features']['train'][test_index]            
            y_true = data[key]['targets']['true'][test_index]
            pipe_y = data[key]['targets']['pipe']
            inputs_test = {key: X_test for key in ['cite', 'multi']}

            n_samples = 300
            y_pred = None
            for _ in range(n_samples):
                y_sample = model.predict(inputs_test, batch_size=1024)[key]
                if y_pred is None:
                    y_pred = y_sample
                else:
                    y_pred += y_sample
            y_pred /= n_samples           

            y_pred = pipe_y.inverse_transform(y_pred)    

            assert y_true.shape == y_pred.shape
            score = correlation_score(y_true, y_pred)
            scores[key].append(score)  

            print(f'Fold{k} {key.upper()}:', score)
        
        break
    
    for key in ['cite', 'multi']:
        print(f'CV {key.upper()}:', np.mean(scores[key]))
    


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/train_single_model.py
    """
    cv()
    #train()


"""
v0:
Fold0 CITE: 0.8913680532445706
Fold0 MULTI: 0.6642353573187195

same embed for input and prior: 
Fold0 CITE: 0.8898267112046303  loss: 2282.7856 - cite_loss: 160.8306 - multi_loss: 710.8846
Fold0 MULTI: 0.6667190846528076

862/1862
separate models with embedding (common embedding for cite and multi is slightly worse)
Fold0 CITE: 0.8991286993312776  loss: 2154.3472 - cite_loss: 154.8472 - multi_loss: 696.8810
Fold0 MULTI: 0.6720333236375496
"""