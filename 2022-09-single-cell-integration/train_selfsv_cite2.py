import sys, os, json, dill
import numpy as np
import pandas as pd

#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
print(tf.__version__)
tf.get_logger().setLevel('ERROR')

from utils import correlation_score
import dataload, datagen
from datagen import setup_autoshard
import losses
import bayesian_builder as builder
from config import *

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

# data directory
PATH_X = os.path.join(PATH_WORKING, 'data_cite_x256')
PATH_Y = os.path.join(PATH_WORKING, 'data_cite_y140')

# model directory with results
PATH_MODEL = os.path.join(PATH_WORKING, 'selfsv_cite_x512_y140')
if not os.path.exists(PATH_MODEL):
    os.mkdir(PATH_MODEL)

pipe_y = dill.load(open(os.path.join(PATH_Y, 'pipe_y.dill'), 'rb'))

def load_data():
    # load data
    X_train = np.load(os.path.join(PATH_X, 'X_train.npy'))
    y_train = np.load(os.path.join(PATH_Y, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')    
    X_test = np.load(os.path.join(PATH_X, 'X_test.npy'))
    X_unlabeled = np.load(os.path.join(PATH_X, 'X_unlabeled.npy'))
    return X_train, y_train, y_true, X_test, X_unlabeled


def load_data_decoder():
    # load data
    X_train = np.load(os.path.join(PATH_WORKING, 'data_cite_x128', 'X_train.npy'))
    X_test = np.load(os.path.join(PATH_WORKING, 'data_cite_x128', 'X_test.npy'))
    X_unlabeled = np.load(os.path.join(PATH_WORKING, 'data_cite_x128', 'X_unlabeled.npy'))
    return X_train, X_test, X_unlabeled


def cv_ffn_vae():    

    folder = os.path.join(PATH_MODEL, 'cv_ffn_vae')
    if not os.path.exists(folder):
        os.mkdir(folder)    

    # load data
    X, y, y_true, _, X_unlabeled = load_data()

    params = {
        'n_folds': 5,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'epochs': 60
    }

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=params['n_folds'], shuffle=True, random_state=SEED*4).split(X)):          

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        xx = np.concatenate([X_train, X_unlabeled, X_test], axis=0)
        y1 = np.concatenate([X_train, y_train, np.ones((y_train.shape[0],1))], axis=1)        
        y2 = np.concatenate([X_unlabeled, np.zeros((X_unlabeled.shape[0], y_train.shape[1]+1))], axis=1)
        y3 = np.concatenate([X_test, np.zeros((X_test.shape[0], y_train.shape[1]+1))], axis=1)
        yy = np.concatenate([y1, y2, y3], axis=0)

        ds = datagen.create_ds(xx, yy, params['batch_size'])
        ds_pred = datagen.create_ds(xx[:len(X_train)], yy[:len(X_train)], params['batch_size'])

        n_features = X_train.shape[1]
        n_targets = y_train.shape[1]
        latent_dim = 64
        units = [256, 128]
        data_size = X_train.shape[0]

        model = builder.build_ffn_vae(n_features, n_targets, latent_dim, units, data_size)
        model.summary()

        tf.keras.backend.clear_session()
        loss = losses.LossReconstractionPrediction(n_features, n_targets)
        if True: # one step            
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), 
                          loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},)
            model.fit(ds, epochs=params['epochs'])

        else:
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate']*10), 
                          loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},
                          loss_weights = {'reconstruction': 1., 'prediction': 1e-1})
            model.fit(ds, epochs=params['epochs'])
            #model.save_weights(os.path.join(folder, 'step1', 'weights')) # save_weights load_weights

            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate']), 
                          loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},
                          loss_weights = {'reconstruction': 1e-1, 'prediction': 1.})
            model.fit(ds_pred, epochs=params['epochs'])
            #model.save_weights(os.path.join(folder, 'step2', 'weights'))

        n_samples = 300
        y_pred = np.zeros( (X_test.shape[0], n_targets) )   
        for _ in range(n_samples):
            y_pred += model.predict(X_test, batch_size=1024)['prediction']
        y_pred /= n_samples
        
        y_pred = pipe_y.inverse_transform(y_pred)    

        score = correlation_score(y_true[test_index], y_pred)
        scores.append(score)
        '''history['score'] = score     
        
        if 'lr' in history:
            del history['lr']

        with open(os.path.join(folder, 'history.json'), 'w') as f:
            f.write(json.dumps(history, indent=4))'''
        
        print('Fold:', k, 'Score:', score)
        break
    
    print('CV score:', np.mean(scores))

#Fold: 0 Score: 0.8990928373457976 (after updating X_unlabeled) predictor_loss: 156.0225 - decoder_loss: 657.6879

#CV score: 0.8988370725812551 # after fixing data (batch_size=64) predictor_loss: 155.6947 - decoder_loss: 655.8051
#CV score: 0.8988906391759321 # epochs=100 for n_splits=5

"""MVN
Fold: 0 Score: 0.8834469444358429 predictor_loss: 148.0366 - decoder_loss: 656.1914
"""

def train_ffn_vae():

    folder = os.path.join(PATH_MODEL, 'train_ffn_vae')
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load data
    X_train, y_train, _, X_test, X_unlabeled = load_data()

    X = np.concatenate([X_train, X_unlabeled], axis=0)
    y1 = np.concatenate([X_train, y_train, np.ones((y_train.shape[0],1))], axis=1)
    y2 = np.concatenate([X_unlabeled, np.zeros((X_unlabeled.shape[0], y_train.shape[1]+1))], axis=1)
    y = np.concatenate([y1, y2], axis=0)

    ds = datagen.create_ds(X, y, 64)

    n_features = X_train.shape[1]
    n_targets = y_train.shape[1]
    latent_dim = 64
    units = [256, 128]
    data_size = X_train.shape[0]

    model = builder.build_ffn_vae(n_features, n_targets, latent_dim, units, data_size)
    model.summary()
    if False:
        # load existing weights
        model.load_weights(os.path.join(folder, 'weights'))
    else:
        # train and get new weights
        loss = losses.LossReconstractionPrediction(n_features, n_targets)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},)        

        history = model.fit(ds, epochs=100).history
        model.save_weights(os.path.join(folder, 'weights'))

        with open(os.path.join(folder, 'history.json'), 'w') as f:
            f.write(json.dumps(history, indent=4))   

    # predict test
    n_samples = 300
    y_pred = np.zeros((X_test.shape[0], y_train.shape[1]))  
    for _ in range(n_samples):
        y_pred += model.predict(X_test, batch_size=1024)['prediction']
    y_pred /= n_samples
    
    y_pred = pipe_y.inverse_transform(y_pred)
    np.save(os.path.join(folder, 'y_pred.npy'), y_pred)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/train_selfsv_cite2.py
    """
    cv_ffn_vae()
    #train_ffn_vae()