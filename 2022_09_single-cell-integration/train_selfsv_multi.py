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

from config import *
from utils import correlation_score
import dataload, datagen
import losses
import bayesian_builder as builder

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

PATH_X = os.path.join(PATH_WORKING, 'data_multi_x512')
PATH_Y = os.path.join(PATH_WORKING, 'data_multi_y512')

# model directory with results
PATH_MODEL = os.path.join(PATH_WORKING, 'selfsv_multi_x512_y512')
if not os.path.exists(PATH_MODEL):
    os.mkdir(PATH_MODEL)
    
pipe_y = dill.load(open(os.path.join(PATH_Y, 'pipe_y.dill'), 'rb'))


def load_data():
    # load data
    X_train = np.load(os.path.join(PATH_X, 'X_train.npy'))
    y_train = np.load(os.path.join(PATH_Y, 'y_train.npy'))
    y_true = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values.astype('float32')    
    X_test = np.load(os.path.join(PATH_X, 'X_test.npy'))
    X_unlabeled = np.load(os.path.join(PATH_X, 'X_unlabeled.npy'))
    return X_train, y_train, y_true, X_test, X_unlabeled


class CFG:
    n_folds = 5
    batch_size = 64
    n_units = 256
    epochs = 30
    learning_rate = 1e-4    


def cv_ffn_vae():        

    folder = os.path.join(PATH_MODEL, 'cv_ffn_vae')
    if not os.path.exists(folder):
        os.mkdir(folder)    

    # load data
    #X, y, y_true, X_unlabeled = load_data()
    X, y, y_true, _, X_unlabeled = load_data()

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):          

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        xx = np.concatenate([X_train, X_unlabeled, X_test], axis=0)
        y1 = np.concatenate([X_train, y_train, np.ones((y_train.shape[0],1))], axis=1)        
        y2 = np.concatenate([X_unlabeled, np.zeros((X_unlabeled.shape[0], y_train.shape[1]+1))], axis=1)
        y3 = np.concatenate([X_test, np.zeros((X_test.shape[0], y_train.shape[1]+1))], axis=1)
        yy = np.concatenate([y1, y2, y3], axis=0)

        ds = datagen.create_ds(xx, yy, 64)
        #ds_pred = datagen.create_ds(xx[:len(X_train)], yy[:len(X_train)], 64)

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
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), 
                          loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},)
            model.fit(ds, epochs=80)
            model.save_weights(os.path.join(folder, 'weights', 'weights'))

        else:
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
                          loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},
                          loss_weights = {'reconstruction': 1., 'prediction': 0.1})
            model.fit(ds, epochs=50)
            #model.save_weights(os.path.join(folder, 'step1', 'weights')) # save_weights load_weights

            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), 
                          loss = {'reconstruction': loss.reconstruction, 'prediction': loss.prediction},
                          loss_weights = {'reconstruction': 0.1, 'prediction': 1.})
            model.fit(ds, epochs=50)
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
        #break
    
    print('CV score:', np.mean(scores))

# CV score: 0.6716763243377935 (after fixing data, batch_size=64) predictor_loss: 697.0930 - decoder_loss: 616.0405


def train_ffn_vae():    

    folder = os.path.join(PATH_MODEL, 'train_ffn_vae')
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load data
    #X_train, y_train, y_true, X_test = load_data()
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
    python other_research/kaggle-02-single-cell-integration/train_selfsv_multi.py
    """
    #cv_ffn_vae()
    train_ffn_vae()