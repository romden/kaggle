import sys, os, json, dill
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from utils import correlation_score
import dataload, datagen
from datagen import setup_autoshard
from config import *
import bayesian_builder as builder
import losses

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

# data directory
DATA_DIR = os.path.join(PATH_WORKING, 'data_cite_x512')
pipe_y = dill.load(open(os.path.join(DATA_DIR, 'pipe_y.dill'), 'rb'))

# model directory with results
MODEL_DIR = os.path.join(PATH_WORKING, 'pnn_cite_x512')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class CFG:
    n_samples = 300
    n_folds = 10
    batch_size = 64
    epochs = 50
    n_units = 256
    learning_rate = 1e-4


def scheduler(epoch):
    lr0 = CFG.learning_rate
    lrs = [lr0]*5 + [lr0/2]*5 + [lr0/10]*10 + [lr0/20]*5 + [lr0/100]*5
    return lrs[epoch]


# nll: 683.2532958984375 (lr=1e-4, only X_train)
#    : 656.0444946289062 (lr=1e-3, ...)
def train_vae():    

    folder = os.path.join(MODEL_DIR, 'vae')
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    ds_train = datagen.create_ds(X, X, CFG.batch_size)
    
    model, encoder, decoder = builder.build_vae(n_features=X.shape[1], latent_dim=64, units=[256, 128])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=losses.nll)
    model.summary()
        
    history = model.fit(ds_train, epochs=50).history   

    model.save_weights(os.path.join(folder, 'weights'))

    with open(os.path.join(folder, 'history.json'), 'w') as f:
        f.write(json.dumps(history, indent=4))   


# nll: 172.34451293945312
def train_ffn():

    folder = os.path.join(MODEL_DIR, 'ffn')
    if not os.path.exists(folder):
        os.mkdir(folder)

    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))   

    ds_train = datagen.create_ds(X_train, y_train, 64)

    model = builder.build_ffn(n_features=X_train.shape[1], n_targets=y_train.shape[1], n_units=256, data_size=X_train.shape[0])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses.nll)
    model.summary()

    history = model.fit(ds_train, epochs=50).history

    model.save_weights(os.path.join(folder, 'weights'))

    with open(os.path.join(folder, 'history.json'), 'w') as f:
        f.write(json.dumps(history, indent=4)) 

    # predict test
    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    ds_test = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( X_test ) ).batch(1024)
    
    n_samples = 300
    y_pred = np.zeros(SHAPES['cite'])    
    for _ in range(n_samples):
        y_pred += model.predict(ds_test)    
    y_pred /= n_samples

    y_pred = pipe_y.inverse_transform(y_pred)

    np.save(os.path.join(folder, 'y_pred.npy'), y_pred)


def train_folds():    

    folder = os.path.join(MODEL_DIR, 'folds')
    if not os.path.exists(folder):
        os.mkdir(folder)

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')    
    
    n_inputs = X.shape[1]
    n_outputs = y.shape[1]

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):     

        tf.keras.backend.clear_session()  

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        ds_train = datagen.create_ds(X_train, y_train, CFG.batch_size)
        ds_valid = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(1024)
        
        model = builder.build_ffn(n_inputs, n_outputs, n_units=CFG.n_units, data_size=X_train.shape[0])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate), loss=losses.nll)
        #model.summary()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(folder, f'fold_{k}', 'weights'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=1),
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=False),
        ]

        history = model.fit(ds_train, validation_data=ds_valid, epochs=CFG.epochs, callbacks=callbacks).history
        
        # load weights
        model.load_weights( os.path.join(folder, f'fold_{k}', 'weights') )

        # compute predictions
        y_pred = np.zeros(y_test.shape)
        for _ in range(CFG.n_samples):
            y_pred += model.predict(ds_valid)
        y_pred /= CFG.n_samples

        np.save(os.path.join(folder, f'fold_{k}', 'y_fold.npy'), y_pred)    
        
        # transform predictions
        y_pred = pipe_y.inverse_transform(y_pred)    

        # compute score
        score = correlation_score(y_true[test_index], y_pred)
        scores.append(score)
        history['score'] = score     

        if 'lr' in history:
            del history['lr']

        with open(os.path.join(folder, f'fold_{k}', 'history.json'), 'w') as f:
            f.write(json.dumps(history, indent=4))
        
        print('Fold:', k, 'Score:', score)  
    
    print('CV score:', np.mean(scores))


def predict_test():   

    folder = os.path.join(MODEL_DIR, 'folds')

    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    ds_test = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( X_test ) ).batch(1024)
    y_pred = np.zeros(SHAPES['cite'])

    n_inputs = X_test.shape[1]
    n_outputs = SHAPES['cite'][1]
    n_units = CFG.n_units
    data_size = SHAPES['cite'][0]

    for k in range(CFG.n_folds):

        tf.keras.backend.clear_session()
        model = builder.build_ffn(n_inputs, n_outputs, n_units, data_size)
        model.load_weights( os.path.join(folder, f'fold_{k}', 'weights') )
          
        #y_fold = model.predict(dstest)
        for _ in range(CFG.n_samples):
            y_pred += model.predict(ds_test)
    
    y_pred /= (CFG.n_folds * CFG.n_samples)

    y_pred = pipe_y.inverse_transform(y_pred)

    np.save(os.path.join(folder, 'y_pred.npy'), y_pred)


class CrossValidationWrapper:

    def __init__(self, params, X, y, y_true, pipe_y):
        self.params = params
        self.X = X
        self.y = y
        self.y_true = y_true
        self.pipe_y = pipe_y

    def build_model(self, shapes, use_mirrored=False):

        params = self.params
        data_size, n_inputs = shapes['X_train']
        _, n_outputs = shapes['y_train']

        tf.keras.backend.clear_session()
        loss = losses.nll
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])        
        model = builder.build_ffn(n_inputs, n_outputs, params['n_units'], data_size)
        model.compile(optimizer=optimizer, loss=loss)
        #model.summary() 
        return model

    def compute_score(self, model, X_test, y_test):
        # prediction
        n_samples = 100
        y_pred = np.zeros(y_test.shape)
        for _ in range(n_samples):
            outputs = model.predict(X_test)
            y_pred += outputs[1] if isinstance(outputs, list) else outputs
        y_pred /= n_samples       
        # transform predictions
        y_pred = self.pipe_y.inverse_transform(y_pred)
        # compute score
        score = correlation_score(y_test, y_pred)
        return score
    
    def find_best(self, name, values):
        scores = []  
        for val in values:
            self.params[name] = val
            score = do_cv(self)
            scores.append(score)
            print(f'{name}: {val} score: {score}')
        idx = np.argmax(scores)
        self.params[name] = values[idx]
        print(f'best {name}: {values[idx]}')


def model_selection():

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')

    params = {'learning_rate': 1e-4, 'batch_size': 64, 'n_units': 256, 'epochs': 50, 'n_splits': 7} # default

    cvwrapper = CrossValidationWrapper(params, X, y, y_true, pipe_y)

    #cvwrapper.find_best(name='batch_size', values=[1, 2, 4, 8, 16]) # find best batch_size [32, 48, 64, 96, 128]

    #cvwrapper.find_best(name='learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5]) # find best learning_rate

    #cvwrapper.find_best(name='epochs', values=[30, 50, 70])
    
    #with open(os.path.join(MODEL_DIR, 'cv' 'params.json'), 'w') as f: f.write(json.dumps(params, indent=4))
    

    from cross_validation import do_cv
    
    score = do_cv(cvwrapper, monitor_valid=False)
    print('CV score:', score)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/train_pnn_cite.py
    """
    #train_folds()
    #predict_test()
    #temp()
    #model_selection()
    
    #train_vae()
    #train_ffn()
    train_ffn_vae()


"""
# CV score: 0.8946875556015794
# CV score: 0.8915674738693641 (lr0=1e-4 with schedule)
epochs: 50 score: 0.8960560730555943 # better without BatchNormalization
"""
