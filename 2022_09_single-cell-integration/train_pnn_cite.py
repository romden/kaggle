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
from bayesian_builder import build_cite_ffn

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
    epochs = 30
    n_units = 256
    learning_rate = 1e-4


def scheduler(epoch):
    lr0 = CFG.learning_rate
    lrs = [lr0]*5 + [lr0/2]*5 + [lr0/10]*10 + [lr0/20]*5 + [lr0/100]*5
    return lrs[epoch]

@tf.function
def nll(y_true, y_pred):
    #return -y_pred.log_prob(y_true) # used in tutorials, but do not understand why this way
    return -tf.math.reduce_mean(y_pred.log_prob(y_true))


def train_folds():    

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

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate)
            loss = nll
            model = build_cite_ffn(n_inputs, n_outputs, n_units=CFG.n_units, data_size=X_train.shape[0])

        model.compile(optimizer=optimizer, loss=loss) #, run_eagerly=True
        #model.summary()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, f'fold_{k}', 'weights'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=1),
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=False),
        ]

        history = model.fit(ds_train, validation_data=ds_valid, epochs=CFG.epochs, callbacks=callbacks).history
        
        # load weights
        model.load_weights( os.path.join(MODEL_DIR, f'fold_{k}', 'weights') )

        # compute predictions      
        #y_pred = sum([model.predict(ds_valid) for _ in range(n_samples)])/n_samples
        y_pred = np.zeros(y_test.shape)
        for _ in range(CFG.n_samples):
            y_pred += model.predict(ds_valid)
        y_pred /= CFG.n_samples    

        np.save(os.path.join(MODEL_DIR, f'y_fold_{k}.npy'), y_pred)    
        
        # transform predictions
        y_pred = pipe_y.inverse_transform(y_pred)    

        # compute score
        score = correlation_score(y_true[test_index], y_pred)
        scores.append(score)
        history['score'] = score     

        if 'lr' in history:
            del history['lr']

        with open(os.path.join(MODEL_DIR, f'fold_{k}', 'history.json'), 'w') as f:
            f.write(json.dumps(history, indent=4))
        
        print('Fold:', k, 'Score:', score)  
    
    print('CV score:', np.mean(scores))


def predict_test():   

    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    dstest = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( X_test ) ).batch(1024)
    y_pred = np.zeros(SHAPES['cite'])

    n_inputs = X_test.shape[1]
    n_outputs = SHAPES['cite'][1]
    n_units = CFG.n_units
    data_size = SHAPES['cite'][0]

    for k in range(CFG.n_folds):

        tf.keras.backend.clear_session()
        model = build_cite_ffn(n_inputs, n_outputs, n_units, data_size)
        model.load_weights( os.path.join(MODEL_DIR, f'fold_{k}', 'weights') )
          
        #y_fold = model.predict(dstest)
        for _ in range(CFG.n_samples):
            y_pred += model.predict(dstest)
    
    y_pred /= (CFG.n_folds * CFG.n_samples)

    y_pred = pipe_y.inverse_transform(y_pred)

    np.save(os.path.join(MODEL_DIR, 'y_pred.npy'), y_pred)


def temp():

    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')
    
    n_inputs = X.shape[1]
    n_outputs = SHAPES['cite'][1]
    n_units = CFG.n_units
    data_size = SHAPES['cite'][0]

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        break
    #print('data size', X_train.shape[0])

    model = build_cite_ffn(n_inputs, n_outputs, n_units=CFG.n_units, data_size=X_train.shape[0])
    model.load_weights( os.path.join(MODEL_DIR, f'fold_{k}', 'weights') )

    #ds_valid = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(CFG.batch_size)

    n_samples = 300
    #y_pred = sum([model.predict(X_test, batch_size=1024) for _ in range(n_samples)])/n_samples
    y_pred = np.zeros(y_test.shape)
    for _ in range(n_samples):
        y_pred += model.predict(X_test, batch_size=1024)
    y_pred /= n_samples  

    ''' Used to save on machine
    for i in range(n_samples):
        y_pred = model.predict(X_test)
        np.save(os.path.join(MODEL_DIR, f'fold_{k}', 'samples', f'y_pred_{i}.npy'), y_pred)

    np.save(os.path.join(MODEL_DIR, f'fold_{k}', 'samples', 'y_true.npy'), y_test)'''

    y_pred = pipe_y.inverse_transform(y_pred)    

    score = correlation_score(y_true[test_index], y_pred)
    print('correlation_score', score)


def do_cv(params, X, y, y_true):

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=params['n_splits'], shuffle=True, random_state=SEED*4).split(X)):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        n_inputs = X_train.shape[1]
        n_outputs = y_train.shape[1]
        data_size = X_train.shape[0]

        tf.keras.backend.clear_session()

        ds_train = datagen.create_ds(X_train, y_train, params['batch_size'])
        ds_valid = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(1024)

        if False:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                loss = nll
                model = build_cite_ffn(n_inputs, n_outputs, params['n_units'], data_size)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
            loss = nll
            model = build_cite_ffn(n_inputs, n_outputs, params['n_units'], data_size)
        
        model.compile(optimizer=optimizer, loss=loss)

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, 'cv', 'weights'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=0),
            #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=False),
        ]

        #model.fit(ds_train, validation_data=ds_valid, epochs=params['epochs'], callbacks=callbacks, verbose=0)
        model.fit(ds_train, epochs=params['epochs'], verbose=0)
            
        # load weights
        #model.load_weights( callbacks[0].filepath )

        # compute predictions
        n_samples = 100
        y_pred = np.zeros(y_test.shape)
        for _ in range(n_samples):
            y_pred += model.predict(ds_valid)
        y_pred /= n_samples 
        
        # transform predictions
        y_pred = pipe_y.inverse_transform(y_pred)

        # compute score
        score = correlation_score(y_true[test_index], y_pred)
        scores.append(score)

    return np.mean(scores)


def model_selection():

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')

    def find_best(params, name, values):
        scores = []  
        for val in values:
            params[name] = val
            score = do_cv(params, X, y, y_true)
            scores.append(score)
            print(f'{name}: {val} score: {score}')
        idx = np.argmax(scores)
        params[name] = values[idx]
        print(f'best {name}: {values[idx]}')

    params = {'learning_rate': 1e-4, 'batch_size': 64, 'n_units': 256, 'epochs': 30, 'n_splits': 7} # default

    #find_best(params, name='batch_size', values=[12, 16, 24, 32]) # find best batch_size [32, 48, 64, 96, 128]

    #find_best(params, name='learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5]) # find best learning_rate

    find_best(params, name='epochs', values=[60, 80, 100])
    
    with open(os.path.join(MODEL_DIR, 'cv' 'params.json'), 'w') as f:
        f.write(json.dumps(params, indent=4))


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/train_pnn_cite.py
    """
    #train_folds()
    #predict_test()
    #temp()
    model_selection()


# CV score: 0.8946875556015794
# CV score: 0.8915674738693641 (lr0=1e-4 with schedule)

"""
batch_size: 12 score: 0.8943405645188705
batch_size: 16 score: 0.8940570419774833
batch_size: 24 score: 0.894002433825265
batch_size: 32 score: 0.8941104052164034
batch_size: 48 score: 0.8936315373431354
batch_size: 64 score: 0.8932669104011819
batch_size: 96 score: 0.8931030168020545
batch_size: 128 score: 0.8919038924868519

learning_rate: 0.001 score: 0.8912772144896091
learning_rate: 0.0005 score: 0.8938882965644993
learning_rate: 0.0001 score: 0.8941104052164034
learning_rate: 5e-05 score: 0.8923009199697798
learning_rate: 1e-05 score: 0.8130883676444537

epochs: 30 score: 0.892321214215191
epochs: 35 score: 0.8931637733288946
epochs: 40 score: 0.8936536589233552
epochs: 45 score: 0.893965150424132
epochs: 50 score: 0.8940108715305178
"""
