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
import builder, dataload, datagen
from datagen import setup_autoshard
from config import *

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

# data directory
DATA_DIR = os.path.join(PATH_WORKING, 'data_cite_x512')
#DATA_DIR = os.path.join(PATH_WORKING, 'data_cite')
pipe_y = dill.load(open(os.path.join(DATA_DIR, 'pipe_y.dill'), 'rb'))

# model directory with results
MODEL_DIR = os.path.join(PATH_WORKING, 'nn_cite_x512')
#MODEL_DIR = os.path.join(PATH_WORKING, 'nn_cite')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class CFG:
    n_folds = 10
    batch_size = 64
    epochs = 30
    learning_rate = 1e-4
    n_units = 256
    rate = 0.3
    l2 = 1e-3


def scheduler(epoch):
    lrs = [1e-4]*10 + [9e-5]*5 + [6e-5]*5 + [3e-5]*5 + [1e-5]*5 + [5e-6]*10 + [1e-6]*10
    return lrs[epoch]

def scheduler_selfsv(epoch):
    #lrs = [1e-3]*10 + [5e-4]*10 + [1e-4]*10 + [5e-5]*10 + [1e-5]*10
    lrs = [5e-4]*5 + [1e-4]*15 + [5e-5]*5 + [1e-5]*5
    return lrs[epoch]

def cosine_decay(initial_learning_rate, decay_steps, alpha):
    import math, itertools
    counter = itertools.count()
    def wrapper():
        step = next(counter)
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return initial_learning_rate * decayed
    return wrapper

def get_callbacks(checkpoint_filepath=None, lr_scheduler=None):
    callbacks = []

    if checkpoint_filepath:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                            save_best_only=True,
                                                            monitor='val_loss',
                                                            verbose=1))
    if lr_scheduler:
        callbacks.append( tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=False) )
    
    return callbacks


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
        ds_valid = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(CFG.batch_size)

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate)
            loss = tf.keras.losses.MeanSquaredError()
            model = builder.build_cite_ffn(n_inputs, n_outputs, n_units=CFG.n_units, rate=CFG.rate, l2=CFG.l2)

        model.compile(optimizer=optimizer, loss=loss) #, run_eagerly=True
        #model.summary()
        
        callbacks = get_callbacks(os.path.join(MODEL_DIR, f'fold_{k}', 'model'))

        history = model.fit(ds_train, validation_data=ds_valid, epochs=CFG.epochs, callbacks=callbacks).history
        
        model = tf.keras.models.load_model(callbacks[0].filepath)
        y_pred = model.predict(ds_valid)
        y_pred = pipe_y.inverse_transform(y_pred)    

        score = correlation_score(y_true[test_index], y_pred)
        scores.append(score)
        history['score'] = score     

        if 'lr' in history:
            del history['lr']

        with open(os.path.join(MODEL_DIR, f'fold_{k}', 'history.json'), 'w') as f:
            f.write(json.dumps(history, indent=4))
        
        print('Fold:', k, 'Score:', score)
        break
    
    print('CV score:', np.mean(scores))


def predict_test():   

    X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    dstest = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( X_test ) ).batch(CFG.batch_size)
    y_pred = None

    for k in range(CFG.n_folds):

        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model( os.path.join(MODEL_DIR, f'fold_{k}', 'model') )
          
        y_fold = model.predict(dstest)

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
        
        tf.keras.backend.clear_session()
        model = tf.keras.models.load_model( os.path.join(MODEL_DIR, f'fold_{k}', 'model') )
          
        y_fold = model.predict(X_test)
        y_fold = pipe_y.inverse_transform(y_fold)

        assert y_fold.shape[1] == SHAPES['cite'][1]

        np.save(os.path.join(MODEL_DIR, f'y_fold_{k}.npy'), y_fold)


def train_semi_supervised():

    from losses import semi_supervised

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32')    
    X_unlabeled = np.load(os.path.join(DATA_DIR, 'X_test.npy'))

    n_inputs = X.shape[1]
    n_outputs = y.shape[1]

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):     

        tf.keras.backend.clear_session()  

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        # ============== PRETRAINING ==============
        X_pretrain = np.concatenate([X_train, X_test, X_unlabeled], axis=0)
        y_pretrain = np.concatenate([np.concatenate([y_train, np.ones((y_train.shape[0],1))], axis=1),         
                                     np.zeros( (X_test.shape[0], y_train.shape[1]+1) ),
                                     np.zeros( (X_unlabeled.shape[0], y_train.shape[1]+1) ),
                                     ], axis=0)

        # DATASETS
        # pretraining
        ds_pretrain = datagen.create_ds(X_pretrain, y_pretrain, CFG.batch_size)
        # training
        ds_train = datagen.create_ds(X_train, y_train, CFG.batch_size)
        ds_valid = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(1024)

        # create model
        model, premodel, encoder = builder.build_cite_selfsv(n_inputs, n_outputs, n_units=CFG.n_units, rate=CFG.rate, l2=CFG.l2)
        model.summary()    

        folder = os.path.join(MODEL_DIR, 'pretrain', f'fold_{k}')
        if not os.path.exists(folder):
            os.mkdir(folder)
        
        #callbacks = get_callbacks(os.path.join(folder, 'model'))

        # ===========================
        # second round steps: 1870, 999
        # ===========================
        
        for lr1, lr2 in zip([1e-4], [2e-5]):
            #learning_rate_fn = cosine_decay(initial_learning_rate=lr1, decay_steps=30*1870, alpha=0.6)
            #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
            #optimizer = tf.keras.optimizers.Adam(learning_rate=lr1)
            #premodel.compile(optimizer=optimizer, loss=semi_supervised)
            #premodel.fit(ds_pretrain, epochs=20)
            #premodel.save(os.path.join(folder, 'premodel'))
            premodel.load_weights( os.path.join(folder, 'premodel', 'variables', 'variables' ) )

            learning_rate_fn = cosine_decay(initial_learning_rate=lr2, decay_steps=20*999, alpha=0.1)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
            #optimizer = tf.keras.optimizers.Adam(learning_rate=lr2)
            model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
            #history = model.fit(ds_train, epochs=20).history
        # ===========================
        '''
        # ===========================
        premodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=semi_supervised)
        premodel.fit(ds_pretrain, epochs=25)

        X_tmp = encoder.predict(X_train, batch_size=1024)   
        ds_train = datagen.create_ds(X_tmp, y_train, CFG.batch_size)
        model = builder.build_cite_ffn(X_tmp.shape[1], n_outputs, n_units=CFG.n_units, rate=CFG.rate, l2=CFG.l2)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError())
        history = model.fit(ds_train, epochs=25).history

        y_pred = tf.keras.Sequential([encoder, model]).predict(X_test, batch_size=1024)
        # ===========================
        '''        
        y_pred = model.predict(ds_valid)
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

    
    class CrossValidationWrapper:

        def __init__(self, params, X, y, y_true, pipe_y):
            self.params = params
            self.X = X
            self.y = y
            self.y_true = y_true
            self.pipe_y = pipe_y

        def build_model(self, shapes, use_mirrored=False):

            params = self.params
            _, n_inputs = shapes['X_train']
            _, n_outputs = shapes['y_train']

            tf.keras.backend.clear_session()
            if use_mirrored:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():            
                    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                    loss = tf.keras.losses.MeanSquaredError()
                    model = builder.build_cite_ffn(n_inputs, n_outputs, params['n_units'], params['rate'], params['l2'])
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
                loss = tf.keras.losses.MeanSquaredError()
                model = builder.build_cite_ffn(n_inputs, n_outputs, params['n_units'], params['rate'], params['l2']) 
            model.compile(optimizer=optimizer, loss=loss)
            model.summary() 
            return model

        def compute_score(self, model, X_test, y_true):
            # prediction
            y_pred = model.predict(X_test)        
            # transform predictions
            y_pred = self.pipe_y.inverse_transform(y_pred)
            # compute score
            score = correlation_score(y_true, y_pred)
            return score


    params = {'learning_rate': 1e-4, 'batch_size': 64, 'n_units': 256, 'epochs': 30, 'n_splits': 7, 'rate': 0.3, 'l2': 1e-3}
    cvwrapper = CrossValidationWrapper(params, X, y, y_true, pipe_y)

    from cross_validation import do_cv
    
    score = do_cv(cvwrapper)
    print('CV score:', score)

    #model = build_cite_ffn2(n_inputs, n_outputs, n_units, rate=0.3, l2=1e-3)

    #find_best(params, name='batch_size', values=[1, 2, 4, 8, 16]) # find best batch_size [32, 48, 64, 96, 128]
    #find_best(params, name='learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5]) # find best learning_rate
    #find_best(params, name='epochs', values=[30, 50, 70])
    
    #with open(os.path.join(MODEL_DIR, 'cv' 'params.json'), 'w') as f: f.write(json.dumps(params, indent=4))

    # CV score: 0.8965082538994621 build_cite_ffn2
    # CV score: 0.894862601366049  build_cite_ffn


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-02-single-cell-integration/train_nn_cite.py
    """
    #train_folds()
    #predict_test()
    #predict_folds()
    #train_semi_supervised()
    model_selection()



#========== NN
#CV score: 0.8952340168508777 gelu
#CV score: 0.8945514204180173 relu
#CV score: 0.8945089200372813


#========== LGBM
#CV score: 0.8961377694285847