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
import builder, dataload, datagen
from datagen import setup_autoshard
from config import *

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

# data directory
DATA_DIR = os.path.join(PATH_WORKING, 'data_cite_x512')
pipe_y = dill.load(open(os.path.join(DATA_DIR, 'pipe_y.dill'), 'rb'))

# model directory with results
MODEL_DIR = os.path.join(PATH_WORKING, 'nnlgb_cite_x512')
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


def scheduler1(epoch):
    lr0 = CFG.learning_rate
    lrs = [lr0*10]*10 + [lr0*5]*10 + [lr0*1]*10
    return lrs[epoch]


def scheduler2(epoch):
    lr0 = CFG.learning_rate
    lrs = [lr0/1]*5 + [lr0/10]*5 + [lr0/100]*5 + [lr0/1000]*5
    return lrs[epoch]


def pretrain():
    """Pretrain nn using lgb predictions"""     

    # load x test and y pred by lgb
    xlst = [np.load(os.path.join(DATA_DIR, 'X_test.npy'))]
    ylst = [np.load(os.path.join(PATH_WORKING, 'lgb_cite_x512', 'y_pred.npy'))]

    # load x train
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))

    # assign lgb pred for x train
    for k, (train_index, test_index) in enumerate(KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4).split(X)):
        xlst.append(X[test_index])
        y_fold = np.load(os.path.join(PATH_WORKING, 'lgb_cite_x512', f'y_fold_{k}.npy'))
        ylst.append(y_fold)
    
    # creat train data
    X_train = np.concatenate(xlst)
    y_train = np.concatenate(ylst)

    n_inputs = X_train.shape[1]
    n_outputs = y_train.shape[1]

    ds_train = datagen.create_ds(X_train, y_train, CFG.batch_size)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate)
        loss = tf.keras.losses.MeanSquaredError()
        model = builder.build_cite_ffn(n_inputs, n_outputs, n_units=CFG.n_units, rate=CFG.rate, l2=CFG.l2)
        #model.load_weights(os.path.join(MODEL_DIR, 'fold_0', 'model', 'variables', 'variables'))

    model.compile(optimizer=optimizer, loss=loss) #, run_eagerly=True
    #model.summary()
    
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler1, verbose=False),
    ]

    history = model.fit(ds_train, validation_data=None, epochs=CFG.epochs, callbacks=None).history

    folder = os.path.join(MODEL_DIR, 'pretrain')
    if not os.path.exists(folder):
        os.mkdir(folder)

    #model.save_weights(os.path.join(folder, 'weights'))
    model.save(folder)

    if 'lr' in history:
        del history['lr']

    with open(os.path.join(MODEL_DIR, 'pretrain', 'history.json'), 'w') as f:
        f.write(json.dumps(history, indent=4))



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
            model.load_weights(os.path.join(MODEL_DIR, 'pretrain', 'variables', 'variables'))

        model.compile(optimizer=optimizer, loss=loss) #, run_eagerly=True
        #model.summary()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, f'fold_{k}', 'model'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1),
            tf.keras.callbacks.LearningRateScheduler(scheduler2, verbose=False),
        ]        

        history = model.fit(ds_train, validation_data=ds_valid, epochs=CFG.epochs-10, callbacks=callbacks).history 
        
        model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'fold_{k}', 'model'))
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


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-02-single-cell-integration/train_nnlgb_cite.py
    """
    #pretrain()
    #train_folds()
    predict_test()