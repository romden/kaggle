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

tf.random.set_seed(SEED)
np.random.seed(SEED)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# alternatively seting in commandline: export TF_XLA_FLAGS=--tf_xla_enable_xla_devices


def lrfn(epoch):
    lr0 = 1e-4
    delta = 0.1e-4
    lb = 0.2e-4
    step = 10
    lr = max(lr0 - (epoch//step) * delta, lb)    
    return lr 


def load_data(folder='data_multi_1024'):

    X_train = np.load(os.path.join(PATH_WORKING, folder, 'X_train.npy'))
    y_train = np.load(os.path.join(PATH_WORKING, folder, 'y_train.npy'))

    '''
    fname = os.path.join(PATH_WORKING, folder, 'transSVD_y.dill')
    if trans_y and os.path.exists(fname):
        transSVD_y = dill.load(open(fname, 'rb'))
        y_train = transSVD_y.transform(y_train)'''
    
    dtrain, dvalid = dataload.split_data(X_train, y_train, k=10000)

    return dtrain, dvalid


def train(batch_size=64, epochs=30):

    dtrain, dvalid = load_data(folder='data_multi_x1024_yo')

    n_samples, n_inputs = dtrain[0].shape
    n_outputs = dtrain[1].shape[1]

    ds_train = setup_autoshard( tf.data.Dataset.from_tensor_slices(dtrain) ).shuffle(buffer_size=n_samples, reshuffle_each_iteration=True).batch(batch_size)
    ds_valid = setup_autoshard( tf.data.Dataset.from_tensor_slices(dvalid) ).batch(batch_size)    


    #tf.compat.v1.enable_eager_execution()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = tf.keras.losses.MeanSquaredError()
        #metrics = [correlation_score] 
        model = builder.build_multi_3(n_inputs, n_outputs, n_units=256, rate=0.1, l2=1e-3)
        
    model.compile(optimizer=optimizer, loss=loss) #, metrics=metrics, run_eagerly=True
    model.summary()
    

    folder = os.path.join(PATH_WORKING, 'tmp')
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(folder, 'model'),
            save_best_only=True,
            #save_weights_only=True,
            #monitor='val_<lambda>',
            #mode='max',
            monitor='val_loss',
            verbose=1)
        ]
    history = model.fit(ds_train, validation_data=ds_valid, epochs=epochs, callbacks=callbacks)

    with open(os.path.join(folder, 'history.json'), 'w') as f:
        f.write(json.dumps(history.history, indent=4))


def evaluate(batch_size=128):

    _, dvalid = load_data(trans_y=False)
    
    #ds_valid = setup_autoshard( tf.data.Dataset.from_tensor_slices(dvalid) ).batch(batch_size)
    
    X_test, y_test = dvalid
    n_inputs = X_test.shape[1]
    n_outputs = y_test.shape[1]

    INPUT_DIR = 'data_multi_1024'

    #model = builder.build_ffn(n_inputs, n_outputs, n_units=256, rate=0.3, l2=1e-3)
    #model.load_weights(os.path.join(PATH_WORKING, 'res_ffn_cite512', 'weights'))

    model = builder.build_ffn(n_inputs, n_outputs, n_units=256*2, rate=0.3, l2=1e-3)
    model.load_weights(os.path.join(PATH_WORKING, INPUT_DIR, 'res_ffn', 'weights'))

    y_pred = model.predict(X_test, batch_size=batch_size)
    y_true = y_test

    fname = os.path.join(PATH_WORKING, INPUT_DIR, 'transSVD_y.dill')
    if os.path.exists(fname) and False:
        transSVD_y = dill.load(open(fname, 'rb'))
        y_pred = transSVD_y.inverse_transform(y_pred)

    score = correlation_score(y_true, y_pred)
    print('score', score)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-02-single-cell-integration/train.py
    """
    train()
    #evaluate()


# Total params: 398,337 (build_multi_2)