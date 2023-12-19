import sys, os, json, dill
import numpy as np
import pandas as pd

#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from utils import correlation_score
import builder, dataload, datagen
from config import *

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

# data directory
DATA_DIR = os.path.join(PATH_WORKING, 'data_multi_x1024')
pipe_y = dill.load(open(os.path.join(DATA_DIR, 'pipe_y.dill'), 'rb'))

# model directory with results
MODEL_DIR = os.path.join(PATH_WORKING, 'nn_multi_x1024')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


class CFG:
    n_folds = 5
    batch_size = 64
    epochs = 100
    learning_rate = 1e-4
    rate = 0.1
    l2 = 1e-3


def scheduler(epoch):
    lr0 = CFG.learning_rate
    lrs = [lr0]*50 + [lr0*0.5]*25 + [lr0*0.01]*25
    return lrs[epoch]


def train_folds():    

    # load data
    X = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    y = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_true = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values.astype('float32')    

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
            optimizer = tf.keras.optimizers.Adam(CFG.learning_rate)
            loss = tf.keras.losses.MeanSquaredError()
            model = builder.build_multi_ffn(n_inputs, n_outputs, rate=CFG.rate, l2=CFG.l2)

        model.compile(optimizer=optimizer, loss=loss) #, run_eagerly=True
        model.summary()
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, f'fold_{k}', 'model'),
                save_best_only=True,
                monitor='val_loss',
                verbose=1),
            tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=False)
        ]

        history = model.fit(ds_train, validation_data=ds_valid, epochs=CFG.epochs, callbacks=callbacks).history
        
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

    assert y_pred.shape == SHAPES['multi']

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

        assert y_fold.shape[1] == SHAPES['multi'][1]

        np.save(os.path.join(MODEL_DIR, f'y_fold_{k}.npy'), y_fold)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-02-single-cell-integration/train_nn_multi.py
    """
    #train_folds()
    #predict_test()
    predict_folds()


# Cross-validation score: 0.6541472783317437 (y_train_original)
# Cross-validation score: 0.6422927987197485 (data_multi_x1024_y512)
# Cross-validation score: 0.6422852297651096 (original scaled)
# Cross-validation score: 0.661 (with batch norm)
# Cross-validation score: 0.6653848057329819 (rate=0.1)

#==========after fix
# Cross-validation score: 0.6634326089548509 gelu
# Cross-validation score: 0.66531991515415 relu (0.806 with)
# Cross-validation score: 0.6662299781588444 relu (epochs=70)