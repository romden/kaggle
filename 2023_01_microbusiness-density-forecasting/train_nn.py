import sys, os, json, dill
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
print(tf.__version__)
tf.get_logger().setLevel('ERROR')

import modeler
from config import PATH_INPUT, PATH_WORKING, SEED

tf.random.set_seed(SEED*2)
np.random.seed(SEED*3)

INPUT_DIR = f'{PATH_WORKING}/n_targets_7_lags_census/data'

# model directory with results
OUTPUT_DIR = f'{PATH_WORKING}/n_targets_7_lags_census/linear_smape'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def setup_autoshard(ds):
    """Disable AutoShard.
    
    https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)
    return ds


class CFG:
    n_folds = 10
    batch_size = 64
    epochs = 100
    learning_rate = 1e-4
    n_units = 0
    rate = 0.5
    l1 = 0
    l2 = 1e-3


def train_folds():    

    # load data
    X = np.load(os.path.join(INPUT_DIR, 'x_train.npy'))
    y = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))    
    print('X.shape = ', X.shape, 'y.shape = ', y.shape)

    n_inputs = X.shape[1]
    n_outputs = y.shape[1]//2

    scores = []
    kfold = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=SEED*4)
    for k, (train_index, test_index) in enumerate(kfold.split(X)):          

        X_train = X[train_index]
        y_train = y[train_index]

        X_test = X[test_index]
        y_test = y[test_index]

        tf.keras.backend.clear_session()

        ds_train = setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_train, y_train) ) ).batch(CFG.batch_size)
        ds_valid = setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(1024*4)

        #strategy = tf.distribute.MirroredStrategy()
        #with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate)
        loss = modeler.smape_masked #mean_squared_error_masked # tf.keras.losses.MeanSquaredError()
        model = modeler.build_ffn(n_inputs, n_outputs, n_units=CFG.n_units, rate=CFG.rate, l1=CFG.l1, l2=CFG.l2)

        model.compile(optimizer=optimizer, loss=loss)  
        
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]

        history = model.fit(ds_train, validation_data=ds_valid, epochs=CFG.epochs, callbacks=callbacks).history
        
        tf.keras.models.save_model(model, os.path.join(OUTPUT_DIR, f'fold_{k}', 'model'))

        y_pred = model.predict(ds_valid)
        score = model.loss(y_test.astype('float32'), y_pred).numpy().item()
        #score = mean_squared_error(y_test, y_pred)        
        #score = min(history['val_loss'])
        
        scores.append(score)
        history['score'] = score     

        if 'lr' in history:
            del history['lr']

        with open(os.path.join(OUTPUT_DIR, f'fold_{k}', 'history.json'), 'w') as f:
            f.write(json.dumps(history, indent=4))
        
        print('Fold:', k, 'Score:', score)
        #break
    
    print('CV score:', np.mean(scores))


def predict_test():   

    x_test = np.load(os.path.join(INPUT_DIR, 'x_test.npy'))
    dstest = setup_autoshard( tf.data.Dataset.from_tensor_slices( x_test ) ).batch(1024*4)
    y_test = None
    n_folds = 0

    for k in range(CFG.n_folds):

        tf.keras.backend.clear_session()
        #with tf.keras.utils.custom_object_scope({'mean_squared_error_masked': modeler.mean_squared_error_masked}):
        with tf.keras.utils.custom_object_scope({'smape_masked': modeler.smape_masked}):
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/fold_{k}/model')
          
        y_fold = model.predict(dstest)

        if any(np.isnan(y_fold).ravel()):
            print(f'skipped fold {k}')
            continue        

        y_test = y_fold if y_test is None else y_test + y_fold
        n_folds += 1
    
    y_test = y_test.T.ravel() / n_folds
    
    submission = pd.read_csv(f'{PATH_INPUT}/sample_submission.csv')
    submission['microbusiness_density'] = y_test.tolist() + [0]*(len(submission)-len(y_test)) # add non-scored for june
    submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)


def train_oneshot(is_train=True, is_submission=True):  

    class CFG:
        batch_size = 96
        epochs = 100
        learning_rate = 1e-4
        n_units = 0
        rate = 0
        l1 = 0
        l2 = 1e-3  

    INPUT_DIR = f'{PATH_WORKING}/n_targets_7/data'
    OUTPUT_DIR = f'{PATH_WORKING}/n_targets_7/linear_smape_oneshot'
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if is_train:
        # load data
        X = np.load(os.path.join(INPUT_DIR, 'x_train.npy'))
        y = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))

        n_inputs = X.shape[1]
        n_outputs = y.shape[1]//2        

        ds_train = setup_autoshard( tf.data.Dataset.from_tensor_slices( (X, y) ) ).batch(CFG.batch_size)

        tf.keras.backend.clear_session()
        #strategy = tf.distribute.MirroredStrategy()
        #with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.learning_rate)
        loss = modeler.smape_masked #mean_squared_error_masked # tf.keras.losses.MeanSquaredError()
        model = modeler.build_ffn(n_inputs, n_outputs, n_units=CFG.n_units, rate=CFG.rate, l1=CFG.l1, l2=CFG.l2)

        model.compile(optimizer=optimizer, loss=loss)
        model.fit(ds_train, epochs=CFG.epochs)        
        tf.keras.models.save_model(model, os.path.join(OUTPUT_DIR, 'model'))


    if is_submission:
        x_test = np.load(os.path.join(INPUT_DIR, 'x_test.npy'))
        dstest = setup_autoshard( tf.data.Dataset.from_tensor_slices( x_test ) ).batch(1024*4)
        y_pred = model.predict(dstest)
        y_test = y_pred.T.ravel()
        
        submission = pd.read_csv(f'{PATH_INPUT}/sample_submission.csv')
        submission['microbusiness_density'] = y_test.tolist() + [0]*(len(submission)-len(y_test)) # add non-scored for june
        submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other-dev/kaggle-03-microbusiness-density-forecasting/train_nn.py
    """
    train_folds()
    predict_test()
    #train_oneshot()