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
import dataload, datagen
from datagen import setup_autoshard
from config import *


def do_cv(cvwrapper, monitor_valid=True):

    params = cvwrapper.params
    X = cvwrapper.X
    y = cvwrapper.y
    y_true = cvwrapper.y_true

    scores = []

    for k, (train_index, test_index) in enumerate(KFold(n_splits=params['n_splits'], shuffle=True, random_state=SEED*4).split(X)):
        
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]      

        shapes = {
            'X_train': X_train.shape,
            'y_train': y_train.shape,
            'X_test': X_test.shape,
            'y_test': y_test.shape
        }

        ds_train = datagen.create_ds(X_train, y_train, params['batch_size'])
        ds_valid = datagen.setup_autoshard( tf.data.Dataset.from_tensor_slices( (X_test, y_test) ) ).batch(1024)

        model = cvwrapper.build_model(shapes=shapes)

        if monitor_valid:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(PATH_WORKING, 'cv', 'weights'),
                    save_best_only=True,
                    save_weights_only=True,
                    monitor='val_loss',
                    verbose=0),
                #tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=False),
            ]
            # fit model
            model.fit(ds_train, validation_data=ds_valid, epochs=params['epochs'], callbacks=callbacks, verbose=0)                
            # load best weights on valid dataset
            model.load_weights( callbacks[0].filepath )

        else:
            model.fit(ds_train, epochs=params['epochs'], verbose=0)

        # compute score
        score = cvwrapper.compute_score(model, ds_valid, y_true[test_index])
        scores.append(score)
        print('Fold:', k, 'Score:', score)

    return np.mean(scores)