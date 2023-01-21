"""
Note: Was used with decision trees.
Used for model selection. Two approaches are implemented: grid serch and bayesian optimization.
Scripts may require some adaptations before being used.
"""
import sys, os, json, dill
import itertools
import numpy as np

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import builder, losses, dataload, utils, custom
from config import PATH_DATA, PATH_WORKING, SEED

tf.random.set_seed(SEED)
np.random.seed(SEED)

for pkg in ['/home/XXsssXX/aapf/libs/hyperopt-master']: # newer version, old version gives error
    if pkg not in sys.path:
        sys.path.append(pkg)
# needed for hyperopt
import imp
imp.load_package('py4j','/opt/anaconda/4.2.0/lib/python3.5/site-packages/py4j')   

from hyperopt import hp, fmin, tpe, space_eval

T, n_feat = dataload.get_modeldims()

# get data
dtrain, dvalid, dtest = dataload.load(train=True, valid=True, test=False)
data = dataload.merge(dtrain, dvalid)

from train import lrfn

# define an objective function
def objective(params, goal='min'):

    scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=100)
    for train_index, test_index in skf.split(data['y'], data['y']):

        dtrain = {key: data[key][train_index] for key in data.keys()}
        dvalid = {key: data[key][test_index] for key in data.keys()}

        ds_train, ds_valid = dataload.create_ds(dtrain=dtrain, dvalid=dvalid, batch_size=512, batch_size_valid=10000)

        tf.keras.backend.clear_session()
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            loss = tf.keras.losses.BinaryCrossentropy()
            metrics = ['binary_crossentropy']
            model, premodel = builder.build_model(T=T, n_feat=n_feat, params=params)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        callbacks = [tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)]

        history = model.fit(ds_train, validation_data=ds_valid, batch_size=1, epochs=33, verbose=0, callbacks=callbacks) 

        name = 'val_binary_crossentropy'
        if goal == 'min':
            score = min(history.history[name])
        elif goal == 'max':
            score = -max(history.history[name])
        scores.append(score)

    # return objective value    
    return np.mean(scores)


def main_hyperopt():
    """Searches for best parameter setting for classifier using hyperopt library."""

    # define a search space
    space = hp.choice('params',
                [
                    {
                        'n_blocks': hp.choice('n_blocks', [1, 2, 3, 4]),
                        'n_heads': hp.choice('n_heads', [1, 2, 3, 4]),
                        'n_units': hp.choice('n_units', [16, 32, 64, 128]),
                        'n_ffn': hp.choice('n_ffn', [8, 16, 32, 64]),
                        'rate': hp.choice('rate', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]),
                        'l2': hp.choice('l2', [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3, 4.5e-3, 5e-3]),
                    }
                ])

    # minimize the objective over the space
    best = fmin(objective, space, algo=tpe.suggest, max_evals=20)

    params = space_eval(space, best) # extract parameters
    
    print('hyperopt search completed')
    print(params)

    with open(os.path.join(PATH_WORKING, 'results', 'hyperopt.txt'), 'w') as f:
        f.write(json.dumps(params, indent=4))


def main_exhaustive():

    keys = ['n_blocks', 'n_heads', 'n_units', 'n_ffn', 'rate', 'l2']

    parameters = {
        'n_blocks': [1, 2, 3, 4],
        'n_heads': [1, 2, 3, 4],
        'n_units': [16, 32, 64, 128],
        'n_ffn': [8, 16, 32, 64],
        'rate': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        'l2': [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3, 4.5e-3, 5e-3],
    }

    best_score = None
    best_params = None

    for values in itertools.product(*[parameters[key] for key in keys]):
        params = dict(zip(keys, values))
        score = objective(params)
        if best_score is None or score < best_score:
            best_score = score
            best_params = params
        print(params, score)
    
    print('exhaustive search completed')
    print(best_params)
    print(best_score)



if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-01-amex/model_selection.py
    """
    #main_exhaustive()
    main_hyperopt()