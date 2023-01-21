import numpy as np
import pandas as pd

import os
import dill

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import builder, dataload
from custom import compute_predictions_gpu, compute_predictions_cpu
from config import PATH_DATA, PATH_WORKING, PATH_DATAREADY

T, n_x = dataload.get_modeldims()


def extract_features():
    """Feature extraction using trained model."""

    folder = os.path.join(PATH_WORKING, 'results', 'model')

    dtrain, dvalid, _ = dataload.load(train=True, valid=True, test=False)
    ds_train, ds_valid = dataload.create_ds(dtrain=dtrain, dvalid=dvalid, batch_size=4096, batch_size_valid=4096)

    model, premodel = builder.build_model(T, n_x)
    model.load_weights(os.path.join(folder, 'weights'))
    model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    # train
    y_true, y_pred = compute_predictions_gpu(model, ds_train)        
    np.save(os.path.join(folder, 'y_train.npy'), y_true)
    np.save(os.path.join(folder, 'X_train.npy'), y_pred)

    # valid
    y_true, y_pred = compute_predictions_gpu(model, ds_valid)        
    np.save(os.path.join(folder, 'y_valid.npy'), y_true)
    np.save(os.path.join(folder, 'X_valid.npy'), y_pred)

    print('extract_features completed')


def single_submission(folder, modeltype='transformer', tosave=True, gpu=False):
    """Creates submission file.
    
    Args:
        folder - path for the folder with weights, sumbission file saved here.
    """
    _, _, dtest = dataload.load(train=False, valid=False, test=True)
    ds, _ = dataload.create_ds(dtrain=dtest, batch_size=4096, goal='predict')

    tf.keras.backend.clear_session() # needed if used in a loop
    model, premodel = builder.build_model(T, n_x, modeltype)
    #model, premodel = builder.build_transformer(T, n_x) 
    premodel.load_weights(os.path.join(folder, 'weights'))

    if gpu:
        _, y_pred = compute_predictions_gpu(model, ds)
    else:
        _, y_pred = compute_predictions_cpu(model, ds)
    y_pred = y_pred.ravel()

    # create submission csv
    if tosave:
        labels = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv')) # read default
        labels['prediction'] = y_pred # assign predictions
        labels.to_csv(os.path.join(folder, 'sample_submission.csv'), index=False) # save to input folder
    print('single_submission completed', folder)

    return y_pred


def multiple_submissions(folder, combine=True, modeltype='reccurent'):
    """Creates multiple submission files."""
    if combine:
        labels = pd.read_csv(os.path.join(PATH_DATA, 'sample_submission.csv'))
        labels['prediction'] = np.zeros((len(labels),))
        n = 0
    
    for subfolder in [f'fold_{k}' for k in range(5)]:#os.listdir(folder):        
        path = os.path.join(folder, subfolder)
        y_pred = single_submission(path, modeltype=modeltype)
        if combine:
            labels['prediction'] += y_pred
            n += 1
    
    if combine:
        # create submission csv
        labels['prediction'] /= n
        labels.to_csv(os.path.join(folder, 'sample_submission.csv'), index=False)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-01-amex/predict.py
    """
    #extract_features()    
    single_submission(folder=os.path.join(PATH_WORKING, 'results', 'premodel_trans_cosine'))    
    #multiple_submissions(folder=os.path.join(PATH_WORKING, 'results', 'dataprep1_outliers'))