import numpy as np
import pandas as pd
import os
import dill

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from dataload import load_data
from config import *


def read_evaluations():
    # Read the table of rows and columns required for submission
    eval_ids = pd.read_csv(FP_EVALUATION_IDS, index_col='row_id')

    # Convert the string columns to more efficient categorical types
    #eval_ids.cell_id = eval_ids.cell_id.apply(lambda s: int(s, base=16))
    eval_ids.cell_id = eval_ids.cell_id.astype(pd.CategoricalDtype())
    eval_ids.gene_id = eval_ids.gene_id.astype(pd.CategoricalDtype())

    return eval_ids


def reduce_multi_test_inputs():

    eval_ids = read_evaluations()
    cell_id_set = set(eval_ids.cell_id)
            
    # Read the 5000 rows and select the 30 % subset which is needed for the submission
    df = pd.read_hdf(FP_MULTIOME_TEST_INPUTS)
    mask = df.index.isin(cell_id_set)
    df = df.loc[mask]    

    df.to_hdf(FP_MULTIOME_TEST_INPUTS_reduced, key='df')
    print(df.shape)


def prep_cell_gene_ids():
    
    cite = {
        'cell_id': pd.read_hdf(FP_CITE_TEST_INPUTS).index.tolist(),
        'gene_id': pd.read_hdf(FP_CITE_TRAIN_TARGETS, start=0, stop=1).columns.tolist()
    }
    multi = {
        'cell_id': pd.read_hdf(FP_MULTIOME_TEST_INPUTS_reduced).index.tolist(),
        'gene_id': pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=0, stop=1).columns.tolist()
    }
    data = {'cite': cite, 'multi': multi}

    dill.dump(data, open(os.path.join(PATH_DATA, 'test_ids.dill'), 'wb'))


def dimentionality_reduction(dstype='cite', n_x=None, n_y=None, folder=PATH_WORKING):

    if not os.path.exists(folder):
        os.makedirs(folder)   

    X_train, y_train, X_test = load_data(dstype=dstype, train_x=(not (n_x is None)), train_y=(not (n_y is None)), test=(not (n_x is None)))

    # process X
    if n_x is not None:
        data = np.concatenate([X_train, X_test], axis=0)

        pipe_x = make_pipeline(TruncatedSVD(n_components=n_x, n_iter=7), StandardScaler()) # likely n_iter=7

        x_feat = pipe_x.fit_transform(data).astype(np.float32)

        np.save(os.path.join(folder, 'X_train.npy'), x_feat[:len(X_train)])
        np.save(os.path.join(folder, 'X_test.npy'),  x_feat[len(X_train):])

        dill.dump(pipe_x, open(os.path.join(folder, 'pipe_x.dill'), 'wb'))

    # process Y
    if n_y is not None:        
        pipe_y = make_pipeline(TruncatedSVD(n_components=n_y, n_iter=7), StandardScaler())

        y_feat = pipe_y.fit_transform(y_train).astype(np.float32)

        np.save(os.path.join(folder, f'y_train_{n_y}.npy'), y_feat)

        dill.dump(pipe_y, open(os.path.join(folder, f'pipe_y_{n_y}.dill'), 'wb'))             

    # process Y
    if True:        
        pipe_y = StandardScaler()

        y_feat = pipe_y.fit_transform(y_train).astype(np.float32)

        np.save(os.path.join(folder, 'y_train.npy'), y_feat)

        dill.dump(pipe_y, open(os.path.join(folder, 'pipe_y.dill'), 'wb'))


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/dataprep.py
    """
    #reduce_multi_test_inputs()
    #prep_cell_gene_ids()
    dimentionality_reduction(dstype='cite', n_x=512, n_y=128, folder=os.path.join(PATH_WORKING, 'data_cite_x512'))
    dimentionality_reduction(dstype='multi', n_x=1024, n_y=512, folder=os.path.join(PATH_WORKING, 'data_multi_x1024'))