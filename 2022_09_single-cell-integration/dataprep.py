import numpy as np
import pandas as pd
import os
import dill
import itertools
import numpy.lib.recfunctions as rfn
import gc

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
            
    # Read rows and select the 30 % subset needed for the submission
    df = pd.read_hdf(FP_MULTIOME_TEST_INPUTS)
    mask = df.index.isin(cell_id_set)
    df = df.loc[mask]    

    df.to_hdf(FP_MULTIOME_TEST_INPUTS_reduced, key='df')
    print(df.shape)


def prepare_test_ids():
    
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


def prepare_data(dstype='cite', n_x=None, n_y=None, y_o=True):

    if dstype == 'cite':
        columns = pd.read_hdf(FP_CITE_TRAIN_INPUTS, start=0, stop=1).columns.tolist()
        X_train = pd.read_hdf(FP_CITE_TRAIN_INPUTS)[columns].values.astype('float32')  # (70988, 22050)
        y_train = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32') # (70988, 140)
        X_test = pd.read_hdf(FP_CITE_TEST_INPUTS)[columns].values.astype('float32')    # (48663, 22050)
        X_test_27678 = pd.read_hdf(FP_CITE_TEST_INPUTS_day_2_donor_27678)[columns].values.astype('float32') # (7016, 22085) fix cols
        X_unlabeled = np.concatenate([X_test[7476:], X_test_27678], axis=0) # due to data fix by organizers

    elif dstype == 'multi':        
        columns = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=0, stop=1).columns.tolist()
        X_train = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS)[columns].values.astype('float32')  # (105942, 228942)
        y_train = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values.astype('float32') # (105942, 23418)
        X_test = pd.read_hdf(FP_MULTIOME_TEST_INPUTS_reduced)[columns].values.astype('float32') # (16780, 228942) 
        X_unlabeled = pd.read_hdf(FP_MULTIOME_TEST_INPUTS)[columns].values.astype('float32') # (55935, 228942)

    # process X
    if n_x is not None:
        folder = os.path.join(PATH_WORKING, f'data_{dstype}_x{n_x}')
        if not os.path.exists(folder):
            os.makedirs(folder)

        data = np.concatenate([X_train, X_unlabeled], axis=0)

        pipe_x = make_pipeline(TruncatedSVD(n_components=n_x, n_iter=7), StandardScaler()) # likely n_iter=7
        pipe_x.fit(data)

        np.save(os.path.join(folder, 'X_train.npy'), pipe_x.transform(X_train).astype('float32'))
        np.save(os.path.join(folder, 'X_test.npy'),  pipe_x.transform(X_test).astype('float32'))
        np.save(os.path.join(folder, 'X_unlabeled.npy'),  pipe_x.transform(X_unlabeled).astype('float32'))

        dill.dump(pipe_x, open(os.path.join(folder, 'pipe_x.dill'), 'wb'))         

    # process Y
    if n_y is not None:
        folder = os.path.join(PATH_WORKING, f'data_{dstype}_y{n_y}')
        if not os.path.exists(folder):
            os.makedirs(folder)

        pipe_y = make_pipeline(TruncatedSVD(n_components=n_y, n_iter=7), StandardScaler()) 
        pipe_y.fit(y_train)

        np.save(os.path.join(folder, 'y_train.npy'), pipe_y.transform(y_train).astype('float32'))

        dill.dump(pipe_y, open(os.path.join(folder, 'pipe_y.dill'), 'wb'))

    
    # process Y without dimentionality reduction
    if y_o:
        folder = os.path.join(PATH_WORKING, f'data_{dstype}_y{y_train.shape[1]}')
        if not os.path.exists(folder):
            os.makedirs(folder)

        pipe_y = StandardScaler()
        pipe_y.fit(y_train)

        np.save(os.path.join(folder, 'y_train.npy'), pipe_y.transform(y_train).astype('float32'))

        dill.dump(pipe_y, open(os.path.join(folder, 'pipe_y.dill'), 'wb'))


def prepare_data_and_metadata(dstype='cite', n_x=None, n_y=None, y_o=True):
    """Prepares data with metadata"""

    metadata_df = pd.read_csv(FP_CELL_METADATA)
    metadata_df2 = pd.read_csv(FP_CELL_METADATA_cite_day_2_donor_27678)
    metadata_df = pd.concat([metadata_df, metadata_df2], axis=0)
    metadata_df = metadata_df.set_index('cell_id')[['day', 'donor', 'cell_type', 'technology']]
    
    from sklearn.preprocessing import LabelEncoder
    for col in metadata_df.columns:
        if col == 'day':
            metadata_df['day'] = metadata_df['day'].apply(lambda x: int(x>4)) # 0: days 2,3,4; 1: days: 7,10
        else:
            metadata_df[col] = LabelEncoder().fit_transform(metadata_df[col].values)

    if dstype == 'cite':

        columns = pd.read_hdf(FP_CITE_TRAIN_INPUTS, start=0, stop=1).columns.tolist()

        dfs = {}
        dfs['X_train'] = pd.read_hdf(FP_CITE_TRAIN_INPUTS)[columns].join(metadata_df) # (70988, 22050)
        dfs['y_train'] = pd.read_hdf(FP_CITE_TRAIN_TARGETS) # (70988, 140)
        dfs['X_test'] = pd.read_hdf(FP_CITE_TEST_INPUTS)[columns].join(metadata_df) # (48663, 22050)
        dfs['X_test_27678'] = pd.read_hdf(FP_CITE_TEST_INPUTS_day_2_donor_27678)[columns].join(metadata_df) # (7016, 22085) fix cols
        dfs['X_unlabeled'] = pd.concat([dfs['X_test'][7476:], dfs['X_test_27678']], axis=0) # due to data fix by organizers 

    elif dstype == 'multi':

        columns = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=0, stop=1).columns.tolist()

        dfs = {}
        dfs['X_train'] = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS)[columns].join(metadata_df) # (105942, 228942)
        dfs['y_train'] = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS) # (105942, 23418)
        dfs['X_test'] = pd.read_hdf(FP_MULTIOME_TEST_INPUTS_reduced)[columns].join(metadata_df) # (16780, 228942) 
        dfs['X_unlabeled'] = pd.read_hdf(FP_MULTIOME_TEST_INPUTS)[columns].join(metadata_df) # (55935, 228942)

    values = {}
    metadata = {}
    for key, df in dfs.items():
        assert all(df.notnull().all())
        if key == 'y_train':
            values[key] = df.values.astype('float32')
        elif key in ['X_train', 'X_test', 'X_unlabeled']:
            values[key] = df.iloc[:,:-4].values.astype('float32')
            metadata[key] = df.iloc[:,-4:].values.astype('float32')        
    
    # save metadata
    folder = os.path.join(PATH_WORKING, f'metadata_{dstype}')
    if not os.path.exists(folder):
        os.makedirs(folder)

    for key in ['X_train', 'X_test', 'X_unlabeled']:
        np.save(os.path.join(folder, f'{key}.npy'), metadata[key].astype('float32'))
    

    # process X
    if n_x:
        folder = os.path.join(PATH_WORKING, f'data_{dstype}_x{n_x}')
        if not os.path.exists(folder):
            os.makedirs(folder)

        data = np.concatenate([values['X_train'], ['X_unlabeled']], axis=0)

        pipe_x = make_pipeline(TruncatedSVD(n_components=n_x, n_iter=7), StandardScaler()) # likely n_iter=7
        pipe_x.fit(data)

        np.save(os.path.join(folder, 'X_train.npy'), pipe_x.transform(['X_train']).astype('float32'))
        np.save(os.path.join(folder, 'X_test.npy'),  pipe_x.transform(['X_test']).astype('float32'))
        np.save(os.path.join(folder, 'X_unlabeled.npy'),  pipe_x.transform(['X_unlabeled']).astype('float32'))

        dill.dump(pipe_x, open(os.path.join(folder, 'pipe_x.dill'), 'wb'))         

    # process Y
    if n_y:
        folder = os.path.join(PATH_WORKING, f'data_{dstype}_y{n_y}')
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        data = values['y_train']

        pipe_y = make_pipeline(TruncatedSVD(n_components=n_y, n_iter=7), StandardScaler()) 
        pipe_y.fit(data)

        np.save(os.path.join(folder, 'y_train.npy'), pipe_y.transform(values['y_train']).astype('float32'))

        dill.dump(pipe_y, open(os.path.join(folder, 'pipe_y.dill'), 'wb'))

    
    # process Y without dimentionality reduction
    if y_o:
        folder = os.path.join(PATH_WORKING, f'data_{dstype}_y{y_train.shape[1]}')
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        data = values['y_train']

        pipe_y = StandardScaler()
        pipe_y.fit(data)

        np.save(os.path.join(folder, 'y_train.npy'), pipe_y.transform(values['y_train']).astype('float32'))

        dill.dump(pipe_y, open(os.path.join(folder, 'pipe_y.dill'), 'wb'))


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-02-single-cell-integration/dataprep.py
    """
    #reduce_multi_test_inputs()
    #prepare_test_ids()

    prepare_data(dstype='cite', n_x=256, n_y=None, y_o=False)
    prepare_data(dstype='multi', n_x=256, n_y=None, y_o=False)

    prepare_data(dstype='cite', n_x=1024, n_y=None, y_o=False)
    prepare_data(dstype='multi', n_x=1024, n_y=None, y_o=False)

    #prepare_data_and_metadata(dstype='cite', n_x=None, n_y=None, y_o=False)   
    #prepare_data_and_metadata(dstype='multi', n_x=None, n_y=None, y_o=False)