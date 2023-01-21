import numpy as np
import pandas as pd
import os
import itertools
from collections import OrderedDict
import dill, json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

from transformers import *
from config import PATH_DATA, PATH_WORKING, PATH_DATAREADY, COLUMNS, SEED


def build_preprocessor():
    """Creates data processing pipeline."""
    
    # numerical pipeline
    pipe = make_pipeline(NonnegativeTransformer(),
                        #PowerTransformer(method='box-cox', standardize=True),
                        #-------------------
                        FunctionTransformer(np.log), 
                        OutlierTreatment(), # experimental
                        StandardScaler(),
                        #-------------------
                        FunctionTransformer(np.nan_to_num))

    # preprocessing pipeline
    preprocessor = ColumnTransformer([('cat', SimpleOnehotEncoder(), COLUMNS['cat']),                                      
                                      ('num', pipe, COLUMNS['num']),
                                      ('nan', NanEncoder(), COLUMNS['num']),
                                     ])

    return preprocessor


def get_dtypes():
    dtypes = {'customer_ID': np.dtype('O')}
    dtypes.update({col: 'category' for col in COLUMNS['cat']})
    dtypes.update({col: np.dtype('float32') for col in COLUMNS['num']})    
    return dtypes


def save_arrays(X, mask, y, folder):   
    """Saves numpy arrays."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    np.save(os.path.join(folder, 'X.npy'), X)
    np.save(os.path.join(folder, 'mask.npy'), mask) 
    np.save(os.path.join(folder, 'y.npy'), y)


def create_inputs(data, labels, preprocessor):
    """Creates numpy arrays that are fed to the model."""

    T = 13
    n_cust = labels.shape[0]    
    n_feat = preprocessor.transform(data[:1]).shape[1]
    
    X = np.zeros((n_cust, T, n_feat), dtype='float32')
    mask = np.zeros((n_cust, T), dtype='float32') # sequence mask (1 indicates present)
    y = np.zeros((n_cust, 1), dtype='float32')

    dsorted = data.sort_values(by='S_2', ascending=False) # sort so that latest are in same position and close to CLS  

    ids = dsorted['customer_ID'].values.reshape(-1, 1)  
    feats = preprocessor.transform(dsorted)
    
    df = pd.DataFrame(np.concatenate([ids, feats], axis=1), columns=['customer_ID']+list(range(feats.shape[1])))

    grouped = df.groupby('customer_ID')

    for i, custid, label in labels.reset_index(drop=True).itertuples():

        arr = grouped.get_group(custid).iloc[:,1:].values                
        t = arr.shape[0]

        X[i,:t,:] = arr
        mask[i,:t] = 1
        y[i,0] = label
    
    return X, mask, y


def process_data(dstype='train', n_splits=4):
    """Creates inputs for ML model.
    
    Args:
        dstype [str]: type of dataset {'train', 'test'}
        n_splits [int]: used to indicate proportion for train/valid split
    """    
    preprocessor = dill.load(open(os.path.join(PATH_DATAREADY, 'preprocessor.dill'), 'rb'))

    filelabels = 'train_labels.csv' if dstype=='train' else 'sample_submission.csv'
    filedata = 'train_data.csv' if dstype=='train' else 'test_data.csv'

    labels = pd.read_csv(os.path.join(PATH_DATA, filelabels))
    data = pd.read_csv(os.path.join(PATH_DATA, filedata), dtype=get_dtypes(), parse_dates=COLUMNS['time'])

    print('process_data loaded', dstype)
    print('labels.shape =', labels.shape)
    print('data.shape =', data.shape)    

    if dstype == 'train':
        # here we do single train/valid data split
        inputs = create_inputs(data, labels, preprocessor)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)  
        train_index, test_index = next(iter(skf.split(inputs[-1], inputs[-1])))

        train_inputs = [arr[train_index] for arr in inputs]
        test_inputs = [arr[test_index] for arr in inputs]
        
        save_arrays(*train_inputs, folder=os.path.join(PATH_DATAREADY, 'train'))
        save_arrays(*test_inputs, folder=os.path.join(PATH_DATAREADY, 'valid'))

    elif dstype == 'test':
        # for performance reasons process in 2 chunks
        cust = labels['customer_ID'].values
        k = len(cust)//2

        chunk = cust[:k]
        inputs1 = create_inputs(data[data['customer_ID'].isin(chunk)], labels[labels['customer_ID'].isin(chunk)], preprocessor)
        chunk = cust[k:]
        inputs2 = create_inputs(data[data['customer_ID'].isin(chunk)], labels[labels['customer_ID'].isin(chunk)], preprocessor)
        inputs = [np.concatenate([a1, a2], axis=0) for a1, a2 in zip(inputs1, inputs2)]

        save_arrays(*inputs, folder=os.path.join(PATH_DATAREADY, 'test'))

    print('array shapes', [arr.shape for arr in inputs])
    print('process_data completed', dstype)


def fit_preprocessor():
    """Fits preprocessing pipeline."""

    if not os.path.exists(PATH_DATAREADY):
        os.makedirs(PATH_DATAREADY)

    dtypes = get_dtypes()
    train = pd.read_csv(os.path.join(PATH_DATA, 'train_data.csv'), dtype=dtypes, parse_dates=COLUMNS['time'])
    test = pd.read_csv(os.path.join(PATH_DATA, 'test_data.csv'), dtype=dtypes, parse_dates=COLUMNS['time'])
    df = pd.concat([train, test], axis=0)
    print('fit_preprocessor df.shape =', df.shape)

    preprocessor = build_preprocessor()
    preprocessor = preprocessor.fit(df)

    dill.dump(preprocessor, open(os.path.join(PATH_DATAREADY, 'preprocessor.dill'), 'wb'))
    print('fit_preprocessor completed')


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-cpu
    python other_research/kaggle-01-amex/dataprep.py
    """
    fit_preprocessor()
    process_data('train')
    process_data('test')