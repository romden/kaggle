import numpy as np
import pandas as pd
from config import *


def load_data(dstype='cite', train_x=True, train_y=True, test=True):

    X_train = None
    y_train = None
    X_test = None

    if dstype == 'cite':
        if train_x:
            X_train = pd.read_hdf(FP_CITE_TRAIN_INPUTS).values.astype('float32')  # (70988, 22050)
        if train_y:
            y_train = pd.read_hdf(FP_CITE_TRAIN_TARGETS).values.astype('float32') # (70988, 140)
        if test:
            X_test = pd.read_hdf(FP_CITE_TEST_INPUTS).values.astype('float32')    # (48663, 22050)

    elif dstype == 'multi':
        if train_x:
            X_train = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS).values.astype('float32')  # (105942, 228942)
        if train_y:
            y_train = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS).values.astype('float32') # (105942, 23418)     
        if test:
            X_test = pd.read_hdf(FP_MULTIOME_TEST_INPUTS).values.astype('float32') # (55935, 228942)
            #X_test = pd.read_hdf(FP_MULTIOME_TEST_INPUTS_reduced).values.astype('float32') # (16780, 228942) 
            
    print('X_train.shape:', X_train.shape if X_train is not None else None)
    print('y_train.shape:', y_train.shape if y_train is not None else None)
    print('X_test.shape:', X_test.shape if X_test is not None else None)

    return X_train, y_train, X_test


def split_data(X_train, y_train, k=10000):    
    dtrain = X_train[:-k,:], y_train[:-k,:]
    dtest = X_train[-k:,:], y_train[-k:,:]
    return dtrain, dtest