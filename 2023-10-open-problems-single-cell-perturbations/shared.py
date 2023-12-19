import os
import random

import numpy as np
import pandas as pd

#import tensorflow as tf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #tf.random.set_seed(seed) 


PATH_DATA = '/home/datashare/datasets/kaggle/open-problems-single-cell-perturbations'


def mrrmse(y_true, y_pred):
    """Mean Rowwise Root Mean Squared Error"""
    return np.mean( np.sqrt( np.mean(np.power(y_true - y_pred, 2), axis=1) ) )


# encoded with OrdinalEncoder
TEST_SM_NAME = {
    0: [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
        28,  29,  30,  32,  33,  34,  35,  36,  37,  38,  39,  40,  43,
        44,  45,  46,  47,  48,  49,  50,  51,  53,  54,  55,  56,  57,
        58,  59,  60,  61,  62,  63,  64,  66,  67,  68,  69,  70,  72,
        73,  74,  76,  77,  79,  80,  81,  82,  83,  84,  85,  86,  88,
        90,  91,  92,  93,  94,  97,  98,  99, 100, 102, 103, 104, 105,
       107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
       120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
       133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143],
    1: [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  13,
        14,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  32,  33,  34,  35,  36,  37,  38,  39,  40,  43,  44,
        45,  46,  47,  48,  49,  50,  51,  53,  54,  55,  56,  57,  58,
        59,  60,  61,  62,  63,  64,  66,  67,  68,  69,  70,  72,  73,
        74,  76,  77,  79,  80,  81,  82,  83,  84,  85,  86,  88,  90,
        91,  92,  93,  94,  97,  98,  99, 100, 102, 103, 104, 105, 107,
       108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
       134, 135, 136, 137, 138, 139, 140, 141, 142, 143]
}

NOT_PRESENT = {
    0: [27],
    1: [20, 30],
    4: [12,  30,  89, 112],
}


def unpivot_predictions(xx, y_pred):
    values = np.zeros((6*144, 18211), dtype='float32')
    labels = []
    for i in range(6):
        for j in range(144):
            labels.append([i,j])
            values[i*144+j,:] = y_pred[xx[:,0]==i,j]
    labels = np.array(labels, dtype='int32')
    data = np.concatenate([labels, values], axis=1) 
    return data

def get_indexes(xx_used, xx_all):
    idxs = []
    for pair in xx_used:
        mask = np.all(xx_all[:,:2]==pair.reshape(1,2), axis=1)
        idx = np.where(mask)[0]
        assert len(idx)==1
        idxs.extend(idx)
    return idxs