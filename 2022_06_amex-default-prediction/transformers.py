import numpy as np
import pandas as pd
import os
import itertools
from collections import OrderedDict
import dill

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer, MinMaxScaler


class SimpleOnehotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encoding. Takes care of missing values.""" 
    def __init__(self):        
        def func(row, labels):
            lst = []
            for col, dct in labels.items():
                idx = dct.get(row[col], -1)
                tmp = [int(i==idx) for i in range(len(dct))]
                lst += tmp
            return lst      
        self.func = func

    def fit(self, X, y=None):
        self.labels = OrderedDict()
        for col in X.columns:  
            keys = set(X[col])     
            self.labels[col] = {key: i for i, key in enumerate(keys)}
        return self

    def transform(self, X, y=None):
        s = X.apply(lambda row: self.func(row, self.labels), axis=1)
        lst = s.values.tolist()
        arr = np.array(lst, dtype='float32')      
        return arr


class SimpleLabelEncoder(BaseEstimator, TransformerMixin):
    """Encodes labels. Unseen, missing and np.nan values are mapped to 0.""" 

    def __init__(self, offset=0, out_dtype='float32'):
        self.offset = offset # starting index of encoding, typically values in {0,1}
        self.out_dtype = out_dtype
        self.columns = None
        self.labels = None
    
    def get_feature_names(self):
        return self.columns

    def fit(self, X, y=None):
        self.columns = X.columns
        self.labels = {}        
        for col in self.columns:
            keys =[key for key in set(X[col]) if not (key is None) and key==key] # ensure removing np.nan and None        
            self.labels[col] = {key: i+self.offset for i, key in enumerate([np.nan] + keys)} # ensure that {np.nan: 0}
        return self

    def transform(self, X, y=None):
        vfunc = np.vectorize(lambda z: self.labels[col].get(z, self.offset)) # use numpy to allow handling nan
        arr = np.empty(X.shape, dtype=self.out_dtype)
        for j, col in enumerate(self.columns):
            arr[:,j] = vfunc(X[col])
        return arr


class NonnegativeTransformer(BaseEstimator, TransformerMixin):
    """Non negative values.""" 
    def __init__(self):
        self.lb = None
        self.ub = None
        
    def fit(self, X, y=None):
        values = X.values if isinstance(X, pd.DataFrame) else X
        self.lb = np.nanmin(values, axis=0, keepdims=True)
        self.ub = np.nanmax(values, axis=0, keepdims=True)
        return self

    def transform(self, X, y=None):
        values = X.values if isinstance(X, pd.DataFrame) else X     
        arr = np.clip(values, self.lb, self.ub) - self.lb + 1
        return arr.astype('float32')


class NanEncoder(BaseEstimator, TransformerMixin):
    """NAN encoding."""
    
    def __init__(self):
        self.mask = None
          
    def fit(self, X, y=None):
        values = X.values if isinstance(X, pd.DataFrame) else X
        self.mask = np.isnan(values).any(axis=0)
        return self    
        
    def transform(self, X, y=None):
        values = X.values if isinstance(X, pd.DataFrame) else X
        return np.isnan(values).astype('float32')#[:,self.mask]


def outlier_bounds(data, rate=1.5, nans=False): 
    # calculate interquartile range
    if nans:
        q25 = np.nanpercentile(data, 25, axis=0, keepdims=True)
        q75 = np.nanpercentile(data, 75, axis=0, keepdims=True) 
    else:
        q25 = np.percentile(data, 25, axis=0, keepdims=True)
        q75 = np.percentile(data, 75, axis=0, keepdims=True) 
    iqr = q75 - q25
    # calculate the outlier bounds
    lb = q25 - iqr * rate
    ub = q75 + iqr * rate
    # identify outliers
    #mask = np.logical_or(data < lb, data > ub)    
    return lb, ub, iqr


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Non negative values.""" 
    def __init__(self):
        self.lb = None
        self.ub = None
        self.ranges = None
        self.scalers = None
        
    def fit(self, X, y=None):
        data = X.values if isinstance(X, pd.DataFrame) else X

        self.lb, self.ub, iqr = outlier_bounds(data, rate=1.5, nans=True)

        self.scalers = {}
        self.scalers['lb'] = MinMaxScaler((-1, 0)).fit( np.where(data > self.lb, np.nan, data) )
        self.scalers['ub'] = MinMaxScaler(( 0, 1)).fit( np.where(data < self.ub, np.nan, data) )

        self.ranges = {}
        self.ranges['lb'] = np.minimum(self.scalers['lb'].data_range_, iqr*0.5).reshape(1,-1)
        self.ranges['ub'] = np.minimum(self.scalers['ub'].data_range_, iqr*0.5).reshape(1,-1)
        return self

    def transform(self, X, y=None):
        data = X.values if isinstance(X, pd.DataFrame) else X
        
        mask_lb = data > self.lb
        mask_ub = data < self.ub
        
        scaled_lb = self.scalers['lb'].transform( np.where(mask_lb, np.nan, data) )
        scaled_ub = self.scalers['ub'].transform( np.where(mask_ub, np.nan, data) )        

        arr = np.where(np.logical_and(mask_lb, mask_ub), data, np.nan)        
        arr = np.where(~mask_lb, self.lb + scaled_lb * self.ranges['lb'], arr)
        arr = np.where(~mask_ub, self.ub + scaled_ub * self.ranges['ub'], arr) 
        return arr