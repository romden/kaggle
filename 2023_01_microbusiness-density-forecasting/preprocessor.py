import numpy as np

from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# not used: 'first_day_of_month', 'active', 'population', 'target'


class SimpleLabelEncoder(BaseEstimator, TransformerMixin):
    """Encodes labels, including np.nan.
    
    Unseen labels and missing values are assigned to separate category. 
    """ 

    def __init__(self, columns=None):
        self.columns = columns
        self.labels = {}

        def func(z, labels):
            return labels.get(z, -1)
        self.vfunc = np.vectorize(func)

    def fit(self, X, y=None):   
        if self.columns is None:
            self.columns = X.columns  
        for col in self.columns:
            self.labels[col] = {key: i for i, key in enumerate(set(X[col]))}
        return self

    def transform(self, X, y=None):
        arr = np.empty((len(X), len(self.columns)), dtype='float32')
        for j, col in enumerate(self.columns):
            arr[:,j] = self.vfunc(X[col], self.labels[col])
        return arr


class CyclicEncoder(BaseEstimator, TransformerMixin):
    """CyclicEncoder""" 

    def __init__(self, colmax=None):
        self.colmax = colmax # {column_name: max_value}

        def cyclic_encoding(values, max_val):
            """Computes cyclic feature for a given array and max possible value."""
            arr = np.empty((values.size, 2), dtype='float32')
            arr[:,0] = np.sin(2*np.pi * values/max_val)
            arr[:,1]  = np.cos(2*np.pi * values/max_val)
            return arr
            
        self.encoding = cyclic_encoding

    def fit(self, X, y=None):   
        if self.colmax is None:
            self.colmax = {col: max(X[col]) for col in X.columns}
        return self

    def transform(self, X, y=None):
        lst = [self.encoding(X[col].values, self.colmax[col]) for col in X.columns]
        arr = np.concatenate(lst, axis=1)
        return arr


def build_preprocessor_nn(census=False):

    n_lags = 7
    embed_cols = ['cfips', 'state',]    
    time_cols = ['month']
    num_cols = [f'state_lag_{i}' for i in range(1, n_lags+1)]
    num_cols += [f'cfips_lag_{i}' for i in range(1, n_lags+1)]
    if census:      
        for key in ['pct_bb', 'pct_college', 'pct_foreign_born', 'pct_it_workers', 'median_hh_inc']:
            for year in range(2017, 2022):
                num_cols.append(f'{key}_{year}')

    cols = embed_cols + time_cols + num_cols

    # preprocessing pipeline 
    featurizer = ColumnTransformer([('embed_cols', SimpleLabelEncoder(), embed_cols),  
                                    ('time_cols', CyclicEncoder({'month': 12}), time_cols),                                                                  
                                    ('num_cols', StandardScaler(), num_cols)])
    
    extractor = FunctionTransformer(lambda df: df[cols])
    imputer = FunctionTransformer(lambda x: np.nan_to_num(x))

    preprocessor = Pipeline(steps=[('extractor', extractor), ('featurizer', featurizer), ('imputer', imputer)])

    return preprocessor
