import os, sys
import pickle
import logging
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
import xgboost

from dataprep import load_train_df
from shared import PATH_DATA, mrrmse, set_seed, unpivot_predictions
set_seed(4243)


class SVDFeaturizer:

    def __init__(self):
        self.embeddings = None

    def svd_embedding(self, values, size):
        reducer  = TruncatedSVD(n_components=size, n_iter=7)
        embedding = reducer.fit_transform(values)        
        return embedding
    
    def compute_aggregates(self, x, yy, func):
        data = np.concatenate([x.reshape(-1,1), yy], axis=1)
        df = pd.DataFrame(data)
        values = df.groupby(0).agg(func).values     
        return values

    def apply_embedding(self, x, embedding):
        n = len(x)
        m = embedding.shape[1]
        features = np.zeros((n, m), dtype='float32')
        for i, idx in enumerate(x.astype('int32')):
            features[i,:] = embedding[idx]
        return features

    def fit(self, xx, yy):
        data = unpivot_predictions(xx[:,:2], yy)
        aggregates = {
            'cell_type': self.compute_aggregates(data[:,0], data[:,2:], func='mean'),
            'sm_name': data[:,2:].T
        } 
        self.embeddings = {
            'cell_type': self.svd_embedding(aggregates['cell_type'], size=6),
            'sm_name': self.svd_embedding(aggregates['sm_name'], size=64)
        }
        return self
    
    def transform(self, xx, yy=None):
        data = [
            self.apply_embedding(xx[:,0], self.embeddings['cell_type']),
            self.apply_embedding(xx[:,1], self.embeddings['sm_name']),
        ]
        features = np.concatenate(data, axis=1)
        return features  
    

class DummyFeaturizer:

    def __init__(self):
        pass

    def fit(self, xx=None, yy=None):
        return self

    def transform(self, xx, yy=None):
        features = xx[:,2:]
        return features     

class DummyReducer:

    def __init__(self):
        pass

    def fit(self, xx=None, yy=None):
        return self

    def transform(self, xx, yy=None):
        return xx  

class FoldEstimator:

    def __init__(self, n_estimators, max_depth, n_trials):
        self.n_estimators = n_estimators
        self.max_depth = max_depth 
        self.n_trials = n_trials
    
    def fit(self, x_train, x_valid, y_train, y_valid):

        n_targets = y_train.shape[1]
        errors = np.zeros((self.n_trials, n_targets), dtype='float32') 
        preds_train = np.zeros(y_train.shape, dtype='float32')
        preds_valid = np.zeros(y_valid.shape, dtype='float32')
        self.trails = []

        for k in range(self.n_trials):
            
            estimators = []
            indexes = np.random.permutation(n_targets)             

            for i, idx in enumerate(indexes):
                if i > 0:
                    features_train = np.concatenate([x_train, preds_train[:,indexes[:i]]], axis=1)
                    features_valid = np.concatenate([x_valid, preds_valid[:,indexes[:i]]], axis=1)
                else:
                    features_train = x_train
                    features_valid = x_valid

                estimator = xgboost.XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, early_stopping_rounds=30, n_jobs=-1, tree_method='gpu_hist')
                estimator = estimator.fit(features_train, y_train[:,idx], eval_set=[(features_valid, y_valid[:,idx])], verbose=0)
                estimators.append(estimator)

                preds_train[:,idx] = estimator.predict(features_train, iteration_range=(0, estimator.best_iteration + 1))
                preds_valid[:,idx] = estimator.predict(features_valid, iteration_range=(0, estimator.best_iteration + 1))

                errors[k, idx] = mean_squared_error(y_valid[:,idx], preds_valid[:,idx])            

            self.trails.append((estimators, indexes))

        self.best_predictors = np.argmin(errors, axis=0)
        return self     
    
    def estimate(self, x, estimators, indexes):
        y_pred = np.zeros((len(x), len(estimators)), dtype='float32')
        preds = []
        for estimator, idx in zip(estimators, indexes):
            features = np.concatenate([x] + preds, axis=1)
            y_pred[:,idx] = estimator.predict(features, iteration_range=(0, estimator.best_iteration + 1))
            preds.append(y_pred[:,[idx]])            
        return y_pred 
    
    def predict(self, features):
        n_targets = len(self.best_predictors)
        n_trials = len(self.trails)
        predictions = np.zeros((len(features), n_targets, n_trials), dtype='float32')  
        for k, (estimators, indexes) in enumerate(self.trails):
            predictions[:,:,k] = self.estimate(features, estimators, indexes)
        y_pred = predictions[:, range(n_targets), self.best_predictors]
        return y_pred
     

class Estimator:

    def __init__(self, scale, n_targets, n_estimators, max_depth, n_trials, n_models):
        self.scale = scale        
        self.n_targets = n_targets
        self.n_estimators = n_estimators
        self.max_depth = max_depth 
        self.n_trials = n_trials
        self.n_models = n_models
    
    def fit(self, xx, yy):   

        self.featurizer = SVDFeaturizer().fit(xx, yy) # DummyFeaturizer SVDFeaturizer
        features = self.featurizer.transform(xx)

        self.reducer = TruncatedSVD(n_components=self.n_targets, n_iter=7)
        yy = self.reducer.fit_transform(yy) 

        cell_type = xx[:,0]
        kfolds = StratifiedKFold(n_splits=5, shuffle=True).split(features, cell_type)

        self.models = []
        for train_index, test_index in kfolds:    
            x_train, x_valid, y_train, y_valid = features[train_index,:], features[test_index,:], yy[train_index], yy[test_index]            
            model  = FoldEstimator(self.n_estimators, self.max_depth, self.n_trials).fit(x_train, x_valid, y_train, y_valid)
            self.models.append(model)
            if len(self.models) >= self.n_models:
                break
        return self
    
    def predict(self, xx):
        features = self.featurizer.transform(xx)
        y_pred = np.zeros((len(features), self.n_targets), dtype='float32')       
        for model in self.models:
            y_pred += model.predict(features)
        y_pred = self.scale * self.reducer.inverse_transform( y_pred / len(self.models) )
        return y_pred
    

class Booster:

    def __init__(self, estimators=None):
        self.estimators = estimators if estimators is not None else []  

    def compute_residuals(self, y_true, y_pred):
        r = y_true - y_pred
        return r

    def fit(self, x_train, y_train, params=None, folder=None):

        train_df = load_train_df()               
        submission_df = pd.read_csv( os.path.join(PATH_DATA, 'submission_0.528_nn.csv') )
        id_map_df = pd.read_csv( os.path.join(PATH_DATA, 'input', 'id_map.csv')  ) 
        id_map_df[['cell_type', 'sm_name']] = np.load(os.path.join(PATH_DATA, 'data', 'x_test_ordinal.npy'))

        if params is None: params = {}
        learning_rate = params.get('learning_rate', 0.1)
        n_estimators = params.get('n_estimators', 100)
        estimator__n_targets = params.get('estimator__n_targets', 32)
        estimator__n_estimators = params.get('estimator__n_estimators', 300)
        estimator__max_depth = params.get('estimator__max_depth', 6)
        estimator__n_trials = params.get('estimator__n_trials', 1)
        estimator__n_models = params.get('estimator__n_models', 1)

        Fi = 0 if len(self.estimators) == 0 else self.predict(x_train)

        for i in range(len(self.estimators), len(self.estimators)+n_estimators):

            ri = self.compute_residuals(y_train, Fi)

            estimator = Estimator(
                scale=learning_rate, 
                n_targets=estimator__n_targets, 
                n_estimators=estimator__n_estimators, 
                max_depth=estimator__max_depth,
                n_trials=estimator__n_trials,
                n_models=estimator__n_models,
            )
            estimator.fit(x_train, ri) 
            self.estimators.append(estimator)    
            
            Fi += estimator.predict(x_train)

            score = mean_squared_error(y_train, Fi)

            data_pred = unpivot_predictions(x_train, Fi)
            pred_df = pd.DataFrame(data_pred, columns=train_df.columns)

            pred_df_train = train_df[['cell_type', 'sm_name']].merge(pred_df, on=['cell_type', 'sm_name'], how='left')
            train_mrrmse = mrrmse(train_df.loc[:,'A1BG':].values, pred_df_train.loc[:,'A1BG':].values)            
                
            test_df_pred = id_map_df.merge(pred_df, on=['cell_type', 'sm_name'], how='left')
            test_mrrmse = mrrmse(submission_df.loc[:,'A1BG':].values, test_df_pred.loc[:,'A1BG':].values) 
            
            n_estim = i + 1
            if True and n_estim % 1 == 0:
                logging.info(f'Booster: n_estimators={n_estim:02}, mse={score:.5f}, train_mrrmse={train_mrrmse:.5f}, test_mrrmse={test_mrrmse:.5f}')   

            if n_estim >= 10 and n_estim % 5 == 0:
                filename = os.path.join(folder, f'n_estimators={n_estim:02}', 'submission.csv')
                os.makedirs(os.path.dirname(filename), exist_ok=True)                
                test_df_pred.drop(columns=['cell_type', 'sm_name']).to_csv(filename, index=False)

            if False and folder is not None:
                booster = Booster(self.estimators)
                pickle.dump(booster, open(os.path.join(folder, 'booster.pkl'), 'wb'))

        return self 

    def predict(self, xx, iteration_range=None):        
        if iteration_range is None:
            iteration_range = range(len(self.estimators))
        y_pred = 0
        for i in iteration_range:  
            y_pred += self.estimators[i].predict(xx)            
        return y_pred
    

def train(folder):

    logging.info(f'Started {sys._getframe().f_code.co_name}')

    xx = np.load(os.path.join(PATH_DATA, 'data_pivot_1', 'xx+features_6+64.npy'))
    yy = np.load(os.path.join(PATH_DATA, 'data_pivot_1', 'targets_on_0.528.npy')) 
    
    params = {'learning_rate': 0.1, 'n_estimators': 300, 'estimator__n_trials': 2}
    booster = Booster().fit(xx, yy, params=params, folder=folder)
    
    pickle.dump(booster, open(os.path.join(folder, 'booster.pkl'), 'wb'))

    logging.info(f'Completed {sys._getframe().f_code.co_name}')


def predictions(folder):

    xx = np.load(os.path.join(PATH_DATA, 'data_pivot_1', 'xx+features_6+64.npy'))
    
    booster = pickle.load(open(os.path.join(folder, 'booster.pkl'), 'rb'))
    y_pred = booster.predict(xx) #, iteration_range=range(50)

    data = unpivot_predictions(xx[:,:2], y_pred)

    np.save(os.path.join(folder, 'xx+predictions.npy'), data) 


if __name__ == "__main__":
    folder = os.path.join(PATH_DATA, 'pivot_1_booster')
    os.makedirs(folder, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(folder, 'log'), filemode='w')
    
    train(folder)
    #predictions(folder)