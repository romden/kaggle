#import lightgbm as lgb 
#print('LGB Version', lgb.__version__) 

import xgboost as xgb
print('XGB Version', xgb.__version__)


def build_XGBClassifier(**kwargs):
    
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    params = {
        'learning_rate': 0.1,
        'max_depth': 4,
        'grow_policy': 'depthwise',

        # categorical features support:
        #'enable_categorical': True,
        #'max_cat_to_onehot': 1, # We use optimal partitioning exclusively

        'n_estimators': 300,
        #'early_stopping_rounds': 100,
        #'eval_metric': 'auc', #auc logloss
        #'scale_pos_weight': 1, # sum(negative instances) / sum(positive instances) 

        'objective': 'binary:logistic',        
        'use_label_encoder': False,
        'tree_method': 'gpu_hist', # hist [default= auto]
        "random_state": 4243,
        'n_jobs': -1,
    }
    params.update(kwargs)

    model = xgb.XGBClassifier(**params)

    return model

"""
def build_LGBClassifier(**kwargs):

    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    params = {
        "learning_rate": 0.1,        
        #"num_leaves": 31, # This is the main parameter to control the complexity of the tree model (default=31)        
        #"max_depth": -1, # limit the max depth for tree model (default=-1, means no limit)
        #"min_data_in_leaf": 2400, # minimal number of data in one leaf (default=20)
        #min_sum_hessian_in_leaf 🔗︎, default = 1e-3
        #"lambda_l1": 0, # L1 regularization (default=0)
        #"lambda_l2": 0, # L2 regularization (default=0)
        #'feature_fraction': 0.8,
        #'bagging_fraction': 0.7,      

        #'is_unbalance': True, # (default=False) Note: choose only one of them: is_unbalance or scale_pos_weight
        #'scale_pos_weight': sum(y_train<1)/sum(y_train), # (default=1) weight of labels with positive class

        "n_estimators": 300, # num_trees As a general rule, if you reduce num_iterations, you should increase learning_rate.
        #"early_stopping_rounds": 100, # if used best model with feval returned, otherwise not!

        "boosting_type": "gbdt",
        "objective": "binary",
        #"metric": "auc",

        "random_state": 4243,
        'n_jobs': -1,
    }
    params.update(kwargs)

    # sklearn API
    model = lgb.LGBMClassifier(**params)

    return model
"""