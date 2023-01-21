import numpy as np
import tensorflow as tf


def amex_metric_mod(y_true, y_pred):
    """ 
    References:
    https://www.kaggle.com/kyakovlev
    https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
    
    Explanation: https://www.kaggle.com/competitions/amex-default-prediction/discussion/327464    
    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)
    

def normal_sample(mu, sigma):
    return mu + tf.random.normal(tf.shape(mu), 0, 1) * sigma


def normal_pdf(x, mu, sigma):
    return 0.398942 * tf.math.exp(-0.5 * tf.math.pow((x - mu) / sigma, 2)) / (sigma + 1e-3)


def logit(y, eps=1e-5):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    y = tf.cast(y, dtype='float32')
    z = tf.keras.backend.clip(y, eps, 1.-eps)
    return -tf.math.log(1. / z - 1.)


def find_closest(queries, keys):
    """For each query finds closest key"""
    indexes = np.zeros((len(queries),), dtype='int32')
    for i, query in enumerate(queries):
        diff = keys - query.reshape(1,-1)
        dist = np.power(diff, 2).sum(axis=1)
        idx = np.argmin(dist)
        indexes[i] = idx
    return indexes


def assign_label(y_true, y_pred, indexes):
    """For each query finds closest key"""
    diff = np.abs(y_true[indexes]-y_pred[indexes])
    labels = np.where(diff<0.1, y_true, -1)
    return labels