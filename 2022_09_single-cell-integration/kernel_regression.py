import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from scipy.spatial import distance_matrix
from concurrent.futures import ThreadPoolExecutor

from dataload import load_data
from utils import correlation_score
from config import *


def predict(Xi, X, Y, variables=None, donorm=True, batch_size=512):

    if donorm and True:
        X = X / np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        Xi = Xi / np.linalg.norm(Xi, ord=2, axis=1, keepdims=True)  

    n_samples, n_features = Xi.shape # number of samples to predict
    n_targets = Y.shape[1] # for each sample, number of targets to predict

    if variables is None:
        gammas = np.ones((n_targets,), dtype='float32')
        #weights = np.ones((1,n_features), dtype='float32')
        #variables = [gammas, weights]
        variables = [gammas]
    
    tail = 0
    preds = []

    while tail < n_samples:        
        head = tail
        tail = head + batch_size
        x_batch = Xi[head:tail,:]
        y_batch = predict_batch_tf(x_batch, X, Y, variables).numpy()
        preds.append(y_batch)
    
    return np.concatenate(preds, axis=0)


def matmult_parallel(A, B, devices=['/device:gpu:0', '/device:gpu:1']):

    def worker(a, b, device):
        with tf.device(device):
            return tf.matmul(a, b, transpose_b=True)

    k = B.shape[0]//2
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(worker, A, B[:k], devices[0])
        future2 = executor.submit(worker, A, B[k:], devices[1])
    
    return tf.concat([future1.result(), future2.result()], axis=1)


def matmult_parallel_batch(A, B):

    tail = 0
    batch_size = 10000
    n_samples = B.shape[0]
    results = []

    while tail < n_samples:        
        head = tail
        tail = head + batch_size
        B_batch = B[head:tail]
        C_batch = matmult_parallel(A, B_batch)
        results.append(C_batch)
    
    return tf.concat(results, axis=1)


def predict_batch_tf(Xi, X, Y, variables, mask=None):    

    gammas = variables[0]
    weights = variables[1] if len(variables) > 1 else None

    #dist = 1 - cosine_similarity_tf(Xi*weights, X*weights)
    #dist = 1 - tf.matmul(Xi, X, transpose_b=True) # cosine similarity
    dist = tf.matmul(Xi, X, transpose_b=True) if mask is not None else matmult_parallel_batch(Xi, X)

    n_targets = Y.shape[1]
    preds = []
    for j in range(n_targets):
        y = tf.reshape(Y[:,j], [1,-1])
        #Kmx = tf.math.exp(-gammas[j] * dist) #+ 1e-5 # kernel matrix 
        Kmx =  tf.pow(dist, gammas[j])

        if mask is not None: # only during train
            Kmx = Kmx * mask

        y_pred = tf.math.reduce_sum(Kmx * y, axis=1, keepdims=True) / tf.math.reduce_sum(Kmx, axis=1, keepdims=True)
        preds.append(y_pred)

    return tf.concat(preds, axis=1)


def train(dtrain, dvalid=None):

    X_train, y_train = dtrain
    X_train = X_train / np.linalg.norm(X_train, ord=2, axis=1, keepdims=True)    

    if dvalid is not None:
        X_valid, y_valid = dvalid
        X_valid = X_valid / np.linalg.norm(X_valid, ord=2, axis=1, keepdims=True)

    n_features = X_train.shape[1]
    n_samples, n_targets = y_train.shape
    epochs = 30
    batch_size = 512
    steps_per_epoch = int(np.floor(n_samples/batch_size))    
    lb = 1#1e-2
    ub = 10#1e+2
    best = {'score': -float('inf'), 'variables': None}

    variables = [] 
    variables.append(tf.Variable([1.]*n_targets, trainable=True)) 
    #variables.append(tf.Variable([[1.]*n_features], trainable=True)) 
    mask = 1 - tf.eye(batch_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    for epoch in range(1, epochs+1):

        indexes = np.random.permutation(n_samples)
        tail = 0
        step = 0
        losses = []        
        pbar = tqdm(total=steps_per_epoch, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ') 

        while tail < n_samples:

            head = tail
            tail = head + batch_size

            batch = indexes[head:tail]

            if len(batch) < batch_size:
                continue
            
            x_batch = X_train[batch,:]
            y_batch = y_train[batch,:]

            # compute loss
            with tf.GradientTape() as tape:
                # Predictions for this minibatch     
                y_pred = predict_batch_tf(x_batch, x_batch, y_batch, variables, mask)         

                # Loss value for this minibatch
                residuals = tf.math.pow(y_batch - y_pred, 2)
                residuals = tf.math.reduce_sum(residuals, axis=1)
                loss = tf.math.reduce_mean(residuals)

            # compute gradient and update model parameters
            gradients = tape.gradient(loss, variables)
            gradients = [tf.clip_by_norm(grad, 1.) for grad in gradients]
            optimizer.apply_gradients(zip(gradients, variables))
            
            # limits for gammas
            for j in range(n_targets):
                t = tf.clip_by_value(variables[0][j], lb, ub)
                variables[0][j].assign(t)            

            if len(variables) > 1:
                # limits for weights
                for j in range(n_features):
                    t = tf.clip_by_value(variables[1][0,j], 0., 10.)
                    variables[1][0,j].assign(t)
            
            # update stat
            losses.append(float(loss.numpy()))
            step += 1
            # show status in progress bar
            pbar.set_description("Epoch %s step %s loss %.4f" % (epoch, step, losses[-1]))
            pbar.update()
        
        # eval on valid data
        variables_np = [var.numpy() for var in variables]
        y_pred = predict(X_valid, X_train, y_train, variables_np, donorm=False)
        score = correlation_score(y_valid, y_pred)
        if score > best['score']:
            best['score'] = score
            best['variables'] = variables_np

        print(f'Epoch: {epoch}, loss: {np.mean(losses)}, val_score: {score}, val_best: {best["score"]}')
    
    return best['variables']


def main_train():

    dtrain, dtest = split_data( load_data(test=False)[:-1] )

    variables = train(dtrain, dtest)

    gammas = variables[0]    
    
    #print('weights', weights.shape, weights.min(), weights.max())  
    print('gammas', gammas.shape, gammas.min(), gammas.max())     
    print(gammas)


def main_predict():

    dstype = ['cite', 'multi'][1]

    X_train, y_train, X_test = load_data(dstype)

    y_pred = predict(X_test, X_train, y_train, batch_size=100)

    np.save(os.path.join(PATH_WORKING, f'y_pred_{dstype}.npy'), y_pred)


def main_predict_ensemble():

    dtrain, dtest = split_data( load_data(test=False)[:-1] )

    X_train, y_train = dtrain
    X_test, y_test = dtest

    n_rounds = 5
    n = X_train.shape[0]
    y_pred = np.zeros(y_test.shape, 'float32')

    for _ in range(n_rounds):
    
        indexes = np.random.permutation(n)            

        idx = indexes[:n//2]
        y_pred += predict(X_test, X_train[idx], y_train[idx])

        idx = indexes[n//2:]
        y_pred += predict(X_test, X_train[idx], y_train[idx])
    
    y_pred /= (n_rounds*2)

    score = correlation_score(y_test, y_pred)

    print('score', score)


if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-02-single-cell-integration/kernel_regression.py
    """
    #main_train()
    main_predict()
    #main_predict_ensemble()


# Epoch: 50, loss: 649.8153189009979, val_score: 0.7737890967312504, val_best: 0.7744472194452328

# cosine kernel (no training): score 0.7627226699464797
# predict_ensemble:            score 0.7627226542955428