"""
Implements functions for custom training and prediction.
"""
import os
import dill
import json
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import tensorflow as tf

from utils import amex_metric_mod


def compute_predictions_cpu(model, ds, from_logits=False):
    """Cumput predictions."""
    # init
    trues = []
    preds = []

    # iterate dataset cumputing predictions
    for x, y in ds:        
        outputs = model(x, training=False)
        y_pred = tf.sigmoid(outputs) if from_logits else outputs
        trues.append(tf.reshape(y, [-1]).numpy())
        preds.append(tf.reshape(y_pred, [-1]).numpy())

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)

    return y_true, y_pred


def compute_predictions_gpu(model, ds, from_logits=False, devices=['/device:gpu:0', '/device:gpu:1']):
    
    from queue import Queue
    from threading import Thread

    def eval_func(device, model, x_batch):
        with tf.device(device):
            outputs = model(x_batch, training=False)
            y_pred = tf.sigmoid(outputs) if from_logits else outputs
        return y_pred   

    def worker(device, model, inputq, outputq):
        while True:
            x_batch = inputq.get()
            y_pred = None if x_batch is None else eval_func(device, model, x_batch)
            outputq.put(y_pred)
            if y_pred is None:
                break
    
    def get_batch(itr):
        try:
            batch = next(itr)
        except:
            batch = None
        return batch
        

    itr = iter(ds)
    preds = []
    trues = []

    # init models and queues
    queues = {}
    for device in devices:
        # init model on devices
        #with tf.device(device):
        #    model.load_weights(folder_step + '/weights')
            
        # init queues
        queues[device] = {'input': Queue(), 'output': Queue()}

        # launch worker thread
        Thread(target=worker, args=[device, model, queues[device]['input'], queues[device]['output']]).start()

    # go through batches
    while True:
        # add x_batch to input queue of each worker
        for device in devices:
            batch = get_batch(itr)
            if batch:
                x_batch, y_batch = batch
                trues.append(y_batch)
            else:
                x_batch, y_batch = None, None
            queues[device]['input'].put(x_batch)   

        # get y_pred from output queue of each worker
        tostop = False            
        for device in devices:
            y_batch = queues[device]['output'].get()
            if y_batch is not None:
                preds.append(y_batch)
            else:
                tostop = True
        
        # break the loop dataset has no more batches
        if tostop:
            # make sure all threads are stopped
            for device in devices:
                queues[device]['input'].put(None)
            break
            
    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)

    return y_true, y_pred


def apply_gradient(batch, model, optimizer, loss):
    """Perform optimization step using gradient descent."""  
    # unpack
    x_batch, y_batch = batch
    # compute loss
    with tf.GradientTape() as tape:
        # Predictions for this minibatch 
        y_pred = model(x_batch)
        # Loss value for this minibatch
        loss_value = loss(y_true=y_batch, y_pred=y_pred)
        # Add extra losses created during this forward pass:
        loss_value += sum(model.losses)
    # compute gradient and update model parameters
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return loss_value


def update_history(history, metrics):
    """Update history dictionary."""
    for key, val in metrics.items():
        if key not in history:
            history[key] = []
        history[key].append(float(val))


def compute_metrics(dataset, model, strategy=None):

    # compute predictions on dataset
    if strategy is not None:
        y_true, y_pred = compute_predictions_gpu(model, dataset)
    else:
        y_true, y_pred = compute_predictions_cpu(model, dataset) # make sure drop_remainder=False
    
    y_true, y_pred = y_true.ravel(), y_pred.ravel()

    # compute metrics
    metrics = {'amex': amex_metric_mod(y_true, y_pred),
               'auc': roc_auc_score(y_true, y_pred),
               'bce': log_loss(y_true, y_pred, eps=1e-5)
               }
    
    return metrics


def print_metrics(template, metrics):
    scores = []
    for name in ['best', 'loss', 'amex', 'auc', 'bce']:
        if name in metrics:
            template += ' ' + name + ': {:.4f}'
            scores.append(metrics[name])
    print(template.format(*scores))


def train(datasets, model, optimizer, loss, epochs, folder, lrfn=None, strategy=None):  
    """Custom model training."""
    # ensure folder for storing results exists
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # init
    history = {'train': {}, 'valid': {}}
    best_metric = compute_metrics(datasets['valid'], model, strategy)['amex'] #-float('inf')
    step = None
    train_metrics = {}
    valid_metrics = {}   
        
    for epoch in range(epochs):

        print('Epoch %d' % (epoch,)) 
        pbar = tqdm(total=step, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')    
        losses = []   
        if lrfn is not None:
            optimizer.lr.assign(lrfn(epoch))
        
        # train model for one epoch on training dataset
        for step, batch in enumerate(datasets['train']):

            # gradient descent on batch
            if strategy:
                per_replica_losses = strategy.run(apply_gradient, args=(batch, model, optimizer, loss))
                loss_value = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            else:
                loss_value = apply_gradient(batch, model, optimizer, loss)

            # store statistics
            losses.append(float(loss_value.numpy()))

            # show status in progress bar
            pbar.set_description("Epoch %s step %s loss %.4f" % (epoch, step, losses[-1]))
            pbar.update()

            # metrics on validation data [track after each gradient update]
            if False: #step == steps_per_epoch: #epoch > 10 or 
                valid_metrics = compute_metrics(datasets['valid'], model, strategy)
                update_history(history['valid'], valid_metrics)

                current_metric = valid_metrics['amex']
                if current_metric > best_metric:
                    best_metric = current_metric
                    model.save_weights(os.path.join(folder, 'weights'))

        # metrics on train data  
        train_metrics = compute_metrics(datasets['train'], model, strategy)
        train_metrics['loss'] = np.mean(losses)
        update_history(history['train'], train_metrics)

        # metrics on validation data        
        valid_metrics = compute_metrics(datasets['valid'], model, strategy)
        update_history(history['valid'], valid_metrics)

        current_metric = valid_metrics['amex']
        if current_metric > best_metric:
            best_metric = current_metric
            model.save_weights(os.path.join(folder, 'weights'))

        # print info
        print_metrics('[Train]', train_metrics)
        print_metrics('[Valid]', {**valid_metrics, **{'best': best_metric}})
    
    with open(os.path.join(folder, 'history.json'), 'w') as f:
        f.write(json.dumps(history, indent=4))