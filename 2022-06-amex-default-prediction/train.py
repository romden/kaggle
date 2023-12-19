import sys, os, json, dill
import numpy as np

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import builder, losses, dataload, utils, custom
from config import PATH_DATA, PATH_WORKING, SEED

tf.random.set_seed(SEED)
np.random.seed(SEED)

FOLDER_RESULT = f'{PATH_WORKING}/results'
T, n_x = dataload.get_modeldims()

def lrfn(epoch):
    lr0 = 1e-4
    delta = 0.1e-4
    lb = 0.2e-4
    step = 10
    lr = max(lr0 - (epoch//step) * delta, lb)    
    return lr 


def pretrain():

    dtrain, dvalid, dtest = dataload.load(train=True, valid=True, test=True)
    ds_train, _ = dataload.create_ds(dtrain=dataload.merge(dtrain, dvalid, dtest), batch_size=512, goal='pretrain')

    strategy = tf.distribute.MirroredStrategy()    
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = losses.semi_supervised_loss()
        _, premodel = builder.build_transformer(T, n_x)

    premodel.compile(optimizer=optimizer, loss=loss)
    premodel.fit(ds_train, batch_size=1, epochs=100, verbose=1)
    premodel.save_weights(os.path.join(FOLDER_RESULT, 'premodel_trans_cosine', 'weights'))


def train():    

    # Train without valid data
    dtrain, dvalid, _ = dataload.load(train=True, valid=True, test=False)
    ds_train, ds_valid = dataload.create_ds(dtrain=dtrain, dvalid=dvalid, batch_size=512)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = ['binary_crossentropy'] 
        model, premodel = builder.build_model(T, n_x)

    premodel.load_weights(os.path.join(FOLDER_RESULT, 'premodel_trans_contrastive', 'weights'))    
    for layer in model.layers:
        layer.trainable = False 
    model.layers[-1].trainable = True
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    folder = os.path.join(FOLDER_RESULT, 'model_trans_contrastive')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath = os.path.join(folder, '{epoch}', 'weights'),
            save_weights_only = True,
            save_freq = 'epoch',
            verbose = 1)
    ]
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(folder, 'weights'),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_binary_crossentropy',
            #mode='max',
            verbose=1)
        ]
    history = model.fit(ds_train, validation_data=ds_valid, batch_size=1, epochs=10, callbacks=callbacks)

    with open(os.path.join(folder, 'history.json'), 'w') as f:
        f.write(json.dumps(history.history, indent=4))


def custom_train():    

    dtrain, dvalid, _ = dataload.load(train=True, valid=True, test=False)
    ds_train, ds_valid = dataload.create_ds(dtrain=dtrain, dvalid=dvalid, batch_size=512, batch_size_valid=512)
    datasets = {'train': ds_train, 'valid': ds_valid}

    tf.compat.v1.enable_eager_execution()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        loss = losses.binary_cross_entropy()
        #loss = losses.focal_bce()
        #loss = losses.amex_loss()        
        model = builder.build_model(T, n_x, model='transformer') #transformer reccurent

    # CUSTOM LEARNING SCHEUDLE  
    #callbacks = [tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)]
    folder = os.path.join(FOLDER_RESULT, 'model_trans')
    
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
    model.summary()
    custom.train(datasets, model, optimizer, loss, epochs=50, folder=folder, lrfn=None, strategy=strategy)


def multiple():

    dtrain, dvalid, dtest = dataload.load(train=True, valid=True, test=False)
    data = dataload.merge(dtrain, dvalid)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

    for k, (train_index, test_index) in enumerate(skf.split(data['y'], data['y'])):

        print('fold:', k)

        dtrain = {key: data[key][train_index] for key in data.keys()}
        dvalid = {key: data[key][test_index] for key in data.keys()}

        ds_train, ds_valid = dataload.create_ds(dtrain=dtrain, dvalid=dvalid, batch_size=512, batch_size_valid=5000)

        datasets = {'train': ds_train, 'valid': ds_valid}
        folder = os.path.join(FOLDER_RESULT, 'dataprep1_outliers_trans', f'fold_{k}')

        tf.keras.backend.clear_session()

        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            #loss = losses.binary_cross_entropy()
            loss = losses.amex_loss(rewarded=True, prob=False, weighted=False)
            model = builder.build_model(T, n_x, model='transformer')
            
        model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
        custom.train(datasets, model, optimizer, loss, epochs=30, folder=folder, strategy=strategy)


def train_self_loop():  

    import pandas as pd
    from predict import single_submission, multiple_submissions

    folder = os.path.join(PATH_WORKING, 'results', 'dataprep1_outliers_gru_self')
    n_folds = 5
    modeltype = 'reccurent'

    # submission update
    if False:
        multiple_submissions(folder=folder, combine=True, modeltype=modeltype)
    
    def update_sample_submission(n_folds, dtest, folder):
        lst = []
        for k in range(n_folds):
            labels = pd.read_csv(os.path.join(folder, f'fold_{k}', 'sample_submission.csv'))
            lst.append(labels['prediction'].values.reshape(-1,1))
        y = np.mean(np.concatenate(lst, axis=1), axis=1, keepdims=True)
        dtest['y'] = y
        labels['prediction'] = y.ravel()
        labels.to_csv(os.path.join(folder, 'sample_submission.csv'), index=False)
        print('sample_submission updated', os.path.join(folder, 'sample_submission.csv'))


    # preprocessed data
    dtrain, dvalid, dtest = dataload.load(train=True, valid=True, test=True)
    data = dataload.merge(dtrain, dvalid)
    
    dfolds = {}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=100)
    for k, (train_index, test_index) in enumerate(skf.split(data['y'], data['y'])):
        dtrain = {key: data[key][train_index] for key in data.keys()}
        dvalid = {key: data[key][test_index] for key in data.keys()}
        dfolds[k] = {'dtrain': dtrain, 'dvalid': dvalid}

    
    # training loop: for number of outer epochs and for each fold
    for i in range(10):

        update_sample_submission(n_folds, dtest, folder)

        for k in range(n_folds):

            print('Iter:', i, 'Fold:', k)            
            
            dtrain, dvalid = dfolds[k]['dtrain'], dfolds[k]['dvalid']
            ds_train, ds_valid = dataload.create_ds(dtrain=dataload.merge(dtrain, dtest), dvalid=dvalid, batch_size=512*4, batch_size_valid=5000)  
            datasets = {'train': ds_train, 'valid': ds_valid}

            tf.keras.backend.clear_session()
            tf.compat.v1.enable_eager_execution()
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
                #loss = losses.binary_cross_entropy()
                loss = losses.amex_loss(rewarded=True, prob=True, weighted=False)
                model = builder.build_model(T, n_x, model=modeltype)
            
            model.load_weights(os.path.join(folder, f'fold_{k}', 'weights'))
            model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
            custom.train(datasets, model, optimizer, loss, epochs=10, folder=os.path.join(folder, f'fold_{k}'), strategy=strategy)

            _ = single_submission(os.path.join(folder, f'fold_{k}'), modeltype=modeltype, tosave=True, gpu=True)


def train_self():  

    dtrain, dvalid, dtest = dataload.load(train=True, valid=True, test=True)
    ds_train, ds_valid = dataload.create_ds(dtrain=dataload.merge(dtrain, dtest), dvalid=dvalid, batch_size=512*4, batch_size_valid=5000)  
    datasets = {'train': ds_train, 'valid': ds_valid}

    tf.compat.v1.enable_eager_execution()
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        #loss = losses.binary_cross_entropy()
        loss = losses.amex_loss()
        model = builder.build_model(T, n_x, model='reccurent')

    # CUSTOM LEARNING SCHEUDLE
    folder = os.path.join(FOLDER_RESULT, 'model_gru_self')
    
    model.load_weights(os.path.join(folder, 'weights'))
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
    model.summary()
    custom.train(datasets, model, optimizer, loss, epochs=50, folder=folder, strategy=strategy)

if __name__ == "__main__":
    """
    eval "$(/opt/anaconda_teams/bin/conda shell.bash hook)"
    conda activate /home/datashare/envs/rden-py37-tf-gpu
    python other_research/kaggle-01-amex/train.py
    """
    pretrain()
    #train()   
    #custom_train()
    #multiple()
    #train_self_loop()
    #train_self()


#cp -R /home/datashare/datasets/kaggle/working/amex-default-prediction/results/dataprep1_outliers_trans /home/datashare/datasets/kaggle/working/amex-default-prediction/results/dataprep1_outliers_trans_self

#amex_loss before sort reconstruct
#[Train] loss: 0.4430 amex: 0.8075 auc: 0.9651 bce: 0.2079
#[Valid] best: 0.7866 amex: 0.7842 auc: 0.9592 bce: 0.2247