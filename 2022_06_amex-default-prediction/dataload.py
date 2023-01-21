import os
import sys
import numpy as np
import tensorflow as tf

import datagen, dataprep

from config import PATH_WORKING, PATH_DATAREADY

KEYS = ('X', 'mask', 'y')


def get_modeldims():
    T = 13
    n_x = np.load(os.path.join(PATH_DATAREADY, 'valid', 'X.npy')).shape[-1]
    #print(f'n_x = {n_x}')
    return T, n_x


def load(train=True, valid=True, test=False):

    dtrain = {key: np.load(os.path.join(PATH_DATAREADY, 'train', f'{key}.npy')) for key in KEYS} if train else None
    dvalid = {key: np.load(os.path.join(PATH_DATAREADY, 'valid', f'{key}.npy')) for key in KEYS} if valid else None
    dtest = {key: np.load(os.path.join(PATH_DATAREADY, 'test', f'{key}.npy')) for key in KEYS} if test else None      
    
    return dtrain, dvalid, dtest


def merge(*data):
    return {key: np.concatenate([d[key] for d in data], axis=0) for key in KEYS}


def setup_autoshard(ds):
    """Disable AutoShard.
    
    https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)
    return ds


def create_ds(dtrain=None, dvalid=None, batch_size=512, batch_size_valid=10000, goal='train'):    
        
    output_signature=((tf.TensorSpec(shape=(None, None, None), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32)),
                        tf.TensorSpec(shape=(None, None), dtype=tf.float32))

    X_train, y_train = [dtrain[key] for key in KEYS[:-1]], dtrain[KEYS[-1]]
    gn_train = datagen.create_generator(X_train, y_train, batch_size, goal)
    ds_train = tf.data.Dataset.from_generator(gn_train, output_signature=output_signature)
    ds_train = setup_autoshard(ds_train)

    if dvalid is not None:
        X_valid, y_valid = [dvalid[key] for key in KEYS[:-1]], dvalid[KEYS[-1]]
        gn_valid = datagen.create_generator(X_valid, y_valid, batch_size_valid, goal='predict')
        ds_valid = tf.data.Dataset.from_generator(gn_valid, output_signature=output_signature)
        ds_valid = setup_autoshard(ds_valid)
    else:
        ds_valid = None

    return ds_train, ds_valid


def create_ds_tabular(batch_size=1024, batch_size_valid=10000, folder='model_fold_0'):

    def setup_ds(ds: tf.data.Dataset, batch_size, shuffle=True, seed=42, drop_remainder=True):        
        if shuffle:
            size_of_dataset = ds.reduce(0, lambda x, _: x + 1).numpy()
            ds = ds.shuffle(buffer_size=size_of_dataset, seed=seed)
        ds = ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    # DATA
    """
    train = {key: np.load(os.path.join(PATH_WORKING, 'train', f'{key}.npy')) for key in KEYS}
    valid = {key: np.load(os.path.join(PATH_WORKING, 'valid', f'{key}.npy')) for key in KEYS} 

    idx = 0
    X_train = np.concatenate([train['x_num'][:,idx,:], train['x_enc'][:,idx,:]], axis=-1)
    y_train = train['y']

    X_valid = np.concatenate([valid['x_num'][:,idx,:], valid['x_enc'][:,idx,:]], axis=-1)
    y_valid = valid['y']"""

    X_train = np.load(os.path.join(PATH_WORKING, 'results', folder, 'X_train.npy'))
    y_train = np.load(os.path.join(PATH_WORKING, 'results', folder, 'y_train.npy'))

    X_valid = np.load(os.path.join(PATH_WORKING, 'results', folder, 'X_valid.npy'))
    y_valid = np.load(os.path.join(PATH_WORKING, 'results', folder, 'y_valid.npy'))

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    ds_train = setup_ds(ds_train, batch_size=batch_size, drop_remainder=False)
    ds_valid = setup_ds(ds_valid, batch_size=batch_size_valid, shuffle=False)

    return ds_train, ds_valid