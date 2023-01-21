import numpy as np
import tensorflow as tf


def setup_autoshard(ds):
    """Disable AutoShard.
    
    https://stackoverflow.com/questions/65322700/tensorflow-keras-consider-either-turning-off-auto-sharding-or-switching-the-a
    https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy
    """
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)
    return ds


def create_ds(X, y, batch_size=128, goal='train'):

    output_signature=((tf.TensorSpec(shape=(None, None), dtype=tf.float32)),
                       tf.TensorSpec(shape=(None, None), dtype=tf.float32))
                       
    gn = create_generator(X, y, batch_size, goal)
    ds = tf.data.Dataset.from_generator(gn, output_signature=output_signature)
    ds = setup_autoshard(ds)

    return ds


def create_generator(X, y, batch_size=128, goal='train'):
    if goal == 'train':
        generator = simple_generator(X, y, batch_size=batch_size, goal=goal) #, oversampling={0: 10, 1: 1}
    elif goal == 'pretrain':
        generator = simple_generator(X, y, batch_size=batch_size, goal=goal)       
    elif goal == 'predict':
        generator = simple_generator(X, y, batch_size=batch_size, shuffle=False, goal=goal)
    return generator


def create_batch(X, y, batch_indexes, goal):
     
    x_batch = X[batch_indexes]
    y_batch = y[batch_indexes]

    return (x_batch, y_batch)


def simple_generator(X, y, batch_size=128, shuffle=True, drop_remainder=True, oversampling=None, seed=1000, goal='train'):
    """The outer enclosing function."""

    np.random.seed(seed)
    
    def wrapper():
        """The nested function that generates the data."""

        # create an array with indexes
        indexes = np.arange(len(y))            
        
        if shuffle:
            np.random.shuffle(indexes)

        # init params        
        batch_indexes = [0]*batch_size
        data_size = len(indexes)
        idx = 0 # current location
        flag = True # indicates if there is next batch

        while flag:

            for i in range(batch_size):
                if idx >= data_size:
                    flag = False
                    idx = 0
                    if shuffle:
                        np.random.shuffle(indexes)                
                batch_indexes[i] = indexes[idx]            
                idx += 1

                # drop remainder if needed
                if idx == data_size and drop_remainder:
                    batch_indexes = batch_indexes[:i+1]
                    flag = False
                    break
            
            batch = create_batch(X, y, batch_indexes, goal)

            yield batch
        
    return wrapper