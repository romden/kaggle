"""
This is problem specific.
"""
import numpy as np
import tensorflow as tf


def create_ds(xx, yy, batch_size=128):

    output_signature=((tf.TensorSpec(shape=(None, None), dtype=tf.float32)),
                       tf.TensorSpec(shape=(None, None), dtype=tf.float32))
                       
    gn = create_generator(xx, yy, batch_size)
    ds = tf.data.Dataset.from_generator(gn, output_signature=output_signature)
    ds = setup_autoshard(ds)

    return ds


def setup_autoshard(ds):
    """Disable AutoShard."""
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    ds = ds.with_options(options)
    return ds


# def create_batch(x_batch, y_batch):
#     return (x_batch, y_batch)


def create_generator(xx, yy, batch_size=128, shuffle=True):
    """It is the problem specific implementation."""
    
    def wrapper():
        """The nested function that generates the data."""        

        data_size = len(xx)
        data_indexes = np.arange(data_size) 

        if shuffle:
            np.random.shuffle(data_indexes)

        # init params        
        batch_indexes = [0]*batch_size        
        idx = 0 # current location
        flag = True # indicates if there is next batch

        while flag:

            for i in range(batch_size):
                if idx >= data_size:
                    flag = False
                    idx = 0
                    if shuffle:
                        np.random.shuffle(data_indexes)  

                batch_indexes[i] = data_indexes[idx]            
                idx += 1

            yield (xx[batch_indexes], yy[batch_indexes])
        
    return wrapper