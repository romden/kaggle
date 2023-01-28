import tensorflow as tf
import numpy as np


def smape_masked(y_true, y_pred):
    """SMAPE: https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error"""
    y_target = y_true[:,:7]
    mask = y_true[:,7:]
    errors = tf.math.abs(y_target - y_pred) / (tf.math.abs(y_target) + tf.math.abs(y_pred))
    errors = tf.where(mask>0, errors, 0)
    return tf.reduce_mean(errors)


def mean_squared_error_masked(y_true, y_pred):
    """Masked MeanSquaredError"""
    y_target = y_true[:,:7]
    mask = y_true[:,7:]
    errors = mask * tf.math.pow(y_target - y_pred, 2)
    return tf.reduce_mean(errors)


def build_ffn(n_inputs, n_outputs, n_units=0, rate=0.1, l1=0, l2=1e-3):

    x = tf.keras.layers.Input(shape=(n_inputs,))

    emb_cfips = tf.keras.layers.Embedding(input_dim=3135, output_dim=2)(x[:,0])
    emb_state = tf.keras.layers.Embedding(input_dim=51, output_dim=2)(x[:,1])

    feat = tf.concat([emb_cfips, emb_state, x[:,2:]], axis=1)

    # regularization penalty is computed as: loss = l1 * reduce_sum(abs(x)) + l2 * reduce_sum(square(x))
    reg = tf.keras.regularizers.L2(l2) if l1==0 else tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    if n_units > 0:
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=reg),
            tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.Dropout(rate),
            tf.keras.layers.Dense(n_outputs, kernel_regularizer=reg)
        ])
    else:
        ffn = tf.keras.layers.Dense(n_outputs, kernel_regularizer=reg)
        
    model = tf.keras.models.Model(inputs=x, outputs=ffn(feat))

    return model


def build_ffn2(n_inputs, n_outputs, n_units=0, rate=0.1, l2=1e-3):

    x = tf.keras.layers.Input(shape=(n_inputs,))

    emb_cfips = tf.keras.layers.Embedding(input_dim=3135, output_dim=2)(x[:,0])
    emb_state = tf.keras.layers.Embedding(input_dim=51, output_dim=2)(x[:,1])
    
    feat_cfips = tf.concat([emb_cfips, x[:,2:8], x[:,-2:]], axis=1) 
    feat_state = tf.concat([emb_state, x[:,8:12], x[:,-2:]], axis=1)

    # regularization penalty is computed as: loss = l1 * reduce_sum(abs(x)) + l2 * reduce_sum(square(x))
    #reg = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
    reg = tf.keras.regularizers.L2(l2)

    y_cfips = tf.keras.layers.Dense(n_outputs, kernel_regularizer=reg)(feat_cfips)
    y_state = tf.keras.layers.Dense(n_outputs, kernel_regularizer=reg)(feat_state)

    y = y_cfips * y_state  
        
    model = tf.keras.models.Model(inputs=x, outputs=y)

    return model