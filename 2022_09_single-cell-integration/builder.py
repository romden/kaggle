import tensorflow as tf
import numpy as np


def build_cite_ffn2(n_inputs, n_outputs, n_units, rate=0.3, l2=1e-3):

    def hidden_block():        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate),
        ])
        return model
    
    def output_block():
        model = tf.keras.Sequential([
            tf.keras.layers.Concatenate(),
            tf.keras.layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(l2)),
        ])
        return model

    x = tf.keras.layers.Input(shape=(n_inputs,))

    x1 = hidden_block()(x)
    x2 = hidden_block()(x1)  
    x3 = hidden_block()(x2)
    x4 = hidden_block()(x3)

    y = output_block()([x1, x2, x3, x4])

    model = tf.keras.models.Model(inputs=x, outputs=y)

    return model


def build_cite_ffn(n_inputs, n_outputs, n_units, rate=0.3, l2=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(l2))
    ])
    return model


def build_multi_ffn(n_inputs, n_outputs, n_units=256, rate=0.1, l2=1e-3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(l2))
    ])
    return model


# =======================================
# With pretraining
# =======================================
def cutmix_fn(t, pmix=0.3): # pmix=0.3 in paper
    return tf.where(tf.random.uniform(tf.shape(t)) < pmix, t, tf.roll(t, -1, axis=0))

def mixup_fn(t, alpha=0.2):
    t1 = tf.roll(t, 1, axis=0)
    return alpha * t + (1 - alpha) * t1

def contrastive_fn(inputs, tau=0.7):

    h, h1 = inputs

    batch_size = tf.shape(h)[0]

    dist0 = tf.matmul(h, h, transpose_b=True) * (1 - tf.eye(batch_size))
    dist1 = tf.matmul(h, h1, transpose_b=True)

    dist = tf.concat([dist0, dist1], axis=1) / tau

    prob = tf.keras.activations.softmax(dist)
    out = tf.linalg.diag_part(prob[:,batch_size:])
    out = tf.reshape(out, [-1,1])

    return out
    

def build_cite_selfsv(n_inputs, n_outputs, n_units, rate=0.3, l2=1e-3):   

    # compute hidden representation of inputs
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu'),
        tf.keras.layers.BatchNormalization(),  

        #tf.keras.layers.Dense(n_units, activation='relu'),
        #tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(n_units, activation='linear'),
        tf.keras.layers.BatchNormalization(),
    ], name='encoder')
    
    # predict targets
    prediction = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),  

        tf.keras.layers.Dense(n_outputs)#, kernel_regularizer=tf.keras.regularizers.L2(l2))
    ], name='prediction')

    # maps representations to the space where contrastive loss is applied
    projection = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(n_units, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(n_units//2),
    ], name='projection') 

    # augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.Lambda(cutmix_fn, name='CutMix'),
        encoder,
        tf.keras.layers.Lambda(mixup_fn, name='MixUp'),
        projection,
    ], name='augmentation')

    x = tf.keras.layers.Input(shape=(n_inputs,))

    # targets prediction
    h = encoder(x)
    y = prediction(h)   
    
    # contrastive    
    o = projection(h)
    o1 = augmentation(x)
    y1 = tf.keras.layers.Lambda(contrastive_fn, name='contrastive')([o, o1])

    model = tf.keras.models.Model(inputs=x, outputs={'contrastive': y1, 'prediction': y})

    return model


# =======================================
# autoencoder
# =======================================
def build_autoencoder(n_features=22050, n_latent=128, units=[256, 192]):

    # compute hidden representation of inputs
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.Dense(units[1], activation='relu'),        
        tf.keras.layers.Dense(n_latent, activation='linear'),
    ])
    
    # predict targets
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(units[1], activation='relu'),
        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.Dense(n_features, activation='linear'),
    ])

    autoencoder = tf.keras.models.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    return autoencoder, encoder