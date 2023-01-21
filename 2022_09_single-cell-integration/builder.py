import tensorflow as tf
import numpy as np


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


def build_multi_2(n_inputs, n_outputs=23418, n_units=256, rate=0.1, l2=1e-3, n_embed=16): 
    """Based on addition"""   

    x = tf.keras.layers.Input(shape=(n_inputs,))

    batch_size = tf.shape(x)[0]
    
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Lambda(lambda t: t[:,None,:]),
    ])
    
    embeder = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=n_outputs, output_dim=n_embed),
        tf.keras.layers.Dense(n_units, kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.Lambda(lambda t: t[None,:,:]),
    ])
    
    decoder = tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),

        tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.Flatten(),
    ])

    x_h = encoder(x)
    
    x_emb = embeder( tf.range(start=0, limit=n_outputs, delta=1) )

    x_feat = x_h + x_emb
    #feats = [x_h[:batch_size//2]+x_emb, x_h[batch_size//2:]+x_emb] # split operation in 2: for GPU memory efficiency
    #x_feat = tf.concat(feats, axis=0)
    
    y = decoder(x_feat) #tf.concat([layer_y(x_feat) for x_feat in feats], axis=0)

    model = tf.keras.models.Model(inputs=x, outputs=y)

    return model


# =======================================
# With pretraining
# =======================================
def cutmix_fn(t, pmix=0.3): # pmix=0.3 in paper
    return tf.where(tf.random.uniform(tf.shape(t)) < pmix, t, tf.roll(t, -1, axis=0))

def mixup_fn(t, alpha=0.2):
    t1 = tf.roll(t, 1, axis=0)
    return alpha * t + (1 - alpha) * t1

def contrastive_fn(h, h1, tau=0.7):

    batch_size = tf.shape(h)[0]

    dist0 = tf.matmul(h, h, transpose_b=True) * (1 - tf.eye(batch_size))
    dist1 = tf.matmul(h, h1, transpose_b=True)

    dist = tf.concat([dist0, dist1], axis=1) / tau

    prob = tf.keras.activations.softmax(dist)
    out = tf.linalg.diag_part(prob[:,batch_size:])
    out = tf.reshape(out, [-1,1])

    return out
    

def build_cite_selfsv(n_inputs, n_outputs, n_units, rate=0.3, l2=1e-3):

    x = tf.keras.layers.Input(shape=(n_inputs,))

    # compute hidden representation of inputs
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),    

        tf.keras.layers.Dense(n_units, activation='linear', kernel_regularizer=tf.keras.regularizers.L2(l2)),  
    ])
    
    # predict targets
    prediction = tf.keras.Sequential([
        #tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),  

        tf.keras.layers.Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.L2(l2))
    ])

    # maps representations to the space where contrastive loss is applied
    projection = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(l2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate),  

        tf.keras.layers.Dense(n_units//2, kernel_regularizer=tf.keras.regularizers.L2(l2))
    ]) 

    # augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.Lambda(cutmix_fn, name='CutMix'),
        encoder,
        tf.keras.layers.Lambda(mixup_fn, name='MixUp'),
        projection,
    ])

    # targets prediction
    h = encoder(x)
    y = prediction(h)    
    
    # contrastive
    o = projection(h)
    o1 = augmentation(x)
    y1 = contrastive_fn(o, o1)
    
    model = tf.keras.models.Model(inputs=x, outputs=y)
    premodel = tf.keras.models.Model(inputs=x, outputs=tf.concat([y, y1], axis=1))

    return model, premodel