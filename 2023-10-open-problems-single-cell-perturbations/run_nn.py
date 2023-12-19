import sys, os, json, pickle
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(4243) 

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# alternatively seting in commandline: export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

import datagen
from dataprep import load_train_df
from shared import PATH_DATA, set_seed, mrrmse, unpivot_predictions
set_seed(4243)


MODEL_DIR = os.path.join(PATH_DATA, 'nn_pivot_1')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def loss_abs(yy, y_pred):
    y_true, mask = tf.split(yy, 2, axis=1)
    errors = tf.math.abs( y_true - y_pred )
    loss = tf.math.reduce_mean(errors * mask)
    return loss


def load_data():

    data = np.load(os.path.join(PATH_DATA, 'data_pivot_1', 'xx+features_6+122.npy'))
    xx = data[:,:2]
    features = data[:,2:]
    targets = np.load( os.path.join(PATH_DATA, 'data_pivot_1', 'targets.npy') )

    scalers = {
        'targets': StandardScaler().fit(targets),
        'features': StandardScaler().fit(features),
    }
    # pickle.dump(scalers, open(os.path.join(PATH_DATA, MODEL_DIR, 'scalers.pkl'), 'wb') )

    targets = scalers['targets'].transform(targets)
    features = scalers['features'].transform(features)    

    print('features.shape', features.shape)
    print('targets.shape', targets.shape)

    return xx, features, targets, scalers


def create_model(n_inputs, n_outputs=144):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs, )),
        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
        tf.keras.layers.Dense(n_outputs),
    ])
    return model  


def train():

    batch_size = 64
    epochs = 100

    _, features, targets, _ = load_data()

    ds_train = datagen.create_ds(features, targets, batch_size)
    ds_valid = None

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'model', 'weights_{epoch:02d}'),
            #save_best_only=True,
            save_weights_only=True,
            monitor='loss', #val_loss
            verbose=1)
    ]
    
    model = create_model(n_inputs=features.shape[1], n_outputs=targets.shape[1])               
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=tf.keras.losses.MeanSquaredError())
    model.summary() 
    #model.load_weights(os.path.join(PATH_DATA, 'modeling2', 'model', 'weights'))
    
    model.fit(ds_train, validation_data=ds_valid, epochs=epochs, callbacks=callbacks)
    #model.save_weights(os.path.join(folder, 'model_proba', 'weights'))    


def evaluate():

    train_df = load_train_df()
    submission_df = pd.read_csv( os.path.join(PATH_DATA, 'submission_0.528_nn.csv') )

    id_map_df = pd.read_csv( os.path.join(PATH_DATA, 'input', 'id_map.csv')  )
    id_map_df[['cell_type', 'sm_name']] = np.load( os.path.join(PATH_DATA, 'data', 'x_test_ordinal.npy') )

    xx, features, _, scalers = load_data()   

    epochs = list(range(96,101))
    y_pred = 0
    for epoch in epochs:
        model = create_model(n_inputs=features.shape[1])
        model.load_weights(os.path.join(PATH_DATA, 'nn_pivot_1', 'model', f'weights_{epoch}'))
        y_pred += model.predict(features, batch_size=1024) 
    y_pred = scalers['targets'].inverse_transform(y_pred / len(epochs))

    data = unpivot_predictions(xx, y_pred) 
    pred_df = pd.DataFrame(data, columns=train_df.columns)
     
    pred_df_train = train_df[['cell_type', 'sm_name']].merge(pred_df, on=['cell_type', 'sm_name'], how='left')
    score = mrrmse(train_df.loc[:,'A1BG':].values, pred_df_train.loc[:,'A1BG':].values)
    print('train score:', score)    
         
    test_df_pred = id_map_df.merge(pred_df, on=['cell_type', 'sm_name'], how='left')
    score = mrrmse(submission_df.loc[:,'A1BG':].values, test_df_pred.loc[:,'A1BG':].values)
    print('test score:', score)
    
    filename = os.path.join(MODEL_DIR, 'submission.csv')
    test_df_pred.drop(columns=['cell_type', 'sm_name']).to_csv(filename, index=False)


if __name__ == "__main__":
    train()
    evaluate()