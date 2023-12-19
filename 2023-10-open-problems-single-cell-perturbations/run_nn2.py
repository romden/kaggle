import sys, os
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

from datagen import setup_autoshard, create_generator
from dataprep import load_train_df
from shared import PATH_DATA, set_seed, mrrmse, TEST_SM_NAME, NOT_PRESENT
set_seed(4243)


MODEL_DIR = os.path.join(PATH_DATA, 'nn2_pivot_1')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def make_mask():
    mask = np.ones((18211, 6, 144), dtype='float32')

    for j, k in NOT_PRESENT.items():
        mask[:,j,k] = 0

    for j, k in TEST_SM_NAME.items():
        mask[:,j,k] = 0.95

    return tf.convert_to_tensor(mask)

#MASK = make_mask()


def mrrmse_loss_T(y_true, y_pred):
    errors = tf.pow(y_true - y_pred, 2)
    errors = tf.math.sqrt( tf.math.reduce_mean(errors, axis=0) )
    loss = tf.math.reduce_mean(errors)
    return loss


def load_data():

    data = np.load(os.path.join(PATH_DATA, 'data_pivot_1', 'xx+features_6+128.npy'))
    xx = data[:,:2]
    features = data[xx[:,0]==0,8:]
    targets = np.load( os.path.join(PATH_DATA, 'data_pivot_1', 'targets.npy') )    

    scalers = {
        'features': StandardScaler().fit(features),
        'targets': StandardScaler().fit(targets),        
    }

    features = scalers['features'].transform(features) 
    targets = scalers['targets'].transform(targets)    

    outputs = np.zeros((18211, 6, 144), dtype='float32')
    for i in range(6):
        outputs[:,i,:] = targets[xx[:,0]==i,:]   

    print('features.shape', features.shape)
    print('outputs.shape', outputs.shape)

    return features, outputs, scalers


class Model(tf.keras.Model):

    def __init__(self, n_inputs, n_outputs=144, units=256):
        super().__init__() 
        
        self.featurizer = []
        for _ in range(6):
            layer = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(n_inputs, )), 
                #tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
                tf.keras.layers.Dense(units)
            ])
            self.featurizer.append(layer)        

        self.attention = tf.keras.layers.Attention(use_scale=True)

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(n_outputs, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.L2(1e-3)),
            tf.keras.layers.Dense(n_outputs),
        ])

    def call(self, inputs):     
        feats = tf.concat([layer(inputs)[:,None,:] for layer in self.featurizer], axis=1)
        data = tf.transpose(feats, [2, 1, 0])
        attn_out = self.attention([data, data])
        attn_out_T = tf.transpose(attn_out, [2, 1, 0])
        outputs = self.ffn(attn_out_T)
        return outputs    
    
    # def call(self, inputs):      
    #     """att2"""
    #     feats = tf.concat([layer(inputs)[:,None,:] for layer in self.featurizer], axis=1)
    #     data = tf.transpose(feats, [1, 2, 0])
    #     attn_out = self.attention([data, data])
    #     attn_out_T = tf.transpose(attn_out, [2, 0, 1])
    #     outputs = self.ffn(attn_out_T)
    #     return outputs
    

def create_ds(xx, yy, batch_size):

    output_signature=((tf.TensorSpec(shape=(None, None), dtype=tf.float32)),
                       tf.TensorSpec(shape=(None, None, None), dtype=tf.float32))
                       
    gn = create_generator(xx, yy, batch_size)
    ds = tf.data.Dataset.from_generator(gn, output_signature=output_signature)
    ds = setup_autoshard(ds)

    return ds


def train():

    batch_size = 18211 #16
    epochs = 300

    inputs, outputs, scalers = load_data()

    ds_train = create_ds(inputs, outputs, batch_size)
    ds_valid = None

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'model', 'weights_{epoch:03d}'),
            #save_best_only=True,
            save_weights_only=True,
            monitor='loss', #val_loss
            verbose=1)
    ]
    
    loss = mrrmse_loss_T # tf.keras.losses.MeanSquaredError()  tf.keras.losses.MeanAbsoluteError()  mrrmse_loss2  loss_abs
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model = Model(n_inputs=inputs.shape[1])               
    model.compile(optimizer=optimizer, loss=loss)
    #model.load_weights(os.path.join(MODEL_DIR, 'model', 'weights_150'))
    
    model.fit(ds_train, validation_data=ds_valid, epochs=epochs, callbacks=callbacks)
    #model.save_weights(os.path.join(folder, 'model_proba', 'weights'))    
    model.summary() 


def unpivot_predictions(preds):    
    xx = [[i,j] for i in range(6) for j in range(144)]
    yy = [preds[:,i,:].T for i in range(6)]       
    xx = np.array(xx, dtype='int32')
    yy = np.concatenate(yy, axis=0) 
    return xx, yy


def evaluate():

    train_df = load_train_df()
    submission_df = pd.read_csv( os.path.join(PATH_DATA, 'submission_0.528_avg.csv') )

    id_map_df = pd.read_csv( os.path.join(PATH_DATA, 'input', 'id_map.csv')  )
    id_map_df[['cell_type', 'sm_name']] = np.load( os.path.join(PATH_DATA, 'data', 'x_test_ordinal.npy') )

    inputs, outputs, scalers = load_data() 

    model = Model(n_inputs=inputs.shape[1]) 

    epochs = list(range(296,301)) # list(range(146,151)) # list(range(130,135)) #
    y_pred = 0
    for epoch in epochs:
        filename = os.path.join(MODEL_DIR, 'model', f'weights_{epoch:03d}')
        model.load_weights(filename)
        y_pred += model.predict(inputs, batch_size=18211) 
    y_pred = y_pred / len(epochs)

    for i in range(6):
        y_pred[:,i,:] = scalers['targets'].inverse_transform(y_pred[:,i,:])

    xx, yy = unpivot_predictions(y_pred) 
    pred_df = pd.DataFrame(np.concatenate([xx,yy], axis=1), columns=train_df.columns)
     
    pred_df_train = train_df[['cell_type', 'sm_name']].merge(pred_df, on=['cell_type', 'sm_name'], how='left')
    score = mrrmse(train_df.loc[:,'A1BG':].values, pred_df_train.loc[:,'A1BG':].values)
    print('train score:', score)    
         
    test_df_pred = id_map_df.merge(pred_df, on=['cell_type', 'sm_name'], how='left')
    score = mrrmse(submission_df.loc[:,'A1BG':].values, test_df_pred.loc[:,'A1BG':].values)
    print('test score:', score)
    
    filename = os.path.join(MODEL_DIR, 'submission.csv')
    test_df_pred.drop(columns=['cell_type', 'sm_name']).to_csv(filename, index=False)

    if False:
        df1 = pd.DataFrame(train_df)
        df1.loc[:,'A1BG':] = train_df.loc[:,'A1BG':].values - pred_df_train.loc[:,'A1BG':].values        

        df2 = pd.DataFrame(test_df_pred.drop(columns=['id']))
        df2.loc[:,'A1BG':] = submission_df.loc[:,'A1BG':].values - test_df_pred.loc[:,'A1BG':].values

        pd.concat([df1, df2], axis=0).reset_index(drop=True).to_parquet(os.path.join(MODEL_DIR, 'residuals_train+test.parquet'))


def find_best_epoch():
    
    submission_df = pd.read_csv( os.path.join(PATH_DATA, 'submission_0.532_public_blend.csv') )

    id_map_df = pd.read_csv( os.path.join(PATH_DATA, 'input', 'id_map.csv')  )
    id_map_df[['cell_type', 'sm_name']] = np.load( os.path.join(PATH_DATA, 'data', 'x_test_ordinal.npy') )
    y_true = submission_df.loc[:,'A1BG':].values

    inputs, outputs, scalers = load_data() 

    model = Model(n_inputs=inputs.shape[1]) 

    best_score = float('inf')
    best_epoch = None
    
    for epoch in range(1,151):
        filename = os.path.join(MODEL_DIR, 'model', f'weights_{epoch:03d}')
        model.load_weights(filename)

        y_pred = model.predict(inputs, batch_size=18211) 

        for i in range(6):
            y_pred[:,i,:] = scalers['targets'].inverse_transform(y_pred[:,i,:])

        xx, yy = unpivot_predictions(y_pred) 
        columns = ['cell_type', 'sm_name'] + submission_df.loc[:,'A1BG':].columns.tolist()
        pred_df = pd.DataFrame(np.concatenate([xx,yy], axis=1), columns=columns)        
         
        test_df_pred = id_map_df.merge(pred_df, on=['cell_type', 'sm_name'], how='left')
        y_pred = test_df_pred.loc[:,'A1BG':].values
        score = mrrmse(y_true, y_pred)
        if score < best_score:
            best_score = score
            best_epoch = epoch

        print(f'Epoch: {epoch}, score: {score:.4f}')
    
    print(f'Best epoch: {best_epoch}, best score: {best_score:.4f}')


if __name__ == "__main__":
    train()
    evaluate()