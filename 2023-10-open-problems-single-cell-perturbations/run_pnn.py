import sys, os
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

tf.get_logger().setLevel('ERROR')
tf.random.set_seed(4243) 
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from run_nn2 import load_data, unpivot_predictions, load_train_df
from shared import PATH_DATA, set_seed, mrrmse
set_seed(4243)


MODEL_DIR = os.path.join(PATH_DATA, 'pivot_1_pnn')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def nll(y_true, dist):
    loss = -tf.math.reduce_mean(dist.log_prob(y_true))
    return loss 


def prior(kernel_size, biase_size, dtype=None):
    n = kernel_size + biase_size
    return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), reinterpreted_batch_ndims=1)


def posterior(kernel_size, biase_size, dtype=None):
    n = kernel_size + biase_size
    c = 0.541324854612918 # np.log(np.expm1(1.))
    posterior_model = tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.DistributionLambda(lambda t: 
            tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)
            )
    ])
    return posterior_model


def build_predictor(n_inputs=256, n_outputs=144, n_units=256, data_size=18211):

    model = tf.keras.Sequential([        
        tfpl.DenseVariational(units=16,
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True,
                              activation=tf.keras.layers.LeakyReLU()),
         
        # tfpl.DenseVariational(units=tfpl.MultivariateNormalTriL.params_size(n_outputs),
        #                       make_prior_fn=prior,
        #                       make_posterior_fn=posterior,
        #                       kl_weight=1/data_size,
        #                       kl_use_exact=True),    

        tf.keras.layers.Dense(tfpl.MultivariateNormalTriL.params_size(n_outputs)),

        tfpl.MultivariateNormalTriL(n_outputs),
        #tfpl.IndependentNormal(n_outputs),
    ])
    return model


def build_featurizer(n_inputs=128, n_outputs=256, n_units=256, data_size=18211):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),       

        tfpl.DenseVariational(units=n_outputs,
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True),        
    ])   
    return model


class Model(tf.keras.Model):

    def __init__(self, n_inputs=128, n_outputs=144, units=256):
        super().__init__()         
        self.featurizer = [build_featurizer() for _ in range(6)]   
        self.predictor = build_predictor()  
        self.attention = tf.keras.layers.Attention(use_scale=True)

    def call(self, inputs):     
        feats = tf.concat([layer(inputs)[:,None,:] for layer in self.featurizer], axis=1)
        data = tf.transpose(feats, [2, 1, 0])
        attn_out = self.attention([data, data])
        attn_out_T = tf.transpose(attn_out, [2, 1, 0])
        dist = self.predictor(attn_out_T)

        # loc = outputs[:,:,:144]
        # scale = tf.nn.softplus(outputs[:,:,144:])
        # dist = tfd.Independent(tfd.Normal(loc=loc, scale=scale))

        #dist = tfpl.MultivariateNormalTriL(144)(outputs)

        return dist  


def train():

    inputs, outputs, scalers = load_data()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'model', 'weights_{epoch:03d}'),
            #save_best_only=True,
            save_weights_only=True,
            monitor='loss', #val_loss
            verbose=1)
    ]
    
    model = Model(n_inputs=inputs.shape[1])               
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=nll)
    #model.load_weights(os.path.join(MODEL_DIR, 'model', 'weights_150'))
    
    model.fit(inputs, outputs, callbacks=callbacks, epochs=100,  batch_size=18211)
    #model.save_weights(os.path.join(folder, 'model_proba', 'weights'))    
    model.summary() 


def evaluate():

    train_df = load_train_df()
    submission_df = pd.read_csv( os.path.join(PATH_DATA, 'submission_0.528_avg.csv') )

    id_map_df = pd.read_csv( os.path.join(PATH_DATA, 'input', 'id_map.csv')  )
    id_map_df[['cell_type', 'sm_name']] = np.load( os.path.join(PATH_DATA, 'data', 'x_test_ordinal.npy') )

    inputs, outputs, scalers = load_data() 

    model = Model(n_inputs=inputs.shape[1])
    model.load_weights( os.path.join(MODEL_DIR, 'model', f'weights_{150}') )

    n_samples = 1#00
    y_pred = 0
    for _ in range(n_samples):        
        y_pred += model.predict(inputs, batch_size=18211) 
    y_pred /= n_samples

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


if __name__ == "__main__": 
    train()
    evaluate()