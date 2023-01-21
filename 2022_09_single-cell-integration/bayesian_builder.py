import tensorflow as tf
import tensorflow_probability as tfp
tfpd = tfp.distributions
tfpl = tfp.layers

print('TF version:', tf.__version__)
print('TFP version:', tfp.__version__)


def prior(kernel_size, biase_size, dtype=None):
    n = kernel_size + biase_size
    #return lambda t: tfpd.Independent(tfpd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), reinterpreted_batch_ndims=1)
    # analogous to the above
    prior_model = tf.keras.Sequential([
        tfpl.DistributionLambda(
            lambda t: tfpd.Independent(tfpd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), reinterpreted_batch_ndims=1)
        )
    ])
    return prior_model


#https://stackoverflow.com/questions/66418959/not-able-to-get-reasonable-results-from-densevariational
def posterior(kernel_size, biase_size, dtype=None):
    n = kernel_size + biase_size
    c = 0.541324854612918#np.log(np.expm1(1.))
    posterior_model = tf.keras.Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n), dtype=dtype),
        tfpl.DistributionLambda(lambda t: 
            tfpd.Independent(
                tfpd.Normal(loc=t[..., :n], scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)
            )
    ])
    return posterior_model


def build_cite_ffn(n_inputs, n_outputs, n_units, data_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        
        tfpl.DenseVariational(units=n_units,
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True,
                              activation='relu'),

        tf.keras.layers.BatchNormalization(),
         
        tfpl.DenseVariational(units=n_units,
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True,
                              activation='relu'),

        tf.keras.layers.BatchNormalization(),

        tfpl.DenseVariational(units=tfpl.IndependentNormal.params_size(n_outputs),
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True),
                              
        tf.keras.layers.BatchNormalization(),

        tfpl.IndependentNormal(n_outputs)
    ])
    return model