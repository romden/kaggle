import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

print('TF version:', tf.__version__)
print('TFP version:', tfp.__version__)


def prior(kernel_size, biase_size, dtype=None):
    n = kernel_size + biase_size
    #return lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), reinterpreted_batch_ndims=1)
    # analogous to the above
    prior_model = tf.keras.Sequential([
        tfpl.DistributionLambda(
            lambda t: tfd.Independent(tfd.Normal(loc=tf.zeros(n, dtype=dtype), scale=1), reinterpreted_batch_ndims=1)
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
            tfd.Independent(
                tfd.Normal(loc=t[..., :n], scale=1e-5 + 0.01*tf.nn.softplus(c + t[..., n:])), reinterpreted_batch_ndims=1)
            )
    ])
    return posterior_model


def build_ffn(n_features, n_targets, n_units, data_size):
    n_units = 128
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),

        tfpl.DenseVariational(units=n_units,
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True,
                              activation='relu'),        

        tfpl.DenseVariational(units=n_units,
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True,
                              activation='relu'),
        
        tfpl.DenseVariational(units=tfpl.IndependentNormal.params_size(n_targets),
                              make_prior_fn=prior,
                              make_posterior_fn=posterior,
                              kl_weight=1/data_size,
                              kl_use_exact=True),
        tfpl.IndependentNormal(n_targets),

        #tfpl.DenseVariational(units=tfpl.MultivariateNormalTriL.params_size(n_targets),
        #                      make_prior_fn=prior,
        #                      make_posterior_fn=posterior,
        #                      kl_weight=1/data_size,
        #                      kl_use_exact=True),                              
        #tfpl.MultivariateNormalTriL(event_size=n_targets),   
    ], name='predictor')
    return model


# =======================================
# VAE
# =======================================
def get_prior(num_modes, latent_dim):
    """
    This function should create an instance of a MixtureSameFamily distribution 
    according to the above specification. 
    The function takes the num_modes and latent_dim as arguments, which should 
    be used to define the distribution.
    Your function should then return the distribution instance.
    """
    prior = tfd.MixtureSameFamily(mixture_distribution = tfd.Categorical(probs=tf.divide(tf.ones([num_modes,]), num_modes)),
                                 components_distribution = tfd.MultivariateNormalDiag(loc = tf.Variable(tf.random.normal([num_modes, latent_dim]), trainable=True, dtype=tf.float32),
                                                                                      scale_diag = tfp.util.TransformedVariable(initial_value = tf.ones([num_modes, latent_dim]), 
                                                                                                                                bijector = tfb.Softplus()))
    )
    
    return prior

def get_kl_regularizer(prior_distribution):
    """
    This function should create an instance of the KLDivergenceRegularizer 
    according to the above specification. 
    The function takes the prior_distribution, which should be used to define 
    the distribution.
    Your function should then return the KLDivergenceRegularizer instance.
    """
    return tfpl.KLDivergenceRegularizer(distribution_b=prior_distribution,  
                                        use_exact_kl=False, 
                                        weight=1.0, 
                                        test_points_fn=lambda q: q.sample(3),
                                        test_points_reduce_axis = (0,1))

def create_encoder(n_features, latent_dim, units, kl_regularizer):

    # compute hidden representation of inputs
    encoder = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),

        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(units[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(units=tfpl.MultivariateNormalTriL.params_size(latent_dim)),
        tfpl.MultivariateNormalTriL(activity_regularizer=kl_regularizer, event_size=latent_dim),        
    ], name='encoder')
    return encoder


def create_decoder(n_features, latent_dim, units):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        
        tf.keras.layers.Dense(units[1], activation='relu'),
        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(n_features)),
        tfpl.IndependentNormal(n_features),        
    ], name='decoder')
    return model


def build_vae(n_features, latent_dim=64, units=[256, 128]):

    prior = get_prior(num_modes=2, latent_dim=64)
    kl_regularizer = get_kl_regularizer(prior)

    encoder = create_encoder(n_features, latent_dim, units, kl_regularizer=kl_regularizer)

    decoder = create_decoder(n_features, latent_dim, units)
    
    vae = tf.keras.models.Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))

    return vae, encoder, decoder


# =======================================
# Bayesian NN + VAE
# =======================================
def build_ffn_vae(n_features, n_targets, latent_dim, units, data_size):
    
    prior = get_prior(num_modes=2, latent_dim=64)
    kl_regularizer = get_kl_regularizer(prior)

    encoder = create_encoder(n_features, latent_dim, units, kl_regularizer=kl_regularizer)

    decoder = create_decoder(n_features, latent_dim, units)

    predictor = build_ffn(latent_dim, n_targets, units[0], data_size) # units[0] latent_dim

    outputs = {'reconstruction': decoder(encoder.outputs), 'prediction': predictor(encoder.outputs)}    

    model = tf.keras.models.Model(inputs=encoder.inputs, outputs=outputs)

    return model


def build_ffn_vae_cond(n_features, n_targets, latent_dim, units, data_size):
    
    prior = get_prior(num_modes=2, latent_dim=64)    
    kl_regularizer = get_kl_regularizer(prior)

    encoder = create_encoder(n_features+n_targets, latent_dim, units, kl_regularizer=kl_regularizer)

    decoder = create_decoder(n_features, latent_dim, units)

    predictor = build_ffn(latent_dim, n_targets, units[0], data_size) #units[0] latent_dim

    x = tf.keras.layers.Input(shape=(n_features,))

    y_hat = predictor( prior.sample(tf.shape(x)[0]) )
    
    xx = tf.concat([x, y_hat], axis=1)

    z = encoder(xx)

    outputs = {'reconstruction': decoder(z), 'prediction': predictor(z)}    

    model = tf.keras.models.Model(inputs=x, outputs=outputs)

    return model