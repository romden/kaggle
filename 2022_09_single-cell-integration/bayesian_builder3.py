import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

import tensorflow as tf

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


def build_ffn(n_features, n_targets, n_units, data_size, name='predictor'):
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

        tfpl.IndependentNormal(n_targets)
    ], name=name)
    return model


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


def create_encoder(n_features, latent_dim, units, kl_regularizer=None, name='encoder'):

    kl_regularizer = get_kl_regularizer( get_prior(num_modes=2, latent_dim=latent_dim) )

    # compute hidden representation of inputs
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),

        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(units[1], activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(units=tfpl.MultivariateNormalTriL.params_size(latent_dim)),
        tfpl.MultivariateNormalTriL(activity_regularizer=kl_regularizer, event_size=latent_dim),
    ], name=name)
    return model


def create_decoder(n_features, latent_dim, units, name='decoder'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(latent_dim,)),
        
        tf.keras.layers.Dense(units[1], activation='relu'),
        tf.keras.layers.Dense(units[0], activation='relu'),
        tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(n_features)),
        tfpl.IndependentNormal(n_features),        
    ], name=name)
    return model


# =======================================
class EmbeddingMetadata(tf.keras.layers.Layer):
    def __init__(self, keys):
        super().__init__()
        self.keys = keys
        self.embeddings = {
            'day': tf.keras.layers.Embedding(2, 2),
            'donor': tf.keras.layers.Embedding(4, 2),
            'cell_type': tf.keras.layers.Embedding(8, 2),
            'technology': tf.keras.layers.Embedding(2, 2),
        }
    def get_n_outputs():
        return sum([self.embeddings[key].output_dim for key in self.keys])    
    def call(self, inputs):
        lst = [self.embeddings[key](inputs[:,i]) for i, key in enumerate(self.keys)]
        return tf.concat(lst, axis=1)


class MainPrior(tf.keras.layers.Layer):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.categorical = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dense(4, activation='softmax'),
            ])        
        self.components = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(latent_dim)*8, activation='relu'),
            tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(latent_dim)*4, activation='linear'),
        ])
        self.normal = tfpl.IndependentNormal(latent_dim)
        
    def call(self, inputs):
        cat = tfd.Categorical(probs=self.categorical(inputs))
        components = [self.normal(t) for t in tf.split(self.components(inputs), num_or_size_splits=4, axis=1)]
        return tfd.Mixture(cat=cat, components=components)


class CellModel(tf.keras.Model):

    def __init__(self, n_features, n_targets, latent_dim, units, data_size):
        super().__init__()
        
        self.tkeys = ['cite', 'multi']
        self.mkeys = ['day', 'donor', 'cell_type', 'technology'] 
        
        #self.embedmeta = EmbeddingMetadata(self.mkeys)
        self.embedmeta = {key: EmbeddingMetadata(self.mkeys) for key in self.tkeys}
        #self.prior = MainPrior(8, latent_dim)        
        self.sample = tf.keras.layers.Lambda(lambda t: t.sample())

        self.encoder = {}#create_encoder(512+8, latent_dim, units)
        self.decoder = {}#create_decoder(512, latent_dim+8, units)
        self.predictor = {}
        for key in self.tkeys:
            self.encoder[key] = create_encoder(n_features[key]+6, latent_dim, units, name=f'encoder_{key}')
            self.decoder[key] = create_decoder(n_features[key], latent_dim, units, name=f'decoder_{key}')
            self.predictor[key] = build_ffn(latent_dim, n_targets[key], units[0], data_size[key], name=f'predictor_{key}')
         
        def divergence_fn(inputs):
            q, p, z = inputs
            return tf.math.reduce_mean(q.log_prob(z) - p.log_prob(z))
        self.divergence = tf.keras.layers.Lambda(divergence_fn)

        def nll_fn(inputs):
            y_true, y_pred = inputs
            return -tf.math.reduce_mean(y_pred.log_prob(y_true))        
        self.reconstruction = tf.keras.layers.Lambda(nll_fn)


    def call(self, inputs):
        
        features = {key: inputs[key][:,:-4] for key in self.tkeys}
        metadata = {key: inputs[key][:,-4:] for key in self.tkeys}
        embeddings = {key: self.embedmeta[key](metadata[key]) for key in self.tkeys}

        #latent_p = {key: self.prior(embeddings[key]) for key in self.tkeys} # distribution
        #latent_q = {key: self.encoder[key](features[key]) for key in self.tkeys} # distribution
        latent_q = {key: self.encoder[key](tf.concat([features[key], embeddings[key][:,:-2]], axis=1)) for key in self.tkeys} # distribution
        latent_z = {key: self.sample(latent_q[key]) for key in self.tkeys} # tensor 

        reconstruction = {key: self.decoder[key](latent_z[key]) for key in self.tkeys} # distribution
        #reconstruction = {key: self.decoder(tf.concat([latent_z[key], embeddings[key]], axis=1)) for key in self.tkeys} # distribution    
        
        outputs = {key: self.predictor[key](latent_z[key]) for key in self.tkeys} # distribution
        #outputs = {key: self.predictor[key](tf.concat([latent_z[key], embeddings[key][:,:-2]], axis=1)) for key in self.tkeys} # distribution
        
        for key in self.tkeys:
            self.add_loss(self.reconstruction([features[key], reconstruction[key]]))
            #self.add_loss(self.divergence([latent_q[key], latent_p[key], latent_z[key]]))

        return outputs