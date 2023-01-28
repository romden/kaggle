import tensorflow as tf
from tensorflow.keras import backend as K


def nll(y_true, y_pred):
    return -tf.math.reduce_mean(y_pred.log_prob(y_true))


def loss_single_model(y_true, y_pred):
    targets = y_true[:,:-1]
    weights = y_true[:,-1]

    sum_weights = tf.math.reduce_sum(weights)
    if sum_weights > 0:
        weights /= sum_weights

    return -tf.math.reduce_sum(y_pred.log_prob(targets) * weights)


class LossReconstractionPrediction:

    def __init__(self, n_features, n_targets):
        self.n_features = n_features
        self.n_targets = n_targets

    def parse_inputs(self, y_true):
        features = y_true[:, :self.n_features]
        targets = y_true[:, self.n_features:-1]
        weights = y_true[:, -1] 

        sum_weights = tf.math.reduce_sum(weights)
        if sum_weights > 0:
            weights = weights / sum_weights

        return features, targets, weights

    def reconstruction(self, y_true, y_pred):
        features, _, _ = self.parse_inputs(y_true)
        loss = -tf.math.reduce_mean(y_pred.log_prob(features))
        return loss

    def prediction(self, y_true, y_pred):
        _, targets, weights = self.parse_inputs(y_true)
        loss = -tf.math.reduce_sum(y_pred.log_prob(targets) * weights)
        return loss


class LossContrastivePrediction:

    def __init__(self):
        self.omega = 10

    def parse_inputs(self, y_true):
        targets = y_true[:,:-1]
        weights = y_true[:,-1]
        sum_weights = tf.math.reduce_sum(weights)
        if sum_weights > 0:
            weights = weights / sum_weights
        return targets, weights

    def contrastive(self, y_true, y_pred):
        return -tf.math.reduce_mean(tf.math.log(tf.clip_by_value(y_pred, 1e-6, 1)))

    def prediction(self, y_true, y_pred):
        targets, weights = self.parse_inputs(y_true)
        mse = tf.math.reduce_mean(tf.math.pow(targets - y_pred, 2), axis=1)
        loss = tf.math.reduce_sum(mse * weights) * self.omega
        return loss


def negative_correlation_loss(y_true, y_pred):
    """Negative correlation loss function for Keras
    
    Precondition:
    y_true.mean(axis=1) == 0
    y_true.std(axis=1) == 1
    
    Returns:
    -1 = perfect positive correlation
    1 = totally negative correlation
    """
    my = K.mean(tf.convert_to_tensor(y_pred), axis=1)
    my = tf.tile(tf.expand_dims(my, axis=1), (1, y_true.shape[1]))
    ym = y_pred - my
    r_num = K.sum(tf.multiply(y_true, ym), axis=1)
    r_den = tf.sqrt(K.sum(K.square(ym), axis=1) * float(y_true.shape[-1]))
    r = tf.reduce_mean(r_num / r_den)
    return - r