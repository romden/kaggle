import tensorflow as tf


def binary_cross_entropy(class_weights=(1, 1), from_logits=False, eps=1e-07):
    """The outer function."""
    min_ = eps
    max_ = 1.0 - eps

    @tf.function
    def wrapper(y_true, y_pred, sample_weights=None):
        """The inner function."""

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_prob = tf.sigmoid(y_pred) if from_logits else y_pred

        y_prob = tf.keras.backend.clip(y_prob, min_, max_)            

        loss = class_weights[0] * (1 - y_true) * tf.math.log(1 - y_prob) + class_weights[1] * y_true * tf.math.log(y_prob)

        if sample_weights is not None:
            loss *= sample_weights

        return -tf.reduce_mean(loss)

    return wrapper


def focal_bce(class_weights=(1, 1), from_logits=False, eps=1e-07, gamma=2.0):
    """The outer function."""
    min_ = eps
    max_ = 1.0 - eps

    @tf.function
    def wrapper(y_true, y_pred):
        """The inner function."""

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        y_prob = tf.sigmoid(y_pred) if from_logits else y_pred        

        p_t = (y_true * y_prob) + (1 - y_true) * (1 - y_prob)

        y_prob = tf.keras.backend.clip(y_prob, min_, max_)

        bce = class_weights[0] * (1 - y_true) * tf.math.log(1 - y_prob) + class_weights[1] * y_true * tf.math.log(y_prob)

        loss = tf.pow(1.0 - p_t, gamma) * bce

        return -tf.reduce_mean(loss)

    return wrapper


def contrastive_loss(y_true, y_pred):
    """Inspired by: Contrastive loss and Triplet loss"""

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    margin = 0.1

    loss = (1 - y_true) * tf.math.maximum(margin + y_pred, 0) + y_true * tf.math.maximum(margin - y_pred, 0)

    return tf.reduce_mean(loss)


def semi_supervised_loss():
    """Loss for semi-supervised training"""

    bce = binary_cross_entropy()

    @tf.function
    def wrapper(y_true, y_pred):
        
        y_true_bce, y_true_sim = y_true[:,0], y_true[:,-1] 
        y_pred_bce, y_pred_sim = y_pred[:,0], y_pred[:,-1]

        sample_weights = tf.where(y_true_bce*(1-y_true_bce) > 0, 0., 1.)
        w_bce = 3
        w_sim = 1
        
        loss_bce = bce(y_true_bce, y_pred_bce, sample_weights)
        #loss_sim = -tf.reduce_mean(tf.math.log(y_pred_sim)) # contrastive SAINT paper
        loss_sim = bce(y_true_sim, y_pred_sim) # cossine similarity
        loss = (w_bce * loss_bce + w_sim * loss_sim) / (w_bce + w_sim)

        return loss

    return wrapper