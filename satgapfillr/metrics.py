import tensorflow as tf

def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))
