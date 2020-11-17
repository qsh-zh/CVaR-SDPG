import tensorflow as tf
import numpy as np
def CVaR_sample(data,alpha,isOptimi=False):
    # see https://arxiv.org/pdf/1901.00997.pdf 3.1
    # higher alpha when optimi, will be optimier
    if isOptimi:
        sort_dir = 'DESCENDING'
    else:
        sort_dir = 'ASCENDING'
    
    data = tf.sort(data,axis=1,direction=sort_dir)
    data_shape = tf.shape(data)
    batch_size = data_shape[0]
    data_size = tf.cast(data_shape[1],tf.float32)
    alpha = tf.cast(1-alpha,tf.float32)
    index = tf.cast(data_size * alpha,tf.int32)
    CVaR_data = data[:,:index]
    return tf.reduce_mean(CVaR_data,1)

def Wang_sample(data,eta):
    # see IQN paper, Section 4
    # just add eta to all random var
    data_distorted = data + eta
    return tf.reduce_mean(data_distorted,1)