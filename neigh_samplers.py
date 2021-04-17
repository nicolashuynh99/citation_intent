from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer



"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info

    def call(self, inputs, training=None, mask=None):
        ids, num_samples = inputs


        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)

        adj_lists = tf.transpose(tf.random.shuffle(tf.transpose(adj_lists)))
        adj_lists = tf.slice(adj_lists, [0,0], [-1, num_samples])
        return adj_lists
