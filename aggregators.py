import tensorflow as tf



class NodeAggregator(tf.keras.layers.Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, act= "relu",
            dropout=0., **kwargs):
        super(NodeAggregator, self).__init__(**kwargs)

    
        if neigh_input_dim is None:
            self.neigh_input_dim = input_dim
        self.mlp_layers = []
        self.act = act
        self.top1 = tf.keras.layers.Dense(units = output_dim)
        self.top2 = tf.keras.layers.Dense(units = output_dim)
        self.hidden_dim = 100
        self.dim_reducer = tf.keras.layers.Dense(units = self.hidden_dim)
        #512 ou 1024
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training = None, mask = None):
        node1, node2 = inputs

        #vecteurs self_vecs = le noeud en lui même
        #vecteurs neigh_vecs = les représentations latentes des voisins


        dims = tf.shape(node1)
        reduced_node1 = self.dropout(self.dim_reducer(node1))
        #shape = (batch_size, dim_hidden)
        reduced_node2 = self.dropout(self.dim_reducer(node2))

        #on veut du shape = (batch_size, 2, dim_hidden)
        pooled_max = tf.stack([reduced_node1, reduced_node2], axis = 1)
        output = tf.reduce_max(pooled_max, axis=1)


        return output
        #return self.act(output)


class MaxPoolingAggregator(tf.keras.layers.Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None, act= "relu",
            dropout=0., **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

   
        if neigh_input_dim is None:
            self.neigh_input_dim = input_dim
        self.mlp_layers = []
        self.hidden_dim = 10
        self.act = act


        #512 ou 1024
        self.input_dim = input_dim
        self.output_dim = output_dim
    def build(self, inputs):
        self.single_layer = tf.keras.layers.Dense(units=self.hidden_dim, activation="relu", use_bias=True)
        self.top1 = tf.keras.layers.Dense(units=self.output_dim)
        self.top2 = tf.keras.layers.Dense(units=self.output_dim)
        self.dropout = tf.keras.layers.Dropout(0.2)
    def call(self, inputs, training = None, mask = None):
        self_vecs, neigh_vecs = inputs

        #vecteurs self_vecs = le noeud en lui même
        #vecteurs neigh_vecs = les représentations latentes des voisins

        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neighbors] x [hidden_dim]
        h_reshaped2 = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))


        h_reshaped = self.single_layer(h_reshaped2)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neigh_h = tf.reduce_max(neigh_h, axis=1)

        from_neighs = self.dropout(self.top1(neigh_h))

        from_self = self.dropout(self.top2(self_vecs))

        output = tf.concat([from_self, from_neighs], axis=1)

        return output
        #return self.act(output)
