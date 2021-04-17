from collections import namedtuple
import tensorflow as tf
from aggregators import MaxPoolingAggregator


class BilstmModel(tf.keras.Model):
    def __init__(self, dims0, dims1,batch_size, **kwargs):

        super(BilstmModel, self).__init__(**kwargs)
     
        self.output_dim_bottom = 200
        self.output_dim_top = 200
        self.number_of_words = dims0
        self.dim_embed = dims1
        self.batch_size= batch_size
        self.masker = tf.keras.layers.Masking(mask_value = 0.,input_shape = (self.number_of_words,self.dim_embed))

        # Word Embedding Layer
    def build(self, inputs):
        print(self.number_of_words)
        self.w = tf.Variable(tf.random.uniform([1,2*self.output_dim_top]), name = "attention")
       # self.embedding =tf.keras.layers.Embedding(input_dim=self.number_of_words, output_dim=16, mask_zero=True)
        
        self.forward_layer = tf.keras.layers.LSTM(self.output_dim_bottom,dropout = 0.5, return_sequences=True,recurrent_activation = 'sigmoid', input_shape=(self.number_of_words, self.dim_embed))
       # self.backward_layer = tf.keras.layers.LSTM(self.output_dim_top,dropout = 0.2, return_sequences=True, go_backwards=True, input_shape = (self.number_of_words, self.dim_embed))
        self.bilstm = tf.keras.layers.Bidirectional(self.forward_layer, merge_mode = "concat")


    def call(self, inputs, training=None, mask=None):
        w1 = tf.tile(tf.expand_dims(self.w, axis = 0), [inputs.shape[0], 1,1 ])
        #rajouter le batch size a la place de 100 
        #masque = self.embedding.compute_mask(inputs)
        #print("masque dim", masque.shape)
        #print("inputs dim", inputs.shape)
        # print(self.masker(inputs))
        inputs2 = self.masker(inputs)
        sortie1 = self.bilstm(inputs2)
        #print(sortie1.shape, "sum shape")
       
        #print(sortie1.shape)
    
        b = tf.matmul(w1,tf.transpose(sortie1,[0,2,1]))
        #print("b shape", b.shape)
        alpha = tf.nn.softmax(b, axis =2)
        #print("alpha shape", alpha.shape)
        sortie2 = tf.matmul(alpha, sortie1)
        sortie3 = tf.squeeze(sortie2, axis = 1)
        #print(sortie3.shape)
        return sortie3




# SAGEInfo is a namedtuple that specifies the parameters
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SampleAndAggregate(tf.keras.Model):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, features, adj, degrees,
            layer_infos,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features. 
                        NOTE: Pass a None object to train in featureless mode (identity features for nodes)!
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - identity_dim: Set to positive int to use identity features (slow and cannot generalize, but better accuracy)
        '''
        super(SampleAndAggregate, self).__init__(**kwargs)

        self.aggregator_cls = MaxPoolingAggregator
        self.model_size = 512

        #INUTILE???????????
        #self.adj_info = adj



        '''
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
           '''

        self.adj = adj
        self.features = features
        self.degrees = degrees
        self.dims = [len(self.features[0])]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
    
        self.batch_size = 100
        self.layer_infos = layer_infos
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.aggregators = None

    def sample(self, inputs, layer_infos):
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        

        batch_size = inputs.shape[0]
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
      
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler

            node = sampler((samples[k], layer_infos[t].num_samples))
            
            samples.append(tf.reshape(node, [support_size * batch_size,]))
            support_sizes.append(support_size)
        return samples, support_sizes


    def aggregate(self, samples, input_features, dims, num_samples, support_sizes,
             name=None, concat=False, model_size="small"):
        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """


        batch_size = self.batch_size

        # length: number of layers + 1

        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        #hidden = features de tous les nodes qui sont à une certaine distance
        #en pratique on prendra input_features = [features] donc tous les titres des noeuds.
        new_agg = self.aggregators is None
        if new_agg:
            self.aggregators = []
        #len(num_samples) est le nombre de layers K
        
        for layer in range(len(num_samples)):
            if new_agg:
                dim_mult = 2 if (layer!=0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    #si on est à la fin, l'activation est l'identité
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=0.2)
                else:
                    #sinon l'activation est le relu (de base)
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act = tf.keras.activations.relu,
                            dropout=0.2)
                self.aggregators.append(aggregator)
            else:
                aggregator = self.aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if (layer!=0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]

                h = aggregator((hidden[hop],tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden


        return hidden[0]
    #on prend 0 car on s'intéresse juste aux embeddings des noeuds de base

    def call(self, inputs, training=None, mask=None):

 
        inputs1 = inputs["batch1"]
        self.batch_size = inputs1.shape[0]
        #inputs1: noeud 1
        inputs2 = inputs["batch2"]

        #inputs2: noeud 2
           
        # perform "convolution"
        samples1, support_sizes1 = self.sample(inputs1, self.layer_infos)

        samples2, support_sizes2 = self.sample(inputs2, self.layer_infos)

        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        outputs1 = self.aggregate(samples1, [self.features], self.dims, num_samples, support_sizes1)
        outputs2 = self.aggregate(samples2, [self.features], self.dims, num_samples, support_sizes2)


        outputs1 = tf.nn.l2_normalize(outputs1, 1)
        outputs2 = tf.nn.l2_normalize(outputs2, 1)
        return outputs1, outputs2

"""
    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._accuracy()
        self.loss = self.loss / tf.cast(self.batch_size, tf.float32)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)

    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs) 
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]

        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)

"""
