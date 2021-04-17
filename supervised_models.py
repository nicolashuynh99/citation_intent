import layers as layers
import models as models
import tensorflow as tf
from aggregators import MaxPoolingAggregator


class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, input,
             features, labels, adj, degrees,
            layer_infos, concat=True, lr = 0.01,
            model_size="small", sigmoid_loss=True, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.SampleAndAggregate.__init__(self, **kwargs)


        self.aggregator_cls = MaxPoolingAggregator()


        # get info from placeholders...
        #self.inputs1 = placeholders["batch"]

        self.model_size = model_size


        #INUTILE????????
        self.adj_info = adj
        self.labels = labels

        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.num_classes = 4
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.weight_decay = .01
        self.batch_size = 2
        self.layer_infos = layer_infos

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



    def call(self, batch):
        samples1, support_sizes1 = self.sample(batch, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        return self.outputs1


"""
    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.edge_preds,
                    labels= .....))


        tf.summary.scalar('loss', self.loss)
"""
    def predict(self):


        return tf.nn.softmax(self.edge_preds)
