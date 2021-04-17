import layers as layers
import models as models
import tensorflow as tf
from tensorflow.keras.models import Model
from aggregators import MaxPoolingAggregator, NodeAggregator
from models import BilstmModel, SampleAndAggregate
from stellargraph.mapper import GraphSAGENodeGenerator

from tensorflow import keras

class EdgeClassifier(Model):
    """Implementation of supervised GraphSAGE."""

    def __init__(self,
                 features, adj, degrees,
                 layer_infos,batch_size,
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

        super(EdgeClassifier, self).__init__(**kwargs)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.features = features
        self.adj = adj
        self.degrees = degrees
        self.layer_infos = layer_infos
        self.outputdim = layer_infos[-1].output_dim*2
        self.batch_size = batch_size
        self.node_embedder = keras.models.load_model("./modeleKeras")
        #self.G = G

        self.batch_size = 100
        self.num_samples = [10,5]
        #self.embeddings = embeddings
    def build(self, inputs):
        self.model_bilstm = BilstmModel(509, 768, self.batch_size)
        self.model_graphsage = SampleAndAggregate(features=self.features, adj=self.adj, degrees=self.degrees, layer_infos=self.layer_infos)
        self.dense1 = tf.keras.layers.Dense(200, activation = "relu")
        self.dropout = tf.keras.layers.Dropout(0.5)
        # 100 est la hidden dimension
        self.dense2 = tf.keras.layers.Dense(3, activation="softmax")
        # 4 est le nombre de classes
        self.maxpool = NodeAggregator(input_dim=self.outputdim, output_dim=self.outputdim)

    def call(self, input, training=None, mask=None):
        #input est le batch de données
        #CHANGER LE SLICING

        embedding_node_1,embedding_node_2 = self.model_graphsage(input)
        
        #embedding_node_1, embedding_node_2 = self.node_embedder(input) changer l'input ici
        #node_gen1 = GraphSAGENodeGenerator(self.G,self.batch_size, self.num_samples).flow(names1)
        #node_gen2 = GraphSAGENodeGenerator(self.G, self.batch_size, self.num_samples).flow(names2)
        #embedding_node_1 = self.node_embedder.predict(node_gen1, workers = 4, verbose = 1)
        #embedding_node_2 = self.node_embedder.predict(node_gen2, workers = 4, verbose = 1)       
        #embedding_node_1 = self.embeddings[input["batch1"]] 
        #embedding_node_2 = self.embeddings[input["batch2"]]
                     
        maxNodes = self.maxpool((embedding_node_1,embedding_node_2))
        #maxNodes est l'embedding obtenu à partir du maxpool des deux noeuds de l'arête

        embedding_edge = self.model_bilstm(input["edge_embeddings"])
   
      
        #on pourrait réduire la dimension ici
       
        #embedding_edge = tf.reshape(embedding_edge, [embedding_edge.shape[0], embedding_edge.shape[1]*embedding_edge.shape[2]])
        #embedding_edge = tf.math.reduce_sum(embedding_edge, axis = 1)
        full_embeddings = tf.concat([embedding_edge, maxNodes], axis = 1)
        
        pred = self.dense2(self.dropout(self.dense1(full_embeddings)))
        #pred = self.dense2(self.dense1(embedding_edge))
        return pred


    def loss(self, predictions, labels):
        # Weight decay loss
        

        """
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
        """
        # classification loss

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=predictions,
                labels=labels))
        return self.loss

