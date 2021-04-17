

import os
import time
import sys
sys.path.append("./GraphSAGE/graphsage")

from edgeMinibatch import EdgeMinibatchIterator
import networkx as nx
import numpy as np
import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)

edgelist = [
    [1,2,  np.random.rand(10,500), 0],
    [3,4,  np.random.rand(10,500), 0],
    [4,5,  np.random.rand(10,500), 0],
    [5,6,  np.random.rand(10,500), 0]
]

G = nx.Graph()
for i in range(1,7):
    G.add_node(i, title = "test", test = False, val = False)

    G.add_node(10*i, title = "test", test = False, val = False)
    G.add_node(10*i+1, title = "test", test = False, val = False)
    G.add_node(10*i+2, title = "test", test = False, val = False)
    G.add_node(10*i+3, title = "test", test = False, val = False)

    G.add_node(100*i, title = "test", test = False, val = False)
    G.add_node(100*i+1, title = "test", test = False, val = False)
    G.add_node(100*i+2, title = "test", test = False, val = False)
    G.add_node(100*i+3, title = "test", test = False, val = False)


    G.add_edge(i,10*i)
    G.add_edge(i,10*i+1)
    G.add_edge(i,10*i+2)
    G.add_edge(i,10*i+3)

    G.add_edge(10*i,100*i)
    G.add_edge(10*i+1,100*i+1)
    G.add_edge(10*i+2,100*i+2)
    G.add_edge(10*i+3,100*i+3)

compteur = 0
dicoIdMap = {}
for node in G.nodes:
    dicoIdMap[node] = compteur
    compteur+=1


#features = train_data[1]
id_map = dicoIdMap
print(G.nodes[10]["test"])
minibatch = EdgeMinibatchIterator(G, edgelist,
                                  id_map,
                                  batch_size=2,
                                  max_degree=25)

print(minibatch.next_minibatch_feed_dict())
# adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
# adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

from neigh_samplers import UniformNeighborSampler
from models import SAGEInfo
from classifier import EdgeClassifier
print(minibatch.adj)
sampler = UniformNeighborSampler(minibatch.adj)
layer_infos = [SAGEInfo("node", sampler, 4, 128),
               SAGEInfo("node", sampler, 4, 256)]
features = []
for node in G.nodes:
    features.append(np.random.rand(5))
features = tf.Variable(features)
labels = [0 for i in range(len(G.nodes))]

modeleFinal = EdgeClassifier(features= features, adj= minibatch.adj, degrees = minibatch.deg,layer_infos=layer_infos)

#print(modeleFinal(minibatch.next_minibatch_feed_dict()))

#print(modeleFinal(minibatch.next_minibatch_feed_dict()))

@tf.function
def normloss(vecA):
    return tf.norm(vecA)
def train_step(model, batch, labels):
  with tf.GradientTape() as tape:
    print(batch)
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    predictions = model(batch, training=True)
    print(predictions)
    loss = normloss(predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  print("LOSS:",loss)
  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

a = minibatch.next_minibatch_feed_dict()
for i in range(500):
    train_step(modeleFinal,a,0)