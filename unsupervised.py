import networkx as nx
import pandas as pd
import numpy as np
import os
import random

import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler
from sklearn.model_selection import train_test_split

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score

from stellargraph import globalvar


from stellargraph import IndexedArray
from stellargraph import StellarGraph

from stellargraph.mapper import GraphSAGENodeGenerator


import os
import time
import sys
sys.path.append("./GraphSAGE/graphsage")
import pickle as pkl
from edgeMinibatch import EdgeMinibatchIterator
import networkx as nx
import json
import numpy as np
import tensorflow as tf
from neigh_samplers import UniformNeighborSampler
from models import SAGEInfo
from classifier import EdgeClassifier
#tf.config.experimental_run_functions_eagerly(True)
import pickle
import tensorflow_addons as tfa





edgelist = pkl.load(open("../list_full_scibert_embs_citations", "rb"))
test_edgelist = pkl.load(open("../test-list_scibert_embs_citations.pkl", "rb"))






G = nx.Graph()
G_test = nx.Graph()
G_train = nx.Graph()



dicoEmbT = {}
for i in range(1,12):
   dico1 = pickle.load(open("../correct-dic_title_embeddings_"+str(i), "rb"))
   dicoEmbT.update(dico1)
dico2 = pickle.load(open("../correct-dic_title_embeddings", "rb"))
dicoEmbT.update(dico2)




test_dicoEmbT = {}
for i in range(1,5):
   dico1 = pickle.load(open("../correct-test-dic_title_embeddings_"+str(i), "rb"))
   test_dicoEmbT.update(dico1)
dico2 = pickle.load(open("../correct-test-dic_title_embeddings", "rb"))
test_dicoEmbT.update(dico2)


print("LOADING DICO FINI")


with open("../listEdgesGrapheStep5.json") as fp:
    data = json.load(fp)
    liste = data["edges"]
    fp.close()


with open("../test-listEdgesGrapheStep5.json") as fp:
    data = json.load(fp)
    test_liste = data["edges"]
    fp.close()


TOTAL_EDGES=len(liste)

i=0
b = False
source = []
target = []
node_features = []
names = []

listeTemp = []
for dico in liste:

    citing = int(dico["citingPaper"])
    cited = int(dico["citedPaper"])
    G.add_edge(citing,  cited)
    G_train.add_edge(citing, cited)
    source.append(citing)
    target.append(cited)
    if citing not in listeTemp:
        node_features.append(dicoEmbT[str(citing)].detach().numpy())
        names.append(citing)
    listeTemp.append(citing)
    if cited not in listeTemp:
        node_features.append(dicoEmbT[str(cited)].detach().numpy())
        names.append(cited)


  
    listeTemp.append(cited)
    i+=1
    if i%1000==0:
        print("{} edges added on {}   --------   {} % ".format(i,TOTAL_EDGES,round(100*i/TOTAL_EDGES)))

"""
test_listeTemp = []
for dico in test_liste:
    citing = int("42"+str(dico["citingPaper"]))
    cited = int("42"+str(dico["citedPaper"]))
    G.add_edge(citing,  cited)
    G_test.add_edge(citing,cited)
    test_listeTemp.append(citing)
    test_listeTemp.append(cited)
"""
print(node_features[0])

#REMPLACER PAR LES NODES FEATURES
aretes = pd.DataFrame({"source": source, "target": target})
noeud = IndexedArray(np.array(node_features), index = names)
G = StellarGraph(noeud, aretes)


nodes = list(G.nodes())
number_of_walks = 1
length = 5
unsupervised_samples = UnsupervisedSampler(
    G, nodes=nodes, length=length, number_of_walks=number_of_walks
)
batch_size = 100
epochs = 3
num_samples = [10, 5]
layer_sizes = [50, 50]



generator = GraphSAGELinkGenerator(G, batch_size, num_samples)
train_gen = generator.flow(unsupervised_samples)


graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=generator, bias=True, dropout=0.0, normalize="l2"
)

x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)

x_inp_src = x_inp[0::2]
x_out_src = x_out[0]
embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

#pickle.dump(embedding_model, open("modeleKeras", "wb"))

embedding_model.save("./modeleKeras")
node_gen = GraphSAGENodeGenerator(G, batch_size, num_samples).flow(names)

node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)
import pickle 
pickle.dump(node_embeddings, open("node_embeddings", "wb"))

