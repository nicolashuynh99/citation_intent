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

from stellargraph import IndexedArray
from stellargraph.mapper import GraphSAGENodeGenerator


from stellargraph import StellarGraph
from tensorflow import keras
import pandas as pd
edgelist = pkl.load(open("../list_full_scibert_embs_citations", "rb"))
test_edgelist = pkl.load(open("../test-list_scibert_embs_citations.pkl", "rb"))



np.random.seed(21)


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
listeTempbis = set()

dicoIdMap = {}
compteur = 0

for dico in liste:

    citing = int(dico["citingPaper"])

    cited = int(dico["citedPaper"])
    G.add_edge(citing,  cited)
    G_train.add_edge(citing, cited)
    listeTemp.append(citing)
    listeTemp.append(cited)
    i+=1
    if i%1000==0:
        print("{} edges added on {}   --------   {} % ".format(i,TOTAL_EDGES,round(100*i/TOTAL_EDGES)))
    source.append(citing)
    target.append(cited)
    if citing not in listeTempbis:
        node_features.append(dicoEmbT[str(citing)].detach().numpy())
        names.append(citing)
        dicoIdMap[citing] = compteur
        compteur +=1
    listeTempbis.add(citing)
    if cited not in listeTempbis:
        node_features.append(dicoEmbT[str(cited)].detach().numpy())
        names.append(cited)
        dicoIdMap[cited] = compteur
        compteur +=1
    listeTempbis.add(cited)
    


test_listeTemp = []
for dico in test_liste:
    citing = int("42"+str(dico["citingPaper"]))
    cited = int("42"+str(dico["citedPaper"]))
    G.add_edge(citing,  cited)
    G_test.add_edge(citing,cited)
    test_listeTemp.append(citing)
    test_listeTemp.append(cited)
    source.append(citing)
    target.append(cited)
    if citing not in listeTempbis:
        node_features.append(test_dicoEmbT[str(citing)[2:]].detach().numpy())
        names.append(citing)
        dicoIdMap[citing] = compteur
        compteur +=1        
    listeTempbis.add(citing)
    if cited not in listeTempbis:
        node_features.append(test_dicoEmbT[str(cited)[2:]].detach().numpy())
        names.append(cited)
        dicoIdMap[cited] = compteur
        compteur +=1
    listeTempbis.add(cited)
   
id_map = dicoIdMap
aretes = pd.DataFrame({"source": source, "target": target})
noeud = IndexedArray(np.array(node_features), index = names)
stellarG = StellarGraph(noeud, aretes)


setTemp = set(listeTemp)
edgelistBis = []
rmv_edge = 0
for edge in edgelist:
   if int(edge[0]) in setTemp and int(edge[1]) in setTemp:
      edgelistBis.append(edge)
"""
      try:
         
          G.remove_edge(int(edge[0]), int(edge[1]))
          rmv_edge +=1

      except:
          a = 0.01
"""                
print(rmv_edge)

test_setTemp = set(test_listeTemp)
test_edgelistBis = []
for edge in test_edgelist:
   if int("42"+str(edge[0])) in test_setTemp and int("42"+str(edge[1])) in test_setTemp:
      edge[0] = int("42" + str(edge[0]))
      edge[1] = int("42" + str(edge[1]))
      test_edgelistBis.append(edge)
      """
      try:
          G.remove_edge(edge[0],edge[1])
      except:
         a = 0.01
"""
print("avant:", len(edgelist))
print("apres:", len(edgelistBis))



edgelist = edgelistBis
test_edgelist = test_edgelistBis




dicoIdMap = {}
compteur = 0
for node in G_train.nodes():
    dicoIdMap[node] = compteur
    compteur+=1

for node in G_test.nodes():
   
    dicoIdMap[node] = compteur
    compteur +=1
id_map = dicoIdMap


minibatch = EdgeMinibatchIterator(G, edgelist,test_edgelist,
                                  id_map,
                                  batch_size=100,
                                  max_degree=3)

sampler = UniformNeighborSampler(minibatch.adj)

layer_infos = [SAGEInfo("node", sampler, 3, 128),
               SAGEInfo("node", sampler, 3, 256)]


features = []

maximum = -1
"""
for node in G.nodes: 

    text_embed = dicoEmbT[str(node)]
    maximum = max(tf.convert_to_tensor(text_embed.detach().numpy()).shape[1], maximum)
print("MAXIMUM", maximum)
for node in G.nodes:
    
    
    text_embed = dicoEmbT[str(node)]

    a = tf.convert_to_tensor(text_embed.detach().numpy())

    dim0 = a.shape[0]
    paddings = tf.constant([[0,maximum-dim0],[0,0]])

    features.append(tf.pad(a, paddings, "CONSTANT"))
"""

for node in G_train.nodes:
    features.append(tf.convert_to_tensor(dicoEmbT[str(node)].detach().numpy()))



for node in G_test.nodes:
    features.append(tf.convert_to_tensor(test_dicoEmbT[str(node)[2:]].detach().numpy()))

#    features.append(np.array([1.]))

#features = tf.Variable(features, name = "features", trainable = False)

batch_size = 100
node_embedder = keras.models.load_model("./modeleKeras")


num_samples = [10,5]
node_gen1 = GraphSAGENodeGenerator(stellarG,batch_size, num_samples).flow(names)

#embeddings = node_embedder.predict(node_gen1, workers = 4, verbose = 1)

modeleFinal = EdgeClassifier(features= features, adj= minibatch.adj, degrees = minibatch.deg,layer_infos=layer_infos,batch_size =  batch_size)



#modeleFinal.load_weights("./checkpoints/checkpoint10")
#print("WEIGHTS")
#edgelist et  test_edgelist



maximum = -1


"""
batch3 = []
for i in range(len(test_edgelist)):
            text_embed = test_edgelist[i][2]
            maximum = max(tf.convert_to_tensor(text_embed.detach().numpy()).shape[0], maximum)
for i in range(len(test_edgelist)):
            text_embed = test_edgelist[i][2]
            a = tf.convert_to_tensor(text_embed.detach().numpy())
            dim0 = a.shape[0]
            paddings = tf.constant([[0,maximum-dim0],[0,0]])
            batch3.append(tf.pad(a, paddings, "CONSTANT"))
"""
dicoMapping = {"background":[1.,0.,0.], "method": [0.,1.,0.], "result": [0.,0.,1.] }
"""
test_citations = {"edge_embeddings": tf.Variable(batch3)}
test_labels =tf.Variable([dicoMapping[test_edgelist[i][3]] for i in range(len(test_edgelist))])
"""



@tf.function
def class_loss(predictions, labels):
        # Weight decay iloss
        """
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += self.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += self.weight_decay * tf.nn.l2_loss(var)
        """
        # classification loss
        # print(tf.nn.softmax_cross_entropy_with_logits(
              #  logits=predictions,

               # labels=labels)       ) 
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=predictions,
                labels=labels))

       

def train_step(model, batch):
  labels = batch["labels"]
  with tf.GradientTape() as tape:


    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).

    predictions = model(batch, training=True)
    labels = tf.stop_gradient(labels)
    #loss = class_loss(predictions, labels)
   
    loss =tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels,
                predictions,
                ))
  gradients = tape.gradient(loss, model.trainable_variables)
 # print([var.name for var in tape.watched_variables()])
  
 
  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

n_epochs = 50
metric = tfa.metrics.F1Score(num_classes=3, average = "macro")

for i in range(n_epochs):
    
    print("EPOCH ", i , "#################################")
    while not minibatch.end():  
        batch = minibatch.next_minibatch_feed_dict()
        loss = train_step(modeleFinal, batch)
        print("loss:", loss)
     
    minibatch.shuffle()
    
    modeleFinal.save_weights("./checkpoints/checkpoint-sage"+str(i))
    print("TEST LOSS: #####################################")
    test_batch = minibatch.batch_feed_dict_test()
    predictions = modeleFinal(test_batch, training = False)
    test_labels =test_batch["labels"]
    loss = class_loss(predictions, test_labels)
    
    print(loss)
    print("F1 SCORE:")
    
    metric.update_state(test_labels, predictions)
    result = metric.result()
    print(result.numpy())
    
    #loss sur le validation set ici ##############################
