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


listeTemp = []
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


test_listeTemp = []
for dico in test_liste:
    citing = int("42"+str(dico["citingPaper"]))
    cited = int("42"+str(dico["citedPaper"]))
    G.add_edge(citing,  cited)
    G_test.add_edge(citing,cited)
    test_listeTemp.append(citing)
    test_listeTemp.append(cited)
 


setTemp = set(listeTemp)
edgelistBis = []
for edge in edgelist:
   if int(edge[0]) in setTemp and int(edge[1]) in setTemp:
      edgelistBis.append(edge)

test_setTemp = set(test_listeTemp)
test_edgelistBis = []
for edge in test_edgelist:
   if int("42"+str(edge[0])) in test_setTemp and int("42"+str(edge[1])) in test_setTemp:
      edge[0] = int("42" + str(edge[0]))
      edge[1] = int("42" + str(edge[1]))
      test_edgelistBis.append(edge)

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



modeleFinal = EdgeClassifier(features= features, adj= minibatch.adj, degrees = minibatch.deg,layer_infos=layer_infos,batch_size =  batch_size )

print(np.array(edgelist).shape)
print(np.array(test_edgelist).shape)
print(edgelist[0])
#assert 6==0

batch = minibatch.next_minibatch_feed_dict()
loss = train_step(modeleFinal, batch)


print(modeleFinal.model_bilstm(np.array([edgelist[0][2].detach().numpy()])))

to_dump_train, to_dump_test = [], []

for x in test_edgelist:
    a,b,c,d,e = x
    cc = modeleFinal.model_bilstm(np.array([c.detach().numpy()]))
    to_dump_test.append([a,b,cc,d,e])
    print(100*len(to_dump_test)/len(test_edgelist))


pickle.dump(to_dump_test,open("embs_test_edgelist.pkl","wb"))

for x in edgelist:
    a,b,c,d,e = x
    cc = modeleFinal.model_bilstm(np.array([c.detach().numpy()]))
    to_dump_train.append([a,b,cc,d,e])
    print(100*len(to_dump_train)/len(edgelist))

pickle.dump(to_dump_train,open("embs_edgelist.pkl","wb"))

assert 9==8

#embs_edgelist = modeleFinal.model_bilstm(np.array([x[2].detach().numpy() for x in edgelist]))
#embs_test_edgelist = modeleFinal.model_bilstm([x[2].detach().numpy() for x in test_edgelist]))


pickle.dump(embs_edgelist,open("embs_edgelist.pkl","wb"))
pickle.dump(embs_test_edgelist,open("embs_test_edgelist.pkl","wb"))

print("dump ok")
