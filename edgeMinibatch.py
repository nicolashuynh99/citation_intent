from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import torch

np.random.seed(123)

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, train_edges,test_edges, id2idx, batch_size=100, max_degree=25,**kwargs):
       
        self.G = G
        self.nodes = G.nodes()
        
        self.id2idx = id2idx
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        #self.test_adj = self.construct_test_adj()
        self.train_edges = train_edges
        self.test_edges  = test_edges
        #self.train_edges = self._remove_isolated(self.train_edges)
        #inutile, o:²n n'a pas d'edge isolé par définition
        self.val_edges = G.edges()
        self.val_set_size = len(self.val_edges)



    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.nodes or not n2 in self.G.nodes:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list


    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree), dtype = np.int32)
        deg = np.zeros((len(self.id2idx)+1,))

        for nodeid in self.G.nodes:
           
            """
            if self.G.nodes[nodeid]['test'] or self.G.nodes[nodeid]['val']:
                continue
            """
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors.astype(int)

        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        batch3 = []
        batch4 = []
        names1 = []
        names2 = []
        dicoMapping = {"background":[1.,0.,0.], "method": [0.,1.,0.], "result": [0.,0.,1.] }
        maximum = -1
        for i in range(len(batch_edges)):
            node1 = batch_edges[i][0]
            node2 = batch_edges[i][1]
            text_embed = batch_edges[i][2]
            batch1.append(self.id2idx[int(node1)])
            batch2.append(self.id2idx[int(node2)])
            names1.append(int(node1))
            names2.append(int(node2))
            batch4.append(dicoMapping[batch_edges[i][3]])


        for i in range(len(self.train_edges)):            
            text_embed = torch.tensor(self.train_edges[i][2])
            maximum = max(tf.convert_to_tensor(text_embed.detach().numpy()).shape[1], maximum)

        for i in range(len(batch_edges)):
            text_embed = batch_edges[i][2]
        
            a = tf.convert_to_tensor(text_embed.detach().numpy())
    
            dim0 = a.shape[0]
            paddings = tf.constant([[0,maximum-dim0],[0,0]])
          
            batch3.append(tf.pad(a, paddings, "CONSTANT"))
        
  

        feed_dict = {}
        feed_dict["batch_size"] = len(batch_edges)
        feed_dict["batch1"] = tf.Variable(batch1)
        feed_dict["batch2"] = tf.Variable(batch2)
        feed_dict["names1"] = np.array(names1)
        feed_dict["names2"] = np.array(names2)
        feed_dict["edge_embeddings"] =tf.Variable(batch3, trainable = False)
        feed_dict["labels"] = tf.Variable(batch4, trainable = False)
        return feed_dict



    def batch_feed_dict_test(self):
        batch_edges = self.test_edges
        batch1 = []
        batch2 = []
        batch3 = []
        batch4 = []
        dicoMapping = {"background":[1.,0.,0.], "method": [0.,1.,0.], "result": [0.,0.,1.] }
        maximum = -1
        for i in range(len(batch_edges)):
            node1 = batch_edges[i][0]

            node2 = batch_edges[i][1]
            text_embed = batch_edges[i][2]

            batch1.append(self.id2idx[int(node1)])
            batch2.append(self.id2idx[int(node2)])
            batch4.append(dicoMapping[batch_edges[i][3]])
        for i in range(len(self.test_edges)):

            text_embed = torch.tensor(self.test_edges[i][2])
            maximum = max(tf.convert_to_tensor(text_embed.detach().numpy()).shape[1], maximum)

        for i in range(len(batch_edges)):
            text_embed = batch_edges[i][2]

            a = tf.convert_to_tensor(text_embed.detach().numpy())

            dim0 = a.shape[0]
            paddings = tf.constant([[0,maximum-dim0],[0,0]])

            batch3.append(tf.pad(a, paddings, "CONSTANT"))


  
        feed_dict = {}
        feed_dict["batch_size"] = len(batch_edges)
        feed_dict["batch1"] = tf.Variable(batch1)
        feed_dict["batch2"] = tf.Variable(batch2)
        feed_dict["edge_embeddings"] =tf.Variable(batch3, trainable = False)
        feed_dict["labels"] = tf.Variable(batch4, trainable = False)
        return feed_dict


   
    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test'] 
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0
