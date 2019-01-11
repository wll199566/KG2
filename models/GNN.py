"""
GNN model for KG2. This is adapted from DGL tutorial
https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
"""
import dgl
import dgl.function as fn
from dgl import DGLGraph

import torch 
import torch.nn as nn
import torch.nn.functional as F

# define massage and reduce function.
gcn_msg = fn.copy_src(src='feat', out='m')
#gcn_reduce = fn.sum(msg='m', out='h')
def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    reduced = torch.sum(nodes.mailbox['m'], 1) + nodes.data['feat']
    return {'feat': reduced}
    

# define the node UDF for apply_nodes, which consists of 
# two fully-connected layer, as stated as (9) in paper
# "Learning Steady-States of Iterative Algorithms over Graphs"
class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, activation):
        """
        Args:
            -in_feats: dimension of the input message feature states for a node (for the first linear layer).
            -hidden_feats: dimension of the output feature states from the first linear layer.
            -out_feats: dimension of the output feature states for a node as its final representation after the second linear layer.
            -activation: activation function for linear layers.
        """
        super(NodeApplyModule, self).__init__()
        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        feat = self.linear1(node.data['feat'])
        feat = self.activation(feat)
        feat = self.linear2(feat)
        
        return {'feat': feat}

# GCN layer
# here, we ommitted the dropout in the GCN paper for simplicity 
class GCNLayer(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, hidden_feats, out_feats, activation)

    def forward(self, g):
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('feat')

# GCN model consists of two GCN layers.
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.gcn = GCNLayer(in_feats, hidden_feats, out_feats, activation)           
    
    def forward(self, g):
        x = self.gcn(g)

        return x

#net = GCN(50, 40, 30, F.relu, 30, 20, F.relu)
#print(net)
