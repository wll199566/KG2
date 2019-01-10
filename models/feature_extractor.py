"""
LSTM model for each graph node to extract features from them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import networkx as nx
import dgl

from nltk.tokenize import word_tokenize

class FeatureExtractor(nn.Module):
    """
    Feature extracter for each node in the graph.
    """
    def __init__(self, token2idx_dict, pretrained_mtx, hidden_size, num_layers=1, freeze_embedding=True):
        """
        Args:
            -token2idx_dict: token_to_index dictionary.
            -pretrained_mtx: pretrained word representation matrix.
            -hidden_size: size of hidden state of LSTM, which is also the dim for each graph node.
            -num_layers: the number of LSTM layers for each step.
            -freeze_embedding: whether to continue to train the pretrained word matrix.
        """
        super(FeatureExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_embeddings, self.embedding_dim = pretrained_mtx.shape
        
        # load the token2idx dictionary
        self.token2idx_dict = token2idx_dict
        # construct the embedding layer
        #self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # initialize the embedding matrix using pretrained_mtx.
        #self.embed.weight.data.copy_(torch.from_numpy(pretrained_mtx))
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_mtx).float(), freeze=freeze_embedding)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, batch_first=True)

    def forward(self, graph):
        """
        Args:
            graph: a networkx graph which is generated by openie2graph.py.
        Outputs:
            dgl_graph: a dgl graph. we cannot convert networkx graph to dgl graph
                       directly.
        """
        # tokenized context for each node to get the index and word vector 
        # for padding and feeding into LSTM
        seqs = []
        for node_index in list(graph.nodes()):
            list_tokenized_words = word_tokenize(graph.nodes[node_index]["context"])
            seqs.append(list_tokenized_words)

        # step 1: construct the vocab2idx dictionary
        # this step is finished, and we can use self.token2idx_dict
        
        # step 2: load indexed data
        vectorized_seqs = [[self.token2idx_dict[tok] for tok in seq] for seq in seqs]

        # step3: make a model
        # we have made embedding layers and lstm
        # we can use self.embedding and self.lstm

        # step 4: pad instance with 0s till max length sequence
        # get the lenth of each seq in our batch (all node contexts in a graph)
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        # initialize the sentence matrix according to the max length and fill it      
        seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
        for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        # step 5: sort instances by sequence length in descending order
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        # NOTE: we use perm_idx to keep the original node index in the sorted list
        seq_tensor = seq_tensor[perm_idx]

        # step 6: embed the instances
        embedded_seq_tensor = self.embedding(seq_tensor)

        # step 7: call pack_padded_sequence with embedded instances
        packed_input = pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)

        # step 8: forward with LSTM
        _, (ht, _) = self.lstm(packed_input)

        # step 9: construct a dgl net.
        dgl_graph = dgl.DGLGraph()
        dgl_graph.add_nodes(len(list(graph.nodes())))
        src = []
        dst = []
        for edge in list(graph.edges()):
            src.append(edge[0])
            dst.append(edge[1])    
        dgl_graph.add_edges(src, dst)

        # step 10: initialize and fill the "feat" field for each dgl graph node.
        feat_init = torch.zeros(ht.squeeze(0).size())
        dgl_graph.ndata['feat'] = feat_init
        # use perm_idx to assign 'feat' for all nodes at once.
        dgl_graph.nodes[perm_idx].data['feat'] = ht.squeeze(0)

        return dgl_graph



