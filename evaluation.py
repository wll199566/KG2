# numpy
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# dgl
import dgl
import dgl.function as fn
from dgl import DGLGraph

# models
from models.feature_extractor import FeatureExtractor
from models.GNN import GCN

# modules
from ARC_Dataset import ARC_Dataset
from modules.dict2graph_each_dict import dict2graph_hypo, dict2graph_support

# utils
from utils.nlp_utils import load_token2idx, load_word_matrix
from utils.file_system_utils import load_config
from utils import torch_utils

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def evaluation(feature_extractor_model, gnn_model, gnn_iter_num, gnn_out_size, criterion, loader, device):
    """
    Args:
        -epoch: the current epoch.
        -feature_extractor_model: the lstm model defined in feature_extractor.py.
        -gnn_model: the gcn model defined in GNN.py.
        -gnn_iter_num: the number of iteraion to get the steady states of graph.
        -gnn_out_size: node representation dimension from gnn.
        -criterion: loss function.
        -loader: data loader.
        -device: cpu or cuda
        -log_callback: used for write into log file.
    """
    feature_extractor_model.eval()
    gnn_model.eval()
    
    # define statistical variables to compute the accuracy
    correct = 0  # number of correct classified samples
    total = 0  # number of total samples
    
    with torch.no_grad():
        # use dataloader to get the data.
        for batch_idx, data in enumerate(loader, 1):
            # hypo_graphs contains hypothesis dict of dict returned by openie2dict.py
            # supp_graphs contains support dict of dict returned by openie2dict.py
            # label contains dict of correct label. 
            hypo_graphs, supp_graphs, label = data
            # NOTE: dictionary cannot be put into cuda directly
    
            # convert hypo_graphs from dict of dicts to dict of graphs
            hypo_graphs = dict2graph_hypo(hypo_graphs)
            # convert supp_graphs from dict of dicts to dict of graphs
            supp_graphs = dict2graph_support(supp_graphs)
    
            # feed in each graph in hypothesis and support dict to feature extractor
            batched_graphs = []  # to store all the graphs in hypothesis and support for batch in GNN 
            
            # for hypothesis
            pred_index_list_hypo = []  # to store the pred index for each hypothesis graph.
            node_num_hypo_graphs = []  # to store the number of total nodes in each hypo graph.
            for i, choice in enumerate(hypo_graphs["graphs_for_question"]):
                hypo_graphs["graphs_for_question"][i]["graph"] = feature_extractor_model(choice['graph'] )
                batched_graphs.append(hypo_graphs["graphs_for_question"][i]["graph"])
                node_num_hypo_graphs.append(hypo_graphs["graphs_for_question"][i]["graph"].number_of_nodes())
                # if the whole graph is Empty
                if choice["pred_idx"] == []:
                    pred_index_list_hypo.append([0])
                else:
                    pred_index_list_hypo.append(choice["pred_idx"]) 
    
            # for supports
            pred_index_list_supp = []  # to store the pred index for each support graph.
            node_num_supp_graphs = []  # to store the number of total nodes in each support graph.
            for i, choice in enumerate(supp_graphs["graphs_for_question"]):
                for j, support in enumerate(choice["graphs_for_choice"]):
                    supp_graphs["graphs_for_question"][i]["graphs_for_choice"][j]["graph"] = feature_extractor(support["graph"]) 
                    batched_graphs.append(supp_graphs["graphs_for_question"][i]["graphs_for_choice"][j]["graph"])
                    node_num_supp_graphs.append(supp_graphs["graphs_for_question"][i]["graphs_for_choice"][j]["graph"].number_of_nodes())
                    # if the whole graph is Empty
                    if support["pred_idx"] == []:
                        pred_index_list_supp.append([0])
                    else:
                        pred_index_list_supp.append(support["pred_idx"])
    
            # batch the all the graphs together
            # NOTE that first n graphs are for hypothesis and the left are supports
            # each row indicates one feature for one node.
            dgl_bg = dgl.batch(batched_graphs)   
            dgl_bg.ndata['feat'] = dgl_bg.ndata['feat'].to(device)

            # feed dgl batched graph into gnn_model
            for i in range(gnn_iter_num):
                batched_result = gnn_model(dgl_bg)
                dgl_bg.ndata['feat'] = batched_result
            
            # actually, we can use pred_index_list and label["num_of_choices"][0] to get the feature of pred nodes.
            # First, according to label["num_of_choice"] to connstruct two 3D tensors to store the features of pred nodes 
            
            # for the hypothesis, one layer contains all pred nodes feature for one choice.
            # get the max number of pred nodes in all hypo graphs
            max_num_pred_nodes = max(len(pred) for pred in pred_index_list_hypo)
            # construct a tensor for all hypo graphs
            # 20 is the output feature dim from GNN
            hypo_pred_nodes_feature = torch.zeros(label["num_of_choices"][0], max_num_pred_nodes, gnn_out_size)
            hypo_pred_nodes_feature = hypo_pred_nodes_feature.to(device)

            # for the supports, one layer contains all pred nodes feature for one choice(20 supp graphs).
            # get the max number of pred nodes in all supp graphs
            for i in range(label["num_of_choices"][0]):
                max_num_pred_nodes = 0
                num_pred_nodes = 0
                for pred in pred_index_list_supp[i: i+20]:
                    num_pred_nodes += len(pred)
                if max_num_pred_nodes < num_pred_nodes:
                    max_num_pred_nodes = num_pred_nodes
            # construct a tensor for all supp graphs
            # 20 is the output feature dim from GNN
            supp_pred_nodes_feature = torch.zeros(label["num_of_choices"][0], max_num_pred_nodes, gnn_out_size)
            supp_pred_nodes_feature = supp_pred_nodes_feature.to(device)
            
            # Second, to put the corresponding pred node feature into corresponding position in those tensors.
            batched_result_row = 0 # records the current batched_result row which will update after each graph
            
            # for the hypothesis
            # to add the pred node feature to the corresponding layer in tensor 
            for graph_idx, pred_node_list in enumerate(pred_index_list_hypo):
                for idx in pred_node_list:
                    row_idx_for_each_layer = 0 # records the row we can fill in each layer.
                    hypo_pred_nodes_feature[graph_idx][row_idx_for_each_layer] = normalize_tensor(batched_result[batched_result_row+idx])
                    row_idx_for_each_layer += 1
                batched_result_row += node_num_hypo_graphs[graph_idx] # update batched_result_row for next graph    
            
            # for the supports
            for choice_idx in range(label["num_of_choices"][0]):
                row_idx_for_each_layer = 0  # records the row we can fill in each layer.
                for graph_idx, pred_node_list in enumerate(pred_index_list_supp[20*choice_idx: 20*(choice_idx+1)]):
                    for idx in pred_node_list:
                        supp_pred_nodes_feature[choice_idx][row_idx_for_each_layer] = normalize_tensor(batched_result[batched_result_row+idx])
                        row_idx_for_each_layer += 1
                    batched_result_row += node_num_supp_graphs[20*choice_idx + graph_idx]  # update batched_result_row for next graph  
    
            # for each layer(choice) of hypo and supp, we need to compute the inner product to get the scoring function
            # can organize this to a function
            max_scores = torch.zeros(label["num_of_choices"][0])  # to compute the max score for each choice
            max_scores = max_scores.to(device)

            for layer in range(label["num_of_choices"][0]):
                inner_product = torch.mm(hypo_pred_nodes_feature[layer], torch.t(supp_pred_nodes_feature[layer]))
                max_inner_product = torch.max(inner_product)
                max_scores[layer] = max_inner_product - 0.5        
            
            max_scores = F.log_softmax(max_scores, dim=0)  # use log_softmax since we use NLLLoss
            
            # computer the accuracy
            _, predicted = torch.max(max_scores, dim=0)
            correct += (predicted==label["answerKey"]).sum().item()
            total += label["answerKey"].size(0)
            accuracy = correct / total

            # compute the loss
            loss = criterion(max_scores.unsqueeze(0), label["answerKey"].to(device))

        return loss.item(), accuracy  


if __name__ == "__main__":

    params = load_config('config.yaml')

    # define the dataloader
    eval_dir = './data'
    eval_loader = DataLoader(ARC_Dataset(eval_dir, dataset="test", is_easy=params['is_easy']), batch_size=1, shuffle=False)

    # load the token2idx dict and word mtx
    word2vec_dir = './data'
    token2idx_dict = load_token2idx(word2vec_dir)
    word_mtx = load_word_matrix(word2vec_dir)

    # load best model
    best_model_file = '../checkpoints/......pth'
    
    # load models
    feature_extractor = FeatureExtractor(token2idx_dict, word_mtx, params['embedding_size'])
    
    if params['activation'] == 'relu':
        activation_func = F.relu
    elif params['activation'] == 'sigmoid':
        activation_func = F.sigmoid
    else:
        raise NotImplementedError("activation function should be relu or sigmoid")        
    
    gnn = GCN(params['embedding_size'], params['gnn_hidden_size'], params['gnn_out_size'], activation_func)
    
    feature_extractor, gnn = torch_utils.load_model(feature_extractor, gnn, best_model_file)

    # define the loss function
    criterion = nn.NLLLoss()

    # put model into the corresponding device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    gnn.to(device)

    torch.backends.cudnn.benchmark = True

    # compute the 
    _, eval_acc = evaluation(feature_extractor, gnn, params['gnn_iter_num'], params['gnn_out_size'], criterion, eval_loader, device)

    print("the evaluation accuracy is", eval_acc * 100, '%')
