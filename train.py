"""
This script is for training KG2 networks.
"""
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
from utils.file_system_utils import create_folder
from utils.Timer import Timer
from utils.AverageMeter import AverageMeter
from utils import torch_utils 

# system
import argparse
import time

# create folders to store checkpoints and log files
create_folder('checkpoints')
folderPath = 'checkpoints/session_' + Timer.timeFilenameString() + '/'
create_folder(folderPath)

create_folder('log')
logPath = 'log/log_' + Timer.timeFilenameString()

params = load_config('config.yaml')

torch.set_default_tensor_type('torch.cuda.FloatTensor')

batch_time = AverageMeter()
data_time = AverageMeter()
losses = AverageMeter()
losses_list = [AverageMeter() for i in range(6)]
end = time.time()
best_model = params['best_model']

def append_line_to_log(line='\n'):
    with open(logPath, 'a') as f:
        f.write(line + '\n')

def parse_cli():
    parser = argparse.ArgumentParser(description='PyTorch KG2')
    
    # add parser
    # hyperparameter
    parser.add_argument('--batch-size', type=int, default=params['batch_size'], metavar='N',
                        help='input batch size for training (default: ' + str(params['batch_size']) + ')')
    
    parser.add_argument('--epochs', type=int, default=params['epochs'], metavar='N',
                        help='number of epochs to train (default: ' + str(params['epochs']) + ')')

    parser.add_argument('--embedding_size', default=params['embedding_size'], type=int, metavar='EMBED',
                        help="dimension of word vector.")

    parser.add_argument('--gnn_iter_num', default=params['gnn_iter_num'], type=int, metavar="GNNITER",
                        help="the number of iteration of gnn to get the steady states")
    
    parser.add_argument('--gnn_hidden_size', default=params['gnn_hidden_size'], type=int, metavar="HIDDEN",
                        help="the hidden size when compute the aggregation function of massages")
    
    parser.add_argument('--gnn_out_size', default=params['gnn_out_size'], type=int, metavar="OUT",
                        help="the final size when compute the aggregation function of massages")
    
    parser.add_argument('--gnn_activation', default=params['gnn_activation'], type=str, metavar='ACT',
                       help="gnn activation function of aggregation function.")

    parser.add_argument('--epochs', type=int, default=params['epochs'], metavar='N',
                        help='number of epochs to train (default: ' + str(params['epochs']) + ')')

    parser.add_argument('--lr', type=float, default=params['init_learning_rate'], metavar='LR',
                        help='inital learning rate (default: ' + str(params['init_learning_rate']) + ')')

    #parser.add_argument('--decay', type=float, default=params['decay'], metavar='DE',
    #                    help='SGD learning rate decay (default: ' + str(params['decay']) + ')')

    parser.add_argument('--beta1', type=float, default=params['beta1'], metavar='B1',
                        help=' Adam parameter beta1 (default: ' + str(params['beta1']) + ')')

    parser.add_argument('--beta2', type=float, default=params['beta2'], metavar='B2',
                        help=' Adam parameter beta2 (default: ' + str(params['beta2']) + ')')                    
                        
    parser.add_argument('--epsilon', type=float, default=params['epsilon'], metavar='EL',
                        help=' Adam regularization parameter (default: ' + str(params['epsilon']) + ')')

    #parser.add_argument('--dampening', type=float, default=params['dampening'], metavar='DA',
    #                    help='SGD dampening (default: ' + str(params['dampening']) + ')')                    
                   
    # system training
    parser.add_argument('--word2vec_dir', default='./data', type=str, metavar='PATHW2V',
                        help="path to the root folder containing token2idx and word matrix files.")

    parser.add_argument('--train_dir', default='./data', type=str, metavar='PATHT', 
                        help="path to the root folder containing all the train data folders")
    
    parser.add_argument('--val_dir', default='./data', type=str, metavar='PATHV', 
                        help="path to the root folder containing all the validation data folders")
    
    parser.add_argument('--is_easy', default=params['is_easy'], type=bool, metavar='EASY',
                        help="ARC-Easy(default) or ARC-Challenge dataset")

    parser.add_argument('--log-interval', type=int, default=params['log_interval'], metavar='N',
                        help='how many batches to wait before logging training status')                    
    
    parser.add_argument('--seed', type=int, default=params['seed'], metavar='S',
                        help='random seed (default: ' + str(params['seed']) + ')')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')                    

    args = parser.parse_args()

    return args


def normalize_tensor(torch_tensor):
    """
    normalize torch tensor.
    Args:
        -torch_tensor: torch tensor to normalize.
    Returns:
        -the L2-normalized tensor. 
    """
    torch_tensor_n = torch.norm(torch_tensor, p=2).detach()
    
    return torch_tensor.div(torch_tensor_n.expand_as(torch_tensor))


def train(epoch, feature_extractor_model, gnn_model, gnn_iter_num, gnn_out_size, optimizer, scheduler, criterion, loader, device, log_callback):
    """
    Args:
        -epoch: the current epoch.
        -feature_extractor_model: the lstm model defined in feature_extractor.py.
        -gnn_model: the gcn model defined in GNN.py.
        -gnn_iter_num: the number of iteraion to get the steady states of graph.
        -gnn_out_size: node representation dimension from gnn.
        -optimizer: pytorch optimizer.
        -scheduler: pytorch optimization scheduler.
        -criterion: loss function.
        -loader: data loader.
        -device: cpu or cuda
        -log_callback: used for write into log file.
    """
    end = time.time()
    feature_extractor_model.train()
    gnn_model.train()

    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']

    # use dataloader to get the data.
    for batch_idx, data in enumerate(train_loader, 1):
        # hypo_graphs contains hypothesis dict of dict returned by openie2dict.py
        # supp_graphs contains support dict of dict returned by openie2dict.py
        # label contains dict of correct label. 
        hypo_graphs, supp_graphs, label = data
        # NOTE: dictionary cannot be put into cuda directly
        
        data_time.update(time.time() - end)

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
        for layer in range(label["num_of_choices"][0]):
            inner_product = torch.mm(hypo_pred_nodes_feature[layer], torch.t(supp_pred_nodes_feature[layer]))
            max_inner_product = torch.max(inner_product)
            max_scores[layer] = max_inner_product - 0.5        
        
        max_scores = F.log_softmax(max_scores, dim=0)  # use log_softmax since we use NLLLoss

        # compute the loss
        loss = criterion(max_scores.unsqueeze(0), label["answerKey"])
        
        # to the backprop
        feature_extractor_model.zero_grad()
        gnn_model.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # record essential informations into log file.
        if batch_idx % args.log_interval == 0:
            log_callback('Epoch: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1} batches, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1} batches, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'.format(
                epoch, args.log_interval, learning_rate, batch_time=batch_time,
                data_time=data_time))
            
            log_callback('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader), loss.item()))
            log_callback()
            
            log_callback('Loss = {loss:.8f}\t'
                    .format(loss=loss.item()))
            
            log_callback()
            log_callback("current time: " + Timer.timeString())
            
            batch_time.reset()
            data_time.reset()

    torch_utils.save(folderPath + 'KG2_' + str(epoch) + '.cpkt', epoch, feature_extractor_model, gnn_model, optimizer, scheduler)


def validation(epoch, feature_extractor_model, gnn_model, gnn_iter_num, gnn_out_size, criterion, loader, device, log_callback):
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
    end = time.time()
    feature_extractor_model.val()
    gnn_model.val()

    with torch.no_grad():
        # use dataloader to get the data.
        for batch_idx, data in enumerate(train_loader, 1):
            # hypo_graphs contains hypothesis dict of dict returned by openie2dict.py
            # supp_graphs contains support dict of dict returned by openie2dict.py
            # label contains dict of correct label. 
            hypo_graphs, supp_graphs, label = data
            # NOTE: dictionary cannot be put into cuda directly
            
            data_time.update(time.time() - end)
    
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
            for layer in range(label["num_of_choices"][0]):
                inner_product = torch.mm(hypo_pred_nodes_feature[layer], torch.t(supp_pred_nodes_feature[layer]))
                max_inner_product = torch.max(inner_product)
                max_scores[layer] = max_inner_product - 0.5        
            
            max_scores = F.log_softmax(max_scores, dim=0)  # use log_softmax since we use NLLLoss
    
            # compute the loss
            loss = criterion(max_scores.unsqueeze(0), label["answerKey"])

            # records essential information into log file.
        log_callback('epoch: {0}\t'
                'Time {batch_time.sum:.3f}s / {1} epochs, ({batch_time.avg:.3f})\t'
                'Data load {data_time.sum:.3f}s / {1} epochs, ({data_time.avg:3f})\n'
                'Loss = {loss:.8f}\n'.format(
            epoch, batch_idx, batch_time=batch_time,
            data_time=data_time, loss=loss.item()))
        
        log_callback()
        
        log_callback('Loss = {loss:.8f}\t'
                .format(loss=loss.item()))

        log_callback(Timer.timeString())

        batch_time.reset()
         
        return loss.item()        


############################################ main #########################################
if __name__ == "__main__":
    # add argparser here
    args = parse_cli()
    
    # to make everytime the ransomization is the same
    torch.manual_seed(args.seed)

    start_epoch = 1

    # get the train root path and val root path.
    train_dir = args.train_dir
    val_dir = args.val_dir
    
    # define the dataloader
    train_loader = DataLoader(ARC_Dataset(train_dir, dataset="train", is_easy=args.is_easy), batch_size=1, shuffle=True)
    val_loader = DataLoader(ARC_Dataset(train_dir, dataset="dev", is_easy=args.is_easy), batch_size=1, shuffle=True)

    # load the token2idx dict and word mtx
    token2idx_dict = load_token2idx(args.word2vec_dir)
    word_mtx = load_word_matrix(args.word2vec_dir)

    # instantiate the feature extractor model
    feature_extractor = FeatureExtractor(token2idx_dict, word_mtx, args.embedding_size)

    # instantiate the gnn model 
    if args.activation == 'relu':
        activation_func = F.relu
    elif args.activation == 'sigmoid':
        activation_func = F.sigmoid
    else:
        raise NotImplementedError("activation function should be relu or sigmoid")        
    
    gnn = GCN(args.embedding_size, args.gnn_hidden_size, args.gnn_out_size, activation_func)

    # define optimizer for lstm and gnn at the same time
    optimizer = optim.Adam(filter(lambda p: p.requirs_grad, list(feature_extractor.parameters()) + list(gnn.parameters())), lr=args.lr, betas=(args.beta1,args.beta2), eps=args.epsilon)
    
    # define scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # define the loss function
    criterion = nn.NLLLoss()
    
    if args.resume:
        start_epoch, feature_extractor, gnn, optimizer, scheduler = torch_utils.load(args.resume, feature_extractor, gnn, optimizer, start_epoch, scheduler)
        append_line_to_log('resuming ' + args.resume + '... at epoch ' + str(start_epoch))
 
    # put model into the corresponding device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)
    gnn.to(device)

    append_line_to_log('executing on device: ')
    append_line_to_log(str(device))

    torch.backends.cudnn.benchmark = True
    history = {'validation_loss':[]}

    best_val_loss = np.inf

    # begin to train the model
    for epoch in range(start_epoch, args.epochs + 1):
    
        #train(epoch, model, optimizer, criterion1, criterion2, lamb, train_loader, device, append_line_to_log)
        train(epoch, feature_extractor, gnn, args.gnn_iter_num, args.gnn_out_size, optimizer, scheduler, criterion, train_loader, device, append_line_to_log)
        
        #val_loss = validation(model, criterion1, criterion2, lamb, val_loader, device, append_line_to_log)
        val_loss = validation(epoch, feature_extractor, gnn, args.gnn_iter_num, args.gnn_out_size, criterion, val_loader, device, append_line_to_log)
        
        scheduler.step(val_loss) # to use ReduceLROnPlateau must specify the matric
        
        # save the best model
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
    
        if is_best:
             best_model_file = 'best_model_' + str(epoch) + '.pth'
             best_model_file = folderPath + best_model_file
             torch_utils.save_model(feature_extractor.state_dict(), gnn.state_dict(), best_model_file)
        model_file = 'model_' + str(epoch) + '.pth'
        model_file = folderPath + model_file
    
        torch_utils.save_model(feature_extractor.state_dict(), gnn.state_dict(), model_file)
        append_line_to_log('Saved model to ' + model_file)

print("validation loss history:", history)