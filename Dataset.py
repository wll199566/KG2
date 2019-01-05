"""
This script defines the customized dataset for ARC dataset 
for train, dev and test set.
"""

import json
import networkx as nx
from torch.utils.data.dataset import Dataset

class ARC_Dataset(Dataset):
    def __init__(self, folder, dataset="train", is_easy=True):
        """
        Args:
            folder: the root folder of the ARC dataset, like "../data/"
            dataset: "train", "dev" or "test" set
            is_easy: "easy" or "challenge", for different dataset
        """
        if is_easy:
            if dataset == "train":
                hypo_graph_file = folder + "ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Train-Hypothesis-networkx-graph.txt"
                supp_graph_file = folder + "ARC-Supports-networkx-graph/ARC-Easy/ARC-Easy-Train-Supports-networkx-graph.txt"
            elif dataset == "dev":
                hypo_graph_file = folder + "ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Dev-Hypothesis-networkx-graph.txt"
                supp_graph_file = folder + "ARC-Supports-networkx-graph/ARC-Easy/ARC-Easy-Dev-Supports-networkx-graph.txt"
            elif dataset == "test":
                hypo_graph_file = folder + "ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Test-Hypothesis-networkx-graph.txt"    
                supp_graph_file = folder + "ARC-Supports-networkx-graph/ARC-Easy/ARC-Easy-Test-Supports-networkx-graph.txt"
            else:
                raise NotImplementedError
        else:
            if dataset == "train":
                hypo_graph_file = folder + "ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Train-Hypothesis-networkx-graph.txt"
                supp_graph_file = folder + "ARC-Supports-networkx-graph/ARC-Challenge/ARC-Challenge-Train-Supports-networkx-graph.txt"
            elif dataset == "dev":
                hypo_graph_file = folder + "ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-networkx-graph.txt"
                supp_graph_file = folder + "ARC-Supports-networkx-graph/ARC-Challenge/ARC-Challenge-Dev-Supports-networkx-graph.txt"
            elif dataset == "test":
                hypo_graph_file = folder + "ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Test-Hypothesis-networkx-graph.txt"    
                supp_graph_file = folder + "ARC-Supports-networkx-graph/ARC-Challenge/ARC-Challenge-Test-Supports-networkx-graph.txt"
            else:
                raise NotImplementedError("dataset is not train, dev or test")

        # construct a list to contain all the hypothesis dictionaries read from the file.
        # then convert the graph to networkx graph.
        self.hypo_graphs = []
        with open(hypo_graph_file, 'r') as fin:
            for line in fin:
                hypo_dict = json.loads(line)
                for i in range(len(hypo_dict["graphs_for_question"])):
                    hypo_dict["graphs_for_question"][i]["graph"] = nx.node_link_graph(hypo_dict["graphs_for_question"][i]["graph"])
                self.hypo_graphs.append(hypo_dict)        
        print(hypo_graph_file, "has been loaded!")                        

        # construct a list to contain all the support dictionaries read from the file.
        # then convert the graph to networkx graph.
        self.supp_graphs = []
        with open(supp_graph_file, 'r') as fin:
            for line in fin:
                supp_dict = json.loads(line)
                for dict_for_choice in supp_dict["graphs_for_question"]:
                    for i in range(len(dict_for_choice["graphs_for_choice"])):
                        dict_for_choice["graphs_for_choice"][i]["graph"] = nx.node_link_graph(dict_for_choice["graphs_for_choice"][i]["graph"])
            self.supp_graphs.append(supp_dict)   
        print(supp_graph_file, "has been loaded!")     

        if len(self.hypo_graphs) != len(self.supp_graphs):
            raise ValueError("the number of samples in hypo and supp are different in", dataset)  

    def __len__(self):
        return len(self.hypo_graphs)

    def __getitem__(self, index):
                             

                

        
