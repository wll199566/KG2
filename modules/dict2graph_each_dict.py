"""
This script is to convert the dictionary obtained by the openie result to the graph
using networkx. 
"""

import networkx as nx
import json

import sys
sys.path.append("../")
from utils.file_system_utils import create_folder

def dict2graph_hypo(hypo_ques_dict):
    """
    Convert the dictionary which contains openie results to graph.
    - hypo_ques_dict: a dictionary like that in 
    "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Train-Hypothesis-dict.txt"
    """

    print("Processing hypothesis", hypo_ques_dict["id"][0], " ...")
    # construct a question dict for return
    output_ques_dict = {}
    # assign id
    output_ques_dict['id'] = hypo_ques_dict['id'][0]
    # construct a triples for question list for return
    graphs_for_question = []
    # get the triples for each choice
    for choice in hypo_ques_dict['triples_for_question']:
        # construct a choice dict for return
        dict_for_choice = {}
        dict_for_choice['label'] = choice['label'][0]
        # construct a set containing all the objects appearing in the subject and object in sentences
        piece2idx = {}  # convert all the pieces of sentences into index
        index_of_node = 0  # store the index of each node we go through
        # First, construct piece2idx dictionary for assigning the corrsponding nodes index
        # and avoid repeatition.
        for sentence in choice['triples_for_choice']:
            # print(sentence)
            # print(sentence.values())
            # the empty dictionary
            if sentence == {}:
                continue
            # construct the piece2idx dictionary
            for piece in sentence.values():
                # handle the case that obj is empty
                if piece == []:
                    continue
                if isinstance(piece[0], str) and (piece[0] not in piece2idx):
                    piece2idx[piece[0]] = index_of_node
                    index_of_node += 1
                if isinstance(piece[0], tuple):
                    for obj in piece:
                        if obj[0] not in piece2idx:
                            piece2idx[obj[0]] = index_of_node
                            index_of_node += 1
        # print(piece2idx)
        
        # Second, construct the directed graph for each hypothesis(choice)
        # construct a directed graph for each hypothesis(choice)
        HG = nx.DiGraph()
        # add nodes for the hypothesis graph
        for content, index in piece2idx.items():
            HG.add_node(index, context=content)
        # print(HG.nodes.data())
        # construct a set to record the predicate node index
        pred_index = set()
        # add edges for each hypothesis graph
        for sentence in choice['triples_for_choice']:
            # print(sentence)
            # if empty dictionary, we add a node with index -1 and context 'Empty'
            # NOTE that the Empty sentence node is always added finally to the graph
            # compared to other nodes.
            if sentence == {}:
                HG.add_node(-1, context='Empty')
            else:
                # add edge from pred to sub
                if ('sub' in sentence) and ('pred' in sentence):
                    HG.add_edge(
                        piece2idx[sentence['pred'][0]], piece2idx[sentence['sub'][0]], type='sub')
                # add edge from pred to obj
                if ('obj' in sentence) and (sentence['obj'] != []) and ('pred' in sentence):
                    for obj in sentence['obj']:
                        HG.add_edge(
                            piece2idx[sentence['pred'][0]], piece2idx[obj[0]], type='obj')
                # add edge from pred to time
                if ('time' in sentence) and ('pred' in sentence):
                    HG.add_edge(
                        piece2idx[sentence['pred'][0]], piece2idx[sentence['time'][0]], type='time')
                # add edge from pred to location
                if ('loc' in sentence) and ('pred' in sentence):
                    HG.add_edge(
                        piece2idx[sentence['pred'][0]], piece2idx[sentence['loc'][0]], type='loc')
                
                # add the pred index into the pred node set
                if 'pred' in sentence:
                    pred_index.add(piece2idx[sentence['pred'][0]])
                                
        # print(HG.edges.data())
        # store the graph into dict_for_choice
        dict_for_choice['graph'] = HG
        # store the pred index set into dict_for_choice
        dict_for_choice['pred_idx'] = list(pred_index)
        # add it into triples_for_question list
        graphs_for_question.append(dict_for_choice)
    
    # add it into output_ques_dict
    output_ques_dict['graphs_for_question'] = graphs_for_question
    
    # return the output_qes_dict
    # print(output_ques_dict)
    return output_ques_dict


def dict2graph_support(support_ques_dict):
    """
    Convert the dictionary which contains openie results to graph.
    - hypo_ques_dict: a dictionary like that in 
    "../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Train-Supports-dict.txt"
    """
    
    print("Processing support", support_ques_dict["id"][0], " ...")
        
    # construct a question dict for return
    output_ques_dict = {}
    # assign id
    output_ques_dict['id'] = support_ques_dict['id'][0]
    # construct a graphs for question list for return
    graphs_for_question = []
    # get the triples for each choice
    for choice in support_ques_dict['triples_for_question']:
        # construct a choice dict for return
        dict_for_choice = {}
        dict_for_choice['label'] = choice['label'][0]
        
        # construct a graphs for choice list for return
        graphs_for_choice = []
        
        # get the triples for each support
        for support in choice['triples_for_choice']:
            # construct a support dict for return
            dict_for_support = {}
            dict_for_support['eid'] = support['eid'][0]
            
            # construct a set containing all the objects appearing in the subject and object in sentences
            piece2idx = {}  # convert all the pieces of sentences into index
            index_of_node = 0  # store the index of each node we go throu
            # First, construct piece2idx dictionary for assigning the corrsponding nodes index
            # and avoid repeatition.
            for sentence in support['triples_for_support']:
                # print(sentence)
                # print(sentence.values())
                # the empty dictionary
                if sentence == {}:
                    continue
                # construct the piece2idx dictionary
                for piece in sentence.values():
                    #print(piece)
                    # handle the case that obj is empty
                    if piece == []:
                        continue
                    if isinstance(piece[0], str) and (piece[0] not in piece2idx):
                        piece2idx[piece[0]] = index_of_node
                        index_of_node += 1
                    if isinstance(piece[0], tuple):
                        for obj in piece:
                            if obj[0] not in piece2idx:
                                piece2idx[obj[0]] = index_of_node
                                index_of_node += 1
                # print(piece2id)
            # Second, construct the directed graph for each support
            # construct a directed graph for each support
            SG = nx.DiGraph()
            # add nodes for the hypothesis graph
            for content, index in piece2idx.items():
                SG.add_node(index, context=content)
            # print(SG.nodes.data(
            # construct a set to record the predicate node index
            pred_index = set()
            # add edges for each support graph
            if support['triples_for_support'] == []:
                SG.add_node(-1, context='Empty')
            for sentence in support['triples_for_support']:
                # print(sentence)
                # if empty dictionary, we add a node with index -1 and context 'Empty'
                if sentence == {}:
                    SG.add_node(-1, context='Empty')
                else:
                    # add edge from pred to sub
                    if ('sub' in sentence) and ('pred' in sentence):
                        SG.add_edge(
                            piece2idx[sentence['pred'][0]], piece2idx[sentence['sub'][0]], type='sub')
                    # add edge from pred to obj
                    if ('obj' in sentence) and (sentence['obj'] != []) and ('pred' in sentence):
                        for obj in sentence['obj']:
                            SG.add_edge(
                                piece2idx[sentence['pred'][0]], piece2idx[obj[0]], type='obj')
                    # add edge from pred to time
                    if ('time' in sentence) and ('pred' in sentence):
                        SG.add_edge(
                            piece2idx[sentence['pred'][0]], piece2idx[sentence['time'][0]], type='time')
                    # add edge from pred to location
                    if ('loc' in sentence) and ('pred' in sentence):
                        SG.add_edge(
                            piece2idx[sentence['pred'][0]], piece2idx[sentence['loc'][0]], type='loc')
                    # add pred node index into the pred_index set
                    if 'pred' in sentence:
                        pred_index.add(piece2idx[sentence['pred'][0]])
            # print(SG.edges.data())
            # store the graph into dict_for_support 
            dict_for_support['graph'] = SG
            # store the pred node index into dict_for_support
            dict_for_support['pred_idx'] = list(pred_index)
            # add it into graph_for_choice list
            graphs_for_choice.append(dict_for_support)
            
        dict_for_choice['graphs_for_choice'] = graphs_for_choice
        graphs_for_question.append(dict_for_choice)
    
    # add it into output_ques_dict
    output_ques_dict['graphs_for_question'] = graphs_for_question
    
    # return the output_qes_dict
    # print(output_ques_dict)
    return output_ques_dict


if __name__ == "__main__":

    # dict2graph for hypothesis
    hypo_folders = ["../data/ARC-Hypothesis-networkx-graph",
                    "../data/ARC-Hypothesis-networkx-graph/ARC-Easy",
                    "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge"]
    hypo_input_files = ["../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Train-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Dev-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Test-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Train-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Test-Hypothesis-dict.txt"]
    hypo_output_files = ["../data/ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Train-Hypothesis-networkx-graph.txt",
                         "../data/ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Dev-Hypothesis-networkx-graph.txt",
                         "../data/ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Test-Hypothesis-networkx-graph.txt",
                         "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Train-Hypothesis-networkx-graph.txt",
                         "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-networkx-graph.txt",
                         "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Test-Hypothesis-networkx-graph.txt"]
    with open(hypo_input_files[0], "r") as fin:
        for i, line in enumerate(fin, 1):
            if i > 1:
                break
            hypo_dict = json.loads(line)
            hypo_dict = dict2graph_hypo(hypo_dict)
    print("hypo_dict")
    print(hypo_dict)
    print("\n")
    
    
    # dict2graph for hypothesis
    support_folders = ["../data/ARC-Supports-networkx-graph",
                       "../data/ARC-Supports-networkx-graph/ARC-Easy",
                       "../data/ARC-Supports-networkx-graph/ARC-Challenge"]

    support_input_files = ["../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Train-Supports-dict.txt",
                           "../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Dev-Supports-dict.txt",
                           "../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Test-Supports-dict.txt",
                           "../data/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Train-Supports-dict.txt",
                           "../data/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Dev-Supports-dict.txt",
                           "../data/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Test-Supports-dict.txt"]

    support_output_files = ["../data/ARC-Supports-networkx-graph/ARC-Easy/ARC-Easy-Train-Supports-networkx-graph.txt",
                            "../data/ARC-Supports-networkx-graph/ARC-Easy/ARC-Easy-Dev-Supports-networkx-graph.txt",
                            "../data/ARC-Supports-networkx-graph/ARC-Easy/ARC-Easy-Test-Supports-networkx-graph.txt",
                            "../data/ARC-Supports-networkx-graph/ARC-Challenge/ARC-Challenge-Train-Supports-networkx-graph.txt",
                            "../data/ARC-Supports-networkx-graph/ARC-Challenge/ARC-Challenge-Dev-Supports-networkx-graph.txt",
                            "../data/ARC-Supports-networkx-graph/ARC-Challenge/ARC-Challenge-Test-Supports-networkx-graph.txt"]
    
    with open(support_input_files[0], "r") as fin:
        for i, line in enumerate(fin, 1):
            if i > 1:
                break
            support_dict = json.loads(line)
            support_dict = dict2graph_support(support_dict)
    
    print("support_dict")
    print(support_dict)
    print("\n")
       

    