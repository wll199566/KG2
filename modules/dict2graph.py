"""
This script is to convert the dictionary obtained by the openie result to the graph
using networkx. 
"""

import networkx as nx
import json

import sys
sys.path.append("../")
from utils.file_system_utils import create_folder

def dict2graph_hypo(input_file, output_file):
    """
    Convert the dictionary which contains openie results to graph.
    - input_file: path to Hypothesis-dict file.
    - output_file: path to Hypothesis graph file which will feed into LSTM and GNN.
    """

    with open(input_file, 'r') as fin:
        print("Processing ", input_file, " ...")
        with open(output_file, 'w') as fout:
            for i, line in enumerate(fin, 1):
                # if i > 1:
                #    break
                # read in the hypothesis question dictionary
                hypo_ques_dict = json.loads(line)
                # construct a question dict for output
                output_ques_dict = {}
                # assign id
                output_ques_dict['id'] = hypo_ques_dict['id']
                # construct a triples for question list for output
                graphs_for_question = []

                # get the triples for each choice
                for choice in hypo_ques_dict['triples_for_question']:
                    # construct a choice dict for output
                    dict_for_choice = {}
                    dict_for_choice['label'] = choice['label']

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
                            if isinstance(piece, str) and (piece not in piece2idx):
                                piece2idx[piece] = index_of_node
                                index_of_node += 1
                            if isinstance(piece, list):
                                for obj in piece:
                                    if obj not in piece2idx:
                                        piece2idx[obj] = index_of_node
                                        index_of_node += 1
                        # print(piece2idx)

                    # Second, construct the directed graph for each hypothesis(choice)
                    # construct a directed graph for each hypothesis(choice)
                    HG = nx.DiGraph()
                    # add nodes for the hypothesis graph
                    for content, index in piece2idx.items():
                        HG.add_node(index, context=content)
                    # print(HG.nodes.data())

                    # add edges for each hypothesis graph
                    for sentence in choice['triples_for_choice']:
                        # print(sentence)
                        # if empty dictionary, we add a node with index -1 and context 'Empty'
                        if sentence == {}:
                            HG.add_node(-1, context='Empty')
                        else:
                            # add edge from pred to sub
                            if ('sub' in sentence) and ('pred' in sentence):
                                HG.add_edge(
                                    piece2idx[sentence['pred']], piece2idx[sentence['sub']], type='sub')
                            # add edge from pred to obj
                            if ('obj' in sentence) and (sentence['obj'] != []) and ('pred' in sentence):
                                for obj in sentence['obj']:
                                    HG.add_edge(
                                        piece2idx[sentence['pred']], piece2idx[obj], type='obj')
                            # add edge from pred to time
                            if ('time' in sentence) and ('pred' in sentence):
                                HG.add_edge(
                                    piece2idx[sentence['time']], piece2idx[sentence['pred']], type='time')
                            # add edge from pred to location
                            if ('loc' in sentence) and ('pred' in sentence):
                                HG.add_edge(
                                    piece2idx[sentence['loc']], piece2idx[sentence['pred']], type='loc')
                    # print(HG.edges.data())

                    # store the graph into dict_for_choice (first make it json serializable)
                    dict_for_choice['graph'] = nx.node_link_data(HG)
                    # add it into triples_for_question list
                    graphs_for_question.append(dict_for_choice)

                # add it into output_ques_dict
                output_ques_dict['graphs_for_question'] = graphs_for_question
                # write it into the file
                json.dump(output_ques_dict, fout)
                fout.write('\n')
                # print(output_ques_dict)


def dict2graph_support(input_file, output_file):
    """
    Convert the dictionary which contains openie results to graph.
    - input_file: path to Supports-dict file.
    - output_file: path to Supports graph file which will feed into LSTM and GNN.
    """
    with open(input_file, 'r') as fin:
        print("Processing ", input_file, " ...")
        with open(output_file, 'w') as fout:
            for i, line in enumerate(fin, 1):
                #if i > 1:
                #    break
                # read in the support question dictionary
                support_ques_dict = json.loads(line)
                # construct a question dict for output
                output_ques_dict = {}
                # assign id
                output_ques_dict['id'] = support_ques_dict['id']
                # construct a graphs for question list for output
                graphs_for_question = []

                # get the triples for each choice
                for choice in support_ques_dict['triples_for_question']:
                    # construct a choice dict for output
                    dict_for_choice = {}
                    dict_for_choice['label'] = choice['label']
                    
                    # construct a graphs for choice list for output
                    graphs_for_choice = []
                    
                    # get the triples for each support
                    for support in choice['triples_for_choice']:
                        # construct a support dict for output
                        dict_for_support = {}
                        dict_for_support['eid'] = support['eid']
                        
                        # construct a set containing all the objects appearing in the subject and object in sentences
                        piece2idx = {}  # convert all the pieces of sentences into index
                        index_of_node = 0  # store the index of each node we go through

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
                                if isinstance(piece, str) and (piece not in piece2idx):
                                    piece2idx[piece] = index_of_node
                                    index_of_node += 1
                                if isinstance(piece, list):
                                    for obj in piece:
                                        if obj not in piece2idx:
                                            piece2idx[obj] = index_of_node
                                            index_of_node += 1
                            # print(piece2idx)

                        # Second, construct the directed graph for each support
                        # construct a directed graph for each support
                        SG = nx.DiGraph()
                        # add nodes for the hypothesis graph
                        for content, index in piece2idx.items():
                            SG.add_node(index, context=content)
                        # print(SG.nodes.data())

                        # add edges for each support graph
                        for sentence in support['triples_for_support']:
                            # print(sentence)
                            # if empty dictionary, we add a node with index -1 and context 'Empty'
                            if sentence == {}:
                                SG.add_node(-1, context='Empty')
                            else:
                                # add edge from pred to sub
                                if ('sub' in sentence) and ('pred' in sentence):
                                    SG.add_edge(
                                        piece2idx[sentence['pred']], piece2idx[sentence['sub']], type='sub')
                                # add edge from pred to obj
                                if ('obj' in sentence) and (sentence['obj'] != []) and ('pred' in sentence):
                                    for obj in sentence['obj']:
                                        SG.add_edge(
                                            piece2idx[sentence['pred']], piece2idx[obj], type='obj')
                                # add edge from pred to time
                                if ('time' in sentence) and ('pred' in sentence):
                                    SG.add_edge(
                                        piece2idx[sentence['time']], piece2idx[sentence['pred']], type='time')
                                # add edge from pred to location
                                if ('loc' in sentence) and ('pred' in sentence):
                                    SG.add_edge(
                                        piece2idx[sentence['loc']], piece2idx[sentence['pred']], type='loc')
                        # print(SG.edges.data())

                        # store the graph into dict_for_choice (first make it json serializable)
                        dict_for_support['graph'] = nx.node_link_data(SG)
                        # add it into graph_for_choice list
                        graphs_for_choice.append(dict_for_support)
                        
                    dict_for_choice['graphs_for_choice'] = graphs_for_choice
                    graphs_for_question.append(dict_for_choice)
                
                # add it into output_ques_dict
                output_ques_dict['graphs_for_question'] = graphs_for_question
                # write it into the file
                json.dump(output_ques_dict, fout)
                fout.write('\n')
                #print(output_ques_dict)


if __name__ == "__main__":

    # dict2graph for hypothesis
    # hypo_folders = ["../data/ARC-Hypothesis-networkx-graph",
    #                 "../data/ARC-Hypothesis-networkx-graph/ARC-Easy",
    #                 "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge"]

    # hypo_input_files = ["../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Train-Hypothesis-dict.txt",
    #                     "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Dev-Hypothesis-dict.txt",
    #                     "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Test-Hypothesis-dict.txt",
    #                     "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Train-Hypothesis-dict.txt",
    #                     "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-dict.txt",
    #                     "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Test-Hypothesis-dict.txt"]

    # hypo_output_files = ["../data/ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Train-Hypothesis-networkx-graph.txt",
    #                      "../data/ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Dev-Hypothesis-networkx-graph.txt",
    #                      "../data/ARC-Hypothesis-networkx-graph/ARC-Easy/ARC-Easy-Test-Hypothesis-networkx-graph.txt",
    #                      "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Train-Hypothesis-networkx-graph.txt",
    #                      "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-networkx-graph.txt",
    #                      "../data/ARC-Hypothesis-networkx-graph/ARC-Challenge/ARC-Challenge-Test-Hypothesis-networkx-graph.txt"]
    
    # for folder in hypo_folders:
    #     create_folder(folder)

    # for i in range(len(hypo_input_files)):
    #     dict2graph_hypo(hypo_input_files[i], hypo_output_files[i])

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
    
    for folder in support_folders:
        create_folder(folder)

    for i in range(len(support_input_files)):
        dict2graph_support(support_input_files[i], support_output_files[i])
       

    # test
    # with open("./ARC-Easy-Train-Hypothesis-networkx-graph.txt", 'r') as fin:
    #    for i, line in enumerate(fin, 1):
    #        if i > 1:
    #            break
    #        hypo_ques_dict = json.loads(line)
    #        print(hypo_ques_dict)
