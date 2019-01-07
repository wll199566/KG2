"""
This script defines all the functions used for converting openie results for a hypothesis and
support to a dictionary.
"""

import json
import re
import sys

sys.path.append("../")
# print(sys.path)

from utils.file_system_utils import create_folder
from utils.nlp_utils import clean_data

def max_indeces(score_list):
    """
    Function returns a list containing the positions for sentences of max scores.
    - score_list: a list containing scores for each triple of a sentence
    """
    max_score = max(score_list)
    max_idx = [i for i in range(len(score_list)) if score_list[i] == max_score]
    return max_idx


def openie2dict_hypo(openie_file, output_file):
    """
    To convert openie result to dictionary for each question and write into a file.
    The structure of the dictionary contains question, choice, sentence and line level.
    - openie_file: path to the hypothesis openie result file.
    - output_file: path to the output dictionary file.
    """
    # seperate each choice and each question
    with open(openie_file, 'r') as fin:
        print("Processing ", openie_file, "...")
        with open(output_file, 'w') as fout:
            # construct the whole data structure
            # question level
            hypo_graph = []  # list for each question to store all the hypothesis graphs for it
            # choice level
            # triple for each choice, since one hypothesis(choice) may contain several sentences
            triples_for_choice = []
            # sentence level
            # store all generated triples by openie for that hypothesis sentence
            all_triples_for_sentence = []
            triple_scores_for_sentence = []
            # line level
            separator_line = False  # indicator whether the last line before '\n' is separator line

            for i, line in enumerate(fin, 1):
                # print("line", i)
                # construct a dictionary storing all the hypothesis triples for a question
                # if i <= 135 and i >= 105:
                    # print(line)
                # skip all the return line. '^' means the start of the line.
                if re.search(r'^\n', line) is not None:
                    if not separator_line:
                        # select the triple with highest score for each sentence
                        triple_for_sentence = {}  # store the triple with highest score for each sentence
                        # find the maximum score and the corresponding triple
                        # print(i)
                        # print(triple_scores_for_sentence)

                        # if openie cannot give the result for some sentence
                        # we return a empty dictionary for that sentence
                        if triple_scores_for_sentence != []:
                            indeces = max_indeces(triple_scores_for_sentence)
                            # now indeces stores the max scored position
                            # all_triples_for_sentence stores all the scored triples
                            # add subjective and predicate
                            # print(all_triples_for_sentence)
                            # print(indeces)
                            triple_for_sentence['sub'] = clean_data(all_triples_for_sentence[indeces[0]][0])
                            triple_for_sentence['pred'] = clean_data(all_triples_for_sentence[indeces[0]][1])
                            # add the time and location
                            for piece in all_triples_for_sentence[indeces[0]]:
                                if re.search(r'^T:', piece) is not None:
                                    triple_for_sentence['time'] = clean_data(re.sub(
                                        '^T:', '', piece))
                                if re.search(r'^L:', piece) is not None:
                                    triple_for_sentence['loc'] = clean_data(re.sub(
                                        '^L:', '', piece))
                            # add objects
                            # iterate all the max scored sentence triple to get all the objects for that
                            # if there are more than two max openie score, then collect all obj
                            # of course, when there is only one, we can also process it using the following codes
                            obj = []  # store all the objects for a sentece
                            # iterate all the highest scored sentence
                            for index in indeces:
                                # iterate pieces for the highest scored sentence
                                for piece in all_triples_for_sentence[index][2:]:
                                    if re.search(r'^T:', piece) is None and re.search(r'^L:', piece) is None:
                                        obj.append(clean_data(piece))
                                    else:
                                        continue
                            triple_for_sentence['obj'] = obj
                        # add sentence level structure into the choice level structure
                        triples_for_choice.append(triple_for_sentence)

                        # clean the sentence level structures
                        all_triples_for_sentence = []
                        triple_scores_for_sentence = []
                    # else is to handle the case we don't need to do this for separator lines
                    else:
                        continue

                # get the label for each hypothesis graph and push the hypothesis graph into the list
                elif re.search(r'^(-){20}', line) is not None and re.search(r'(-){20}$', line) is not None:
                    # set the sentence_end to be False
                    separator_line = True
                    # get the label for each hypothesis
                    label = re.search(r'(label:)([A-Z0-9a-z])', line).group(2)
                    # add the label to the hypothesis dictionary
                    dict_for_choice = {}
                    dict_for_choice['label'] = label
                    dict_for_choice['triples_for_choice'] = triples_for_choice
                    # add choice level structure into question level structure
                    hypo_graph.append(dict_for_choice)
                    # clean all choice level structure
                    triples_for_choice = []
                    # print(label)
                    # print(line)

                # get the id for each question and construct the question hypothesis dictionray
                elif re.search(r'^(\*){20}', line) is not None and re.search(r'(\*){20}$', line) is not None:
                    ques_id = re.search(r'(id:)([A-Za-z_0-9]+)', line).group(2)
                    # construct dictionary
                    ques_hypo_triples = {}
                    ques_hypo_triples['id'] = ques_id
                    # add all the hypothesis graph into the list of the dictionary
                    ques_hypo_triples['triples_for_question'] = hypo_graph
                    hypo_graph = []  # NOTE to clean the list for the next question
                    # print(ques_id)
                    # print(line)
                    # write the dictionary into the file
                    # print(ques_hypo_triples)
                    json.dump(ques_hypo_triples, fout)
                    fout.write('\n')

                # get the openie triples for each hypothesis
                elif re.search(r'^0\.\d+\s', line) is not None:
                    # print(line)
                    # eliminate the context part
                    if re.search(r'^0\.\d+\sContext', line) is not None:
                        # get the score for each triple
                        score = float(re.search(r'^0\.\d+', line).group())
                        triple_scores_for_sentence.append(score)
                        # print(score)
                        triple = re.search(r':\((.*?)\)$', line).group()
                        # print(triple)
                        # remove ':' at the beginning of it
                        triple = triple[1:]
                        # remove the bracket of the beginning and the end
                        triple = re.sub(r'^\(', r'', triple)
                        triple = re.sub(r'\)$', r'', triple)
                        # print(triple)
                        # NOTE the there is a space here
                        pieces = triple.split('; ')
                        # print(pieces)
                        all_triples_for_sentence.append(pieces)
                    # triple lines by openie without context
                    elif re.search(r'(^0\.\d+\s)\((.+)\)$', line) is not None:
                        # get the score for each triple
                        score = float(re.search(r'^0\.\d+', line).group())
                        triple_scores_for_sentence.append(score)
                        # print(score)
                        # get the triple for each hypothesis
                        triple = re.search(
                            r'(^0\.\d+\s)\((.+)\)$', line).group(2)
                        # print(triple)
                        # NOTE the there is a space here
                        pieces = triple.split('; ')
                        # print(pieces)
                        all_triples_for_sentence.append(pieces)
                    # sentence started by the numbers
                    else:
                        # print(line)
                        continue
                else:
                    # skip all the lines without score
                    separator_line = False    

def openie2dict_support(openie_file, output_file):
    """
    To convert openie result to dictionary for each question and write into a file.
    The structure of the dictionary contains question, choice, support, sentence and line level.
    - openie_file: path to the support openie result file.
    - output_file: path to the output dictionary file.
    """
    # seperate each choice and each question
    with open(openie_file, 'r') as fin:
        print("Processing ", openie_file, "...")
        with open(output_file, 'w') as fout:
            # construct the whole data structure
            # question level
            support_graph = []  # list for each question to store all the support graphs for it
            # choice level
            # triple for each choice, one choice contains several supports
            triples_for_choice = []
            # triples for each support, since one support may contain several sentences
            triples_for_support = []
            # sentence level
            # store all generated triples by openie for that hypothesis sentence
            all_triples_for_sentence = []
            triple_scores_for_sentence = []
            # line level
            separator_line = False  # indicator whether the last line before '\n' is separator line

            for i, line in enumerate(fin, 1):
                # print("line", i)
                # construct a dictionary storing all the hypothesis triples for a question
                # if i <= 135 and i >= 105:
                    # print(line)
                # skip all the return line. '^' means the start of the line.
                if re.search(r'^\n', line) is not None:
                    if not separator_line:
                        # select the triple with highest score for each sentence
                        triple_for_sentence = {}  # store the triple with highest score for each sentence
                        # print(i)
                        # print(triple_scores_for_sentence)

                        # if openie cannot give the result for some sentence
                        # we return a empty dictionary for that sentence
                        if triple_scores_for_sentence != []:
                            # find the maximum score and the corresponding triple
                            indeces = max_indeces(triple_scores_for_sentence)
                            # now indeces stores the max scored position
                            # all_triples_for_sentence stores all the scored triples
                            # add subjective and predicate
                            # print(all_triples_for_sentence)
                            # print(indeces)
                            triple_for_sentence['sub'] = clean_data(all_triples_for_sentence[indeces[0]][0])
                            triple_for_sentence['pred'] = clean_data(all_triples_for_sentence[indeces[0]][1])
                            # add the time and location
                            for piece in all_triples_for_sentence[indeces[0]]:
                                if re.search(r'^T:', piece) is not None:
                                    triple_for_sentence['time'] = clean_data(re.sub(
                                        '^T:', '', piece))
                                if re.search(r'^L:', piece) is not None:
                                    triple_for_sentence['loc'] = clean_data(re.sub(
                                        '^L:', '', piece))
                            # add objects
                            # iterate all the max scored sentence triple to get all the objects for that
                            # if there are more than two max openie score, then collect all obj
                            # of course, when there is only one, we can also process it using the following codes
                            obj = []  # store all the objects for a sentece
                            # iterate all the highest scored sentence
                            for index in indeces:
                                # iterate pieces for the highest scored sentence
                                for piece in all_triples_for_sentence[index][2:]:
                                    if re.search(r'^T:', piece) is None and re.search(r'^L:', piece) is None:
                                        obj.append(clean_data(piece))
                                    else:
                                        continue
                            triple_for_sentence['obj'] = obj
                        # add sentence level structure into the support level structure
                        triples_for_support.append(triple_for_sentence)

                        # clean the sentence level structures
                        all_triples_for_sentence = []
                        triple_scores_for_sentence = []
                    # else is to handle the case we don't need to do this for separator lines
                    else:
                        continue

                # get the label for each hypothesis graph and push the hypothesis graph into the list
                elif re.search(r'^(-){20}', line) is not None and re.search(r'(-){20}$', line) is not None:
                    # set the sentence_end to be False
                    separator_line = True
                    # get the eid for each support
                    eid = re.search(r'(eid:)([A-Z0-9a-z]+)', line).group(2)
                    # add the eid to the support dictionary
                    dict_for_support = {}
                    dict_for_support['eid'] = eid
                    dict_for_support['triples_for_support'] = triples_for_support
                    # add support level structure into choice level structure
                    triples_for_choice.append(dict_for_support)
                    # clean all support level structure
                    triples_for_support = []

                # get the id for each question and construct the question hypothesis dictionray
                elif re.search(r'^(\*){20}', line) is not None and re.search(r'(\*){20}$', line) is not None:
                    # get the label for each hypothesis
                    label = re.search(r'(label:)([A-Z0-9a-z])', line).group(2)
                    # add the label to the hypothesis dictionary
                    dict_for_choice = {}
                    dict_for_choice['label'] = label
                    dict_for_choice['triples_for_choice'] = triples_for_choice
                    # add choice level structure into question level structure
                    support_graph.append(dict_for_choice)
                    # clean all choice level structure
                    triples_for_choice = []
                    # print(label)
                    # print(line)

                elif re.search(r'^(#){20}', line) is not None and re.search(r'(#){20}$', line) is not None:
                    ques_id = re.search(r'(id:)([A-Za-z_0-9]+)', line).group(2)
                    # construct dictionary
                    ques_support_triples = {}
                    ques_support_triples['id'] = ques_id
                    # add all the hypothesis graph into the list of the dictionary
                    ques_support_triples['triples_for_question'] = support_graph
                    support_graph = []  # NOTE to clean the list for the next question
                    # print(ques_id)
                    # print(line)
                    # write the dictionary into the file
                    # print(ques_hypo_triples)
                    json.dump(ques_support_triples, fout)
                    fout.write('\n')

                # get the openie triples for each hypothesis
                elif re.search(r'^0\.\d+\s', line) is not None:
                    # print(line)
                    # eliminate the context part
                    if re.search(r'^0\.\d+\sContext', line) is not None:
                        # get the score for each triple
                        score = float(re.search(r'^0\.\d+', line).group())
                        triple_scores_for_sentence.append(score)
                        # print(score)
                        triple = re.search(r':\((.*?)\)$', line).group()
                        # print(triple)
                        # remove ':' at the beginning of it
                        triple = triple[1:]
                        # remove the bracket of the beginning and the end
                        triple = re.sub(r'^\(', r'', triple)
                        triple = re.sub(r'\)$', r'', triple)
                        # print(triple)
                        # NOTE the there is a space here
                        pieces = triple.split('; ')
                        # print(pieces)
                        all_triples_for_sentence.append(pieces)
                    # triple lines by openie without context
                    elif re.search(r'(^0\.\d+\s)\((.+)\)$', line) is not None:
                        # get the score for each triple
                        score = float(re.search(r'^0\.\d+', line).group())
                        triple_scores_for_sentence.append(score)
                        # print(score)
                        # get the triple for each hypothesis
                        triple = re.search(
                            r'(^0\.\d+\s)\((.+)\)$', line).group(2)
                        # print(triple)
                        # NOTE the there is a space here
                        pieces = triple.split('; ')
                        # print(pieces)
                        all_triples_for_sentence.append(pieces)
                    # sentence started by the numbers
                    else:
                        # print(line)
                        continue

                else:
                    # skip all the lines without score
                    separator_line = False         
                

if __name__ == "__main__":

    # process hypothesis openie files
    hypo_folders = ['../data/ARC-Hypothesis-dict/',
                   '../data/ARC-Hypothesis-dict/ARC-Easy/',
                   '../data/ARC-Hypothesis-dict/ARC-Challenge/']
    
    hypo_openie_files = ["../data/ARC-Hypothesis-graph/ARC-Easy/ARC-Easy-Train-Hypothesis-graph.txt",
                        "../data/ARC-Hypothesis-graph/ARC-Easy/ARC-Easy-Dev-Hypothesis-graph.txt",
                        "../data/ARC-Hypothesis-graph/ARC-Easy/ARC-Easy-Test-Hypothesis-graph.txt",
                        "../data/ARC-Hypothesis-graph/ARC-Challenge/ARC-Challenge-Train-Hypothesis-graph.txt",
                        "../data/ARC-Hypothesis-graph/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-graph.txt",
                        "../data/ARC-Hypothesis-graph/ARC-Challenge/ARC-Challenge-Test-Hypothesis-graph.txt"]
    
    hypo_output_files = ["../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Train-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Dev-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Test-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Train-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-dict.txt",
                        "../data/ARC-Hypothesis-dict/ARC-Challenge/ARC-Challenge-Test-Hypothesis-dict.txt"]

    # construct all the hypothesis graph folders
    for hypo_folder in hypo_folders:
       create_folder(hypo_folder)
    # generate all the hypothesis dict files
    for i in range(len(hypo_openie_files)):
       openie2dict_hypo(hypo_openie_files[i], hypo_output_files[i])

    # process supports openie files
    support_folders = ['../data/ARC-Supports-dict/',
                       '../data/ARC-Supports-dict/ARC-Easy/',
                       '../data/ARC-Supports-dict/ARC-Challenge/']

    support_openie_files = ["../data/ARC-Supports-graph/ARC-Easy/ARC-Easy-Train-Supports-graph.txt",
                         "../data/ARC-Supports-graph/ARC-Easy/ARC-Easy-Dev-Supports-graph.txt",
                         "../data/ARC-Supports-graph/ARC-Easy/ARC-Easy-Test-Supports-graph.txt",
                         "../data/ARC-Supports-graph/ARC-Challenge/ARC-Challenge-Train-Supports-graph.txt",
                         "../data/ARC-Supports-graph/ARC-Challenge/ARC-Challenge-Dev-Supports-graph.txt",
                         "../data/ARC-Supports-graph/ARC-Challenge/ARC-Challenge-Test-Supports-graph.txt"]

    support_output_files = ["../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Train-Supports-dict.txt",
                         "../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Dev-Supports-dict.txt",
                         "../data/ARC-Supports-dict/ARC-Easy/ARC-Easy-Test-Supports-dict.txt",
                         "../data/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Train-Supports-dict.txt",
                         "../data/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Dev-Supports-dict.txt",
                         "../data/ARC-Supports-dict/ARC-Challenge/ARC-Challenge-Test-Supports-dict.txt"]

    # construct all the supports graph folders
    for support_folder in support_folders:
        create_folder(support_folder)
    # generate all the supports dict files
    for i in range(len(support_openie_files)):
        openie2dict_support(support_openie_files[i], support_output_files[i])
