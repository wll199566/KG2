"""
This script defines all the functions used for converting openie results for a hypothesis and 
support to a dictionary.
"""

import json
import re
import sys

sys.path.append("../")
#print(sys.path)

from utils.file_system_utils import create_folder 

def max_indeces(score_list):
    """
    Function returns a list containing the positions for sentences of max scores.
    - score_list: a list containing scores for each triple of a sentence 
    """
    max_score = max(score_list)
    max_idx = [i for i in range(len(score_list)) if score_list[i]==max_score]
    return max_idx

def openie2dict_hypo(openie_file, output_file):
    """
    To convert openie result to dictionary for each question and write into a file.
    The structure of the dictionary contains question, choice, sentence and line level.
    - openie_file: path to the hypothesis openie result file.
    """
    # seperate each choice and each question
    with open(openie_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            # construct the whole data structure
            # question level
            hypo_graph = []  # list for each question to store all the hypothesis graphs for it 
            # choice level
            triples_for_choice = []  # triple for each choice, since one hypothesis(choice) may contain several sentences
            # sentence level
            all_triples_for_sentence = []  # store all generated triples by openie for that hypothesis sentence
            triple_scores_for_sentence = []
            # line level
            separator_line = False  # indicator whether the last line before '\n' is separator line
            
            for i, line in enumerate(fin, 1):
                #print("line", i)
                # construct a dictionary storing all the hypothesis triples for a question
                #if i <= 135 and i >= 105:
                    #print(line)
                # skip all the return line. '^' means the start of the line.
                if re.search(r'^\n', line) is not None:
                    if not separator_line:
                        # select the triple with highest score for each sentence
                        triple_for_sentence = {} # store the triple with highest score for each sentence
                        # find the maximum score and the corresponding triple
                        #print(i)
                        #print(triple_scores_for_sentence)
                    
                        # if openie cannot give the result for some sentence
                        # we return a empty dictionary for that sentence
                        if triple_scores_for_sentence != []:
                            indeces = max_indeces(triple_scores_for_sentence)
                            # now indeces stores the max scored position
                            # all_triples_for_sentence stores all the scored triples
                            # add subjective and predicate
                            #print(all_triples_for_sentence)
                            #print(indeces)
                            triple_for_sentence['sub'] = all_triples_for_sentence[indeces[0]][0]
                            triple_for_sentence['pred'] = all_triples_for_sentence[indeces[0]][1]
                            # add the time and location
                            for piece in all_triples_for_sentence[indeces[0]]:
                                if re.search(r'^T:', piece) is not None:
                                    triple_for_sentence['time'] = re.sub('^T:', '', piece)
                                if re.search(r'^L:', piece) is not None:
                                    triple_for_sentence['loc'] = re.sub('^L:', '', piece)
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
                                        obj.append(piece)
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
                    
                # skip all the lines without score
                if re.search(r'^[a-zA-Z]', line) is not None:
                    separator_line = False
                
                # get the label for each hypothesis graph and push the hypothesis graph into the list
                if re.search(r'^(-){20}', line) is not None and re.search(r'(-){20}$', line) is not None:
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
                    #print(label)
                    #print(line)
                
                # get the id for each question and construct the question hypothesis dictionray
                if re.search(r'^(\*){20}', line) is not None and re.search(r'(\*){20}$', line) is not None:
                    ques_id = re.search(r'(id:)([A-Za-z_0-9]+)', line).group(2)
                    # construct dictionary
                    ques_hypo_triples = {}
                    ques_hypo_triples['id'] = ques_id
                    # add all the hypothesis graph into the list of the dictionary
                    ques_hypo_triples['triples_for_question'] = hypo_graph
                    hypo_graph = []  # NOTE to clean the list for the next question
                    #print(ques_id)
                    #print(line)  
                    # write the dictionary into the file
                    #print(ques_hypo_triples)
                    json.dump(ques_hypo_triples, fout)
                    fout.write('\n')
                
                # get the openie triples for each hypothesis    
                if re.search(r'^0\.\d+\s', line) is not None:
                    #print(line)
                    # eliminate the context part
                    if re.search(r'^0\.\d+\sContext', line) is not None:
                        # get the score for each triple
                        score = float(re.search(r'^0\.\d+', line).group())
                        triple_scores_for_sentence.append(score)
                        #print(score)
                        triple = re.search(r':\((.*?)\)$', line).group()
                        #print(triple)
                        triple = triple[1:]  # remove ':' at the beginning of it 
                        # remove the bracket of the beginning and the end
                        triple = re.sub(r'^\(', r'', triple)
                        triple = re.sub(r'\)$', r'', triple)
                        #print(triple)
                        pieces = triple.split('; ') # NOTE the there is a space here
                        #print(pieces)
                        all_triples_for_sentence.append(pieces)
                    # triple lines by openie without context
                    elif re.search(r'(^0\.\d+\s)\((.+)\)$', line) is not None: 
                        # get the score for each triple
                        score = float(re.search(r'^0\.\d+', line).group())
                        triple_scores_for_sentence.append(score)
                        #print(score)   
                        # get the triple for each hypothesis
                        triple = re.search(r'(^0\.\d+\s)\((.+)\)$', line).group(2)
                        #print(triple)
                        pieces = triple.split('; ') # NOTE the there is a space here
                    # sentence started by the numbers
                    else:
                        #print(line)
                        continue
                    #print(pieces)
                    all_triples_for_sentence.append(pieces)

                else:
                    pass
                #    print(line)    

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

    hypo_output_files =  ["../data/ARC-Hypothesis-dict/ARC-Easy/ARC-Easy-Train-Hypothesis-dict.txt",
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
