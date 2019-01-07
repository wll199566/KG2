"""
This script is for collecting all the hypothesis in one ARC-Hypothesis file, 
in order to enable openIE to process it. The resulting file contains one hypothesis in one line. 
There is no lable about which support belongs to which hypothesis and which choice, 
so we need to store the number of choice for each hypothesis. 
"""
import os
import json

import sys
sys.path.append("../")

from utils.nlp_utils import lemmatize_sentence, split_into_sentences

def get_hypothesis_for_openie(input_file, output_file, output_path):
    """
    Arguments: 
            input_file: a list containing all hypothesis file names
            output_file: a corresponding list containing all split file names
            output_path: a list containing paths for output files
    """
    # Construct the output folder
    for path in output_path:
        if os.path.isdir(path):
            print("The split folder ", path, "has existed!")
        else:
            print("Creating the folder ", path, "..." )
            os.mkdir(path)
            print(path, " has been created!")
    print("All output folders have been created!")
    print('*'*50)

    # read the supports files and output the supports as one line for one support
    if len(input_file) != len(output_file):
        raise Exception("The input and output files are not correspondent!")
    for index in range(len(input_file)):
        with open(input_file[index], 'rt') as fin:
            if os.path.exists(output_file[index]):
                #raise Exception(output_file[index] + " has existed!")
                print(output_file[index], "has existed!")
            with open(output_file[index], 'wt') as fout:
                for i, line in enumerate(fin, 1):
                    question_set = json.loads(line)
                    id = question_set['id']
                    for hypothesis in question_set['hypothesis']:
                        # split the hypothesis text into sentences
                        # and lemmatize them.
                        sentence_list = split_into_sentences(hypothesis["text"])
                        for k, sentence in enumerate(sentence_list):
                            sentence_list[k] = lemmatize_sentence(sentence)
                        text = '\n'.join(sentence_list)  
                        # write into the output file.  
                        fout.write(text)
                        fout.write('\n')
                        fout.write('\n')
                        fout.write('-'*20 + 'id:' + id + '-label:' + hypothesis["label"] + '-'*20 + '\n')
                        fout.write('\n')
                    fout.write('*'*20 + 'id:' + id + '*'*20 + '\n')
                    fout.write('\n')    
                print(input_file[index], " has ", i, "hypothesis in total.")
            print("Finish processing ", input_file[index])

    print("Finish!")

if __name__ == '__main__':

    input_file = ["../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Train-Hypothesis.jsonl",\
                  "../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Dev-Hypothesis.jsonl",\
                  "../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Test-Hypothesis.jsonl",\
                  "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Train-Hypothesis.jsonl",\
                  "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Dev-Hypothesis.jsonl",\
                  "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Test-Hypothesis.jsonl"]

    output_file = ["../data/ARC-Hypothesis-split/ARC-Easy/ARC-Easy-Train-Hypothesis-split.txt",\
                   "../data/ARC-Hypothesis-split/ARC-Easy/ARC-Easy-Dev-Hypothesis-split.txt",\
                   "../data/ARC-Hypothesis-split/ARC-Easy/ARC-Easy-Test-Hypothesis-split.txt",\
                   "../data/ARC-Hypothesis-split/ARC-Challenge/ARC-Challenge-Train-Hypothesis-split.txt",\
                   "../data/ARC-Hypothesis-split/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-split.txt",\
                   "../data/ARC-Hypothesis-split/ARC-Challenge/ARC-Challenge-Test-Hypothesis-split.txt"]

    output_path = ["../data/ARC-Hypothesis-split", "../data/ARC-Hypothesis-split/ARC-Easy", "../data/ARC-Hypothesis-split/ARC-Challenge"]

    get_hypothesis_for_openie(input_file, output_file, output_path)