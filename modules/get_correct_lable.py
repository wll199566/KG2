"""
This script is to get all the correct label and transfer them to 
the integer for ARC questions.
"""
import json

import sys
sys.path.append("../")
from utils.file_system_utils import create_folder


def get_correct_label(input_file, output_file):
    """
    Args:
        input_file: path to the question file, 
                    like "../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl"
        output_file: path to the output file,
                    like "../data/ARC-labels/ARC-Easy-Dev.json"             
    """
    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            print("Processing", input_file, "...")
            for line in fin:
                dict_for_ques = json.loads(line)
                # construct label to index dictionary
                label2index = {}  # look up dictionary for labels
                for i, choice in enumerate(dict_for_ques["choices"]):
                    label2index[choice["label"]] = i 
                # construct the output dictionary
                dict_for_ans = {}
                dict_for_ans["id"] = dict_for_ques["id"]
                dict_for_ans["num_of_choices"] = len(dict_for_ques["question"]["choices"])
                dict_for_ans["answerKey"] = label2index[dict_for_ques["answerKey"]]
                # write to the output file
                json.dump(dict_for_ans, fout)
                fout.write('\n')

if __name__ == "__main__":

    output_folders = ["../data/ARC-Labels", "../data/ARC-Labels/ARC-Easy", "../data/ARC-Labels/ARC-Challenge"]

    input_files = ["../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"]
    
    output_files = ["../data/ARC-Labels/ARC-Easy/ARC-Easy-Train-Labels.jsonl",\
                    "../data/ARC-Labels/ARC-Easy/ARC-Easy-Dev-Labels.jsonl",\
                    "../data/ARC-Labels/ARC-Easy/ARC-Easy-Test-Labels.jsonl",\
                    "../data/ARC-Labels/ARC-Challenge/ARC-Challenge-Train-Labels.jsonl",\
                    "../data/ARC-Labels/ARC-Challenge/ARC-Challenge-Dev-Labels.jsonl",\
                    "../data/ARC-Labels/ARC-Challenge/ARC-Challenge-Test-Labels.jsonl"] 
    
    for output_folder in output_folders:
        create_folder(output_folder)

    for i, input_file in enumerate(input_files):
        get_correct_label(input_file, output_files[i])    

