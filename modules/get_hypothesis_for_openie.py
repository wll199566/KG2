"""
This script is for collecting all the hypothesis in one ARC-Hypothesis file, 
in order to enable openIE to process it. The resulting file contains one hypothesis in one line. 
There is no lable about which support belongs to which hypothesis and which choice, 
so we need to store the number of choice for each hypothesis. 
"""
import os
import json

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
                raise Exception(output_file[index] + " has existed!")
            with open(output_file[index], 'wt') as fout:
                for i, line in enumerate(fin, 1):
                    question_set = json.loads(line)
                    for hypothesis in question_set['hypothesis']:
                        fout.write(hypothesis["text"])
                        fout.write('\n')
                print(input_file[index], " has ", i, "hypothesis in total.")
            print("Finish processing ", input_file[index])

    print("Finish!")

if __name__ == '__main__':

    input_file = ["/scratch/hw1666/KG2/data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Train-Hypothesis.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Dev-Hypothesis.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Test-Hypothesis.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Train-Hypothesis.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Dev-Hypothesis.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Test-Hypothesis.jsonl"]

    output_file = ["/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Easy/ARC-Easy-Train-Hypothesis-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Easy/ARC-Easy-Dev-Hypothesis-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Easy/ARC-Easy-Test-Hypothesis-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Challenge/ARC-Challenge-Train-Hypothesis-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Challenge/ARC-Challenge-Dev-Hypothesis-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Challenge/ARC-Challenge-Test-Hypothesis-split.txt"]

    output_path = ["/scratch/hw1666/KG2/data/ARC-Hypothesis-split", "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Easy", "/scratch/hw1666/KG2/data/ARC-Hypothesis-split/ARC-Challenge"]

    get_hypothesis_for_openie(input_file, output_file, output_path)