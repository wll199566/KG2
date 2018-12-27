"""
This script is for collecting all the supports in one ARC-Support file, 
in order to enable openIE to process it. The resulting file contains one support in one line. 
There is no lable about which line belongs to which hypothesis and which choice, 
so we need to store the number of choice for each hypothesis. 
"""
import os
import json

def get_supports_for_openie(input_file, output_file, output_path):
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
                    for support in question_set['supports']:
                        for text in support["support"]:
                            fout.write(text["text"])
                print(input_file[index], " has ", i, "supports in total.")
            print("Finish processing ", input_file[index])

    print("Finish!")

if __name__ == '__main__':

    input_file = ["/scratch/hw1666/KG2/data/ARC-Supports/ARC-Easy/ARC-Easy-Train-Supports.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Supports/ARC-Easy/ARC-Easy-Dev-Supports.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Supports/ARC-Easy/ARC-Easy-Test-Supports.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Supports/ARC-Challenge/ARC-Challenge-Train-Supports.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Supports/ARC-Challenge/ARC-Challenge-Dev-Supports.jsonl",\
                  "/scratch/hw1666/KG2/data/ARC-Supports/ARC-Challenge/ARC-Challenge-Test-Supports.jsonl"]

    output_file = ["/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Easy/ARC-Easy-Train-Supports-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Easy/ARC-Easy-Dev-Supports-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Easy/ARC-Easy-Test-Supports-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Challenge/ARC-Challenge-Train-Supports-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Challenge/ARC-Challenge-Dev-Supports-split.txt",\
                   "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Challenge/ARC-Challenge-Test-Supports-split.txt"]
    
    output_path = ["/scratch/hw1666/KG2/data/ARC-Supports-split", "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Easy", "/scratch/hw1666/KG2/data/ARC-Supports-split/ARC-Challenge"]

    get_supports_for_openie(input_file, output_file, output_path)