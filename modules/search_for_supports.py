"""
This script is used for searching the supports for each hypothesis.
The script will read the hypothesis files from ARC_Hypothesis and stores all supports in ARC_Supports folder.
"""
import os, sys
import json
from elasticsearch import Elasticsearch

def search_for_supports(input_files, output_folder, output_files):
    """
    Arguments:
            input_files: path to hypothesis files
            output_folder: names of folders storing the search results
            output_files: path to supports files.
    """
    # construct the output folders
    for folder_name in output_folder:
        if os.path.isdir(folder_name):
            print("The support folder ", folder_name, " has existed!")
        else:
            print("Creating the folder ", folder_name, "...")
            os.mkdir(folder_name)
            print(folder_name, " has been created!")   
    print("All output folders have been created!")
    print('*'*50)        
    
    # Search for the supports for every input file.
    es = Elasticsearch()
    print("Processing...")
    
    if len(input_files) != len(output_files):
        raise Exception("The input and output files are not corresponding!")
    for index in range(len(input_files)):
        with open(input_files[index], 'rt') as fin:
            if os.path.exists(output_files[index]):
                raise Exception(output_files[index] + " has existed!")
            with open(output_files[index], 'wt') as fout:
                for line in fin:
                    support_set = {"id": '', "supports":[]}  # the dictionary to store all the query supports for a problem
                    ARC_hypothesis = json.loads(line)
                    # state the question id
                    support_set["id"] = ARC_hypothesis["id"]
                    # get each hypothesis in a question
                    Hypothesis = ARC_hypothesis["hypothesis"]
                    for hypo in Hypothesis:
                        # get the text and label for each hypothesis
                        hypothesis = hypo["text"]
                        label = hypo["label"]
                        # construct a dict for each hypothesis storing the search result, id in the elasticsearch, and label
                        support = {"support":[], "label":label}
                        # search for the relevant supports using the query hypothesis
                        search_body = {
                            "query": {
                                "match": {"text": hypothesis}
                            }
                        }
                        resp = es.search(index="arc_corpus_clean", body=search_body, size=20)  # get top 20 relevant sentences to store.
                        # get the id and search text for each result of each hypothesis
                        for hit in resp['hits']['hits']:
                            # construct a dict for each search result to store the text and the id in elasticsearch
                            result = {"eid": 0, "text":""}
                            result["eid"] = hit['_id']
                            result["text"] = hit['_source']['text']
                            # store the result into support
                            support["support"].append(result)
                        # store all the results for each hypothesis
                        support_set["supports"].append(support)
                    # write into the output file
                    json.dump(support_set, fout)
                    fout.write("\n")            
        print(input_files[index], "is done!")
        print('\n')
    
    print("Finish!")  

if __name__ == "__main__":

    # Hypothesis files
    input_files = ["../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Train-Hypothesis.jsonl",\
                "../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Dev-Hypothesis.jsonl",\
                "../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Test-Hypothesis.jsonl",\
                "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Train-Hypothesis.jsonl",\
                "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Dev-Hypothesis.jsonl",\
                "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Test-Hypothesis.jsonl"]
    
    # Construct the output folder         
    output_folder = ["../data/ARC-Supports/","../data/ARC-Supports/ARC-Easy/", "../data/ARC-Supports/ARC-Challenge/"]
    
    # Support files
    output_files = ["../data/ARC-Supports/ARC-Easy/ARC-Easy-Train-Supports.jsonl",\
                    "../data/ARC-Supports/ARC-Easy/ARC-Easy-Dev-Supports.jsonl",\
                    "../data/ARC-Supports/ARC-Easy/ARC-Easy-Test-Supports.jsonl",\
                    "../data/ARC-Supports/ARC-Challenge/ARC-Challenge-Train-Supports.jsonl",\
                    "../data/ARC-Supports/ARC-Challenge/ARC-Challenge-Dev-Supports.jsonl",\
                    "../data/ARC-Supports/ARC-Challenge/ARC-Challenge-Test-Supports.jsonl"]

    # search for supports
    search_for_supports(input_files, output_folder, output_files)                