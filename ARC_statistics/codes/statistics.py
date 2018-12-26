"""
This script is for getting the statistics for the ARC Corpus. Since we need to define the 'too long'supporting facts according to KG2 paper, we store the length of every facts into the array and then analyze them.
"""
import re
import numpy as np
import json

input_file = "/scratch/hw1666/KG2/data/ARC-V1-Feb2018-2/ARC_Corpus.txt"
#output_file = "./statistic_for_ARC_corpus.txt"
length_of_sentences = []  # store the length of every sentence in the corpus.
with open(input_file, 'rt') as fin:
    for line in fin:
        count = len(re.findall(r'\S+', line)) # the number of the words
        length_of_sentences.append(count)

length_mean = np.mean(length_of_sentences)
length_median = np.median(length_of_sentences)
length_std = np.std(length_of_sentences)
length_max = np.max(length_of_sentences)
length_min = np.min(length_of_sentences)

statistics_dict = {"max":length_max, "min":length_min, "mean": length_mean, "median": length_median, "standrd variance": length_std}

print(statistics_dict)

#with open(output_file, 'wt') as fout:
#    json.dump(statistics_dict, fout)             
