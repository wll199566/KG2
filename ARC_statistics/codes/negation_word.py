"""
This script is for getting the sentence which contains the negation words in ARC_Corpus.txt
"""
import re

input_file = "/scratch/hw1666/KG2/data/ARC-V1-Feb2018-2/ARC_Corpus.txt"

with open(input_file, 'rt') as fin:
    for i, line in enumerate(fin,1):
        negation = re.search(r" no | not | none | neither | never | doesn't | does | isn't | aren't | wasn't | weren't | shouldn't | wouldn't | couldn't | won't | can't | cannot | don't | haven't | hadn't | except | hardly | scarcely | barely ", line, re.IGNORECASE)
        if negation is not None:
            print(i, line)
