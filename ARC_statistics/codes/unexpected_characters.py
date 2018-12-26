"""
This script is for detecting the characters in the corpus for defining the unexpected characters. First we need to get the statistics for the chars except the space and the english words and numbers.
"""
import re
from operator import itemgetter

input_file = "/scratch/hw1666/KG2/data/ARC-V1-Feb2018-2/ARC_Corpus.txt"
#output_file = "./unexpected_characters_for_ARC_corpus.txt"

# get the total number of characters except space, word and numbers
with open(input_file, 'rt') as fin:
    char_dict = {}  # used for storing the statistics for characters except space, word and numbers
    for i, line in enumerate(fin, 1):
       #unexpected = re.findall(r"[^\w\s,\.?\{\}\[\]\(\)\"\'<>!:;]",line)
        unexpected = re.findall(r"[^\s\w]", line)
        #if len(unexpected) != 0:
           #print(i, unexpected) 
        for char in unexpected:
            if char not in [*char_dict]:
                char_dict[char] = 1
            else:
                char_dict[char] += 1

#print(char_dict)
#print([*char_dict])

# print the count of each character
for key, value in sorted(char_dict.items(), key=itemgetter(0), reverse=True):
    print("{} : {}".format(key, value))

# print the count of such characters whose count is more than 5000
print('-'*100)
print("more than 5000: ")

output_file = "./common_characters.txt"

with open(output_file, 'wt') as fout:
    for  key, value in char_dict.items():
        if value >= 5000:
            print("{} : {}".format(key, value))
            #fout.write(str(key)+" : "+ str(value)+'\n')
            fout.write(key)

             
