"""
This script is for modifying the ARC-Corpus.txt to make cleaner, according to the rules defined in the paper. Of course, we write the modified version into a new .txt file.
"""

import re

input_file = "../data/ARC-V1-Feb2018-2/ARC_Corpus.txt"
output_file = "../data/ARC-V1-Feb2018-2/ARC_Corpus_clean.txt"

with open(input_file, "rt") as fin:
    with open(output_file, "wt") as fout:
        for line in fin:
            negation = (re.search(r" no | not | none | neither | never | doesn't | does | isn't | aren't | wasn't | weren't | shouldn't | wouldn't | couldn't | won't | can't | cannot | don't | haven't | hadn't | except | hardly | scarcely | barely ", line, re.IGNORECASE) is not None)
            un_char = (re.search(r"[^\s\w,\.\(\):\-“”;\?\*/'<>\[\]—+\{\}\"=’–%‘•!`&#~©\$\^■±\|»′@°…«§×−·→£®\\]", line) is not None)
            length = (len(re.findall(r'\S+', line)) > 60)
            if negation or un_char or length:
            # this is the case that we need to clean the sentence
                continue
            else:
                fout.write(line)
