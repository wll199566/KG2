# Construct a dictionary to store the hypothesis sets for each question.

import re
import json

def hypothesis_generator(input_file, output_file): 
    """
    Arguments: 
             input_file: .jsonl file which contains the question and answer
             output_file: .jsonl file which we want to put our hypothesis in 
    Output: 
             .jsonl file which contains all the hypothesis for each question,
             each one is {"id":'', "hypothesis":[], "answerKey":''} 
    """
    with open(input_file, 'rt') as fin_easy_train:
        with open(output_file, 'wt') as fout_easy_train:
            for i, line in enumerate(fin_easy_train,1):
                #print(i)
                attach = False  # records whether we need to attach the answer to the end of the paragraph.
                fill_blank = False  # records whether this question is filling-in-the-blank
                
                # read in every part from the input file.
                hypothesis_sets = {"id":'', "hypothesis":[], "answerKey":''}
                arc_easy_train = json.loads(line) # load a json record (a dictionary)
                hypothesis_sets["id"] = arc_easy_train["id"]
                hypothesis_sets["answerKey"] = arc_easy_train["answerKey"]
                easy_train_paragraph = arc_easy_train["question"]["stem"]  # get the question paragraph
                
                # Find the last sentence to determine the question type
                #last_sentence = re.search(r"\.[\s\w*\.\?,]+$|[^\.]+$|[^\.]+___\.", easy_train_paragraph, re.IGNORECASE)  # find the last sentence of the question
                last_sentence = re.search(r"\. [^\.\?]+\?$|\. [^\.\?]+\.$|[^\.]+$|[^\.]+___\.$", easy_train_paragraph, re.IGNORECASE)  # find the last sentence of the question
                if last_sentence is None:
                    attach = True
                else:
                    last_sentence = last_sentence.group()  # transfer it into a string
                #last_sentences.append(last_sentence)
                paragraph_last_symbol = re.search(r"[^\.\?]$", easy_train_paragraph, re.IGNORECASE)
                if re.search(r"___\.$", easy_train_paragraph) is not None:
                    question_word = "___"
                    fill_blank = True
                else:
                    question_word = re.search(r"___|identify|what|who|where|when|whose|whom|why|how|which of these|which of those|which of the following|which", last_sentence, re.IGNORECASE)  # get the question word in the last sentence
                #question_word = question_word.group()
                if paragraph_last_symbol is not None:  # in this case, we need to attach the answer candidate into the end of the question.
                    attach = True   
                # in this case, we need to substitute the question word with answer candidates
                #ques_word.append(question_word.group())
                if question_word is None:
                    attach = True
                else:    
                    if not fill_blank:
                        question_word = question_word.group()  # transfer it into a string
                #print(question_word)
                for choice in arc_easy_train["question"]["choices"]:  # arc_easy_train["choices"] is a list
                    one_hypothesis_set = {}  # store the dictionary in hypothesis_sets[hypothesis][]
                    answer_text = choice["text"]
                    answer_text = re.sub(r"\.$", r'', answer_text)  # remove the last period of the answer if it is a sentence
                    answer_label = choice["label"]
                    if attach:  # if satisfy the condition, attach the answer directly to the end of the paragraph
                        hypothesis = easy_train_paragraph + ' ' + answer_text + '.'  # generate the hypothesis
                        hypothesis =  re.sub(r'\.+$', r'.', hypothesis)  # eliminate that there is more than one . at the end.
                    else:
                        last_sentence_sub = re.sub(question_word, answer_text, last_sentence)  # substitute the question word in last sentence with the answer candidate.
                        #print("last_sentence_sub: ",last_sentence_sub)
                        find = re.search(last_sentence, easy_train_paragraph, re.IGNORECASE)
                        if find is None:  # cannot find, so attach!
                            #print('last_sentence: ', last_sentence)
                            #print('easy_train_paragraph', easy_train_paragraph)
                            easy_train_paragraph_sub = easy_train_paragraph + ' ' + answer_text + '.'  # generate the hypothesis
                        else: 
                            easy_train_paragraph_sub = re.sub(last_sentence, last_sentence_sub, easy_train_paragraph)  # substitute the last sentence of the original paragraph with the last sentence with the answer 
                            easy_train_paragraph_sub =  re.sub(r'\?+$', r'.', easy_train_paragraph_sub)
                        hypothesis =  re.sub(r'\.+$', r'.', easy_train_paragraph_sub)  # eliminate that there is more than one ? at the end.
                    one_hypothesis_set["text"] = hypothesis
                    one_hypothesis_set["label"] = answer_label
                    hypothesis_sets["hypothesis"].append(one_hypothesis_set)  # append the hypothesis for one answer candidate to the hypothesis set
                        
                json.dump(hypothesis_sets, fout_easy_train)
                fout_easy_train.write("\n")

if __name__ == '__main__':

    input_files = ["../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl",\
                   "../data/ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Test.jsonl"]
    
    output_files = ["../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Train-Hypothesis.jsonl",\
                    "../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Dev-Hypothesis.jsonl",\
                    "../data/ARC-Hypothesis/ARC-Easy/ARC-Easy-Test-Hypothesis.jsonl",\
                    "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Train-Hypothesis.jsonl",\
                    "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Dev-Hypothesis.jsonl",\
                    "../data/ARC-Hypothesis/ARC-Challenge/ARC-Challenge-Test-Hypothesis.jsonl"]

    for index in range(len(input_files)):
        hypothesis_generator(input_files[index], output_files[index])