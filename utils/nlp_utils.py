"""
This script contains all the NLP helper functions.
"""
import numpy as np
import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk.data

def clean_data(string):
    """
     Argument:
             string: the string to be cleaned
    """
    string = re.sub(r"[^A-Za-z0-9,“”;?!=×−±:\.\(\)\-\*\/\'\<\>\[\]\—\+\{\}\"\’\–\%\'\`\|]", " ", string)     
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


def nltk2wn_tag(nltk_tag):
    """
    Convert nltk tag to wordnet tag.
    This function is the helper for lemmatize_sentence.
    """
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    """
    Args:
        sentence: sentence string to be lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))

    return " ".join(res_words) 


def split_into_sentences(string):
    """
    Split a string into sentences.
    Args:
        string: a string to be split.
    Output:
        a list containing each sentence.    
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #print ('\n'.join(tokenizer.tokenize(sentence)))
    return tokenizer.tokenize(string)
    

def load_glove(glove_path):
    """
    Argument: 
             glove_path: the path storing the downloaded glove.6B.50d.txt
    Output:
             a dictionary like {key: word, value: glove vector}         
    """
    # read in the whole glove and store as a dictionay
    # {key: word, value: glove vector}
    with open(glove_path, 'r') as fin:
        glove = {line.split()[0]: np.fromiter(map(float, line.split()[1:]),dtype=np.float) 
             for line in fin}

    return glove    



if __name__ == "__main__":

    #sentence = "my mom said {[(“I have a good son?!”)]}?!=×−±:|#$%^&*&("
    sentence = "I am a good boy."
    sentence_cleaned = clean_data(sentence)
    print(sentence_cleaned)    

    #string = "Florida has a large supply of phosphate that ancient seas deposited millions of years ago. The phosphate contains the remains of animals that were deposited in layers on the sea floor. intrusive igneous type of rock did the phosphate become."
    #string = "I am a good boy."
    #sentence_list = split_into_sentences(string)
    #print(sentence_list)
    #for i, sentence in enumerate(sentence_list):
    #    sentence_list[i] = lemmatize_sentence(sentence)
    #print('\n'.join(sentence_list))    